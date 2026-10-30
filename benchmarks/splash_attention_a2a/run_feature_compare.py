"""Bitwise fwd + bwd comparison of one fused a2a-cp feature against native.

Baseline: native Splash after the Ulysses pre-kernel all-to-all, i.e. inside
a ``shard_map`` over the cp axis: q/k/v all-to-all to head sharding, native
kernel on the local heads (head-grouped operands sliced to the local rows), o
all-to-all back.  Gradients come from autodiff through that graph.
Candidate: the fused a2a-cp kernel on the sequence shards.

Both sides get the same q/k/v/dO bytes; o, logsumexp, max_logits and
dq/dk/dv are compared bit for bit on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = (
    "numpy_mask", "dynamic_mask", "gqa", "grouped_ids", "block_mask_2d",
    "native_layout", "unfused_reciprocal", "sinks", "soft_cap", "max_logit",
)
# Dense masks are materialised as [N, N]; keep those cases small.
MASK_TOKENS = 8192


def _tile_summary(ids, bq, bkv):
  """Active/full tiles of segment-only attention for monotonic segment IDs."""
  import numpy as np
  q_lo, q_hi = ids[:, ::bq], ids[:, bq - 1::bq]
  k_lo, k_hi = ids[:, ::bkv], ids[:, bkv - 1::bkv]
  active = ((q_lo[:, :, None] <= k_hi[:, None, :])
            & (k_lo[:, None, :] <= q_hi[:, :, None]))
  full = ((q_lo == q_hi)[:, :, None] & (k_lo == k_hi)[:, None, :]
          & (q_lo[:, :, None] == k_lo[:, None, :]))
  return active, full


def _grouped_mask_info(sk, jnp, np, ids, bq, bkv, backward):
  """[groups, grid] block mask over the union grid (as the grouped tests)."""
  active, full = _tile_summary(ids, bq, bkv)
  if backward:
    active, full = active.swapaxes(-1, -2), full.swapaxes(-1, -2)
  union = active.any(axis=0)
  indices = np.argwhere(union)
  rows, cols = np.full((2, union.size), -1, np.int32)
  rows[:len(indices)], cols[:len(indices)] = indices.T
  kinds = np.zeros((len(ids), union.size), np.int8)
  kinds[:, :len(indices)] = np.where(
      active[:, indices[:, 0], indices[:, 1]],
      np.where(full[:, indices[:, 0], indices[:, 1]], 2, 1), 0)
  return sk.MaskInfo(
      mask_next=jnp.full((union.size,), -1, jnp.int8),
      active_rows=jnp.asarray(rows), active_cols=jnp.asarray(cols),
      block_mask=jnp.asarray(kinds),
      num_active_blocks=jnp.array([len(indices)], jnp.int32),
      partial_mask_blocks=None, q_sequence=None,
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--cp", type=int, default=8)
  parser.add_argument("--case", required=True, choices=CASES)
  parser.add_argument("--tokens", type=int, default=32768)
  parser.add_argument("--heads", type=int, default=16)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)

  from config import production_config
  from inputs import make_segment_ids, make_sharded_arrays
  from profile_common import set_profile_flags

  set_profile_flags(args.out)
  import jax
  from jax import lax
  import jax.numpy as jnp
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_a2a_cp_kernel as a2a,
  )
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_kernel as sk,
  )
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_mask as sm,
  )

  case, cp = args.case, args.cp
  n = MASK_TOKENS if case in ("numpy_mask", "dynamic_mask") else args.tokens
  heads = args.heads
  kv_heads = heads // 2 if case == "gqa" else heads
  base_cfg = production_config(global_heads=heads, cp_size=cp)
  dqk, dv = base_cfg["qk_head_dim"], base_cfg["v_head_dim"]
  axis = base_cfg["mesh_axis"]
  heads_local = heads // cp

  values = dict(base_cfg["splash"])
  if case == "native_layout":
    values.update(q_layout=2, k_layout=2, v_layout=2, max_logit_const=0.0,
                  use_base2_exp=True, combine_log2_scale=True)
  if case == "unfused_reciprocal":
    values["fuse_reciprocal"] = False
  if case == "soft_cap":
    values["attn_logits_soft_cap"] = 20.0
  for name in ("q_layout", "k_layout", "v_layout"):
    values[name] = sk.QKVLayout(values[name])
  config = sk.SplashConfig(**values)
  bq, bkv = config.block_q, config.block_kv

  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)[:cp]
  if len(devices) != cp:
    raise RuntimeError(f"expected {cp} TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (axis,))

  # Segment IDs: 1023-token segments; grouped cases add a 511-token row.
  ids = make_segment_ids(n, "seg_lt_bkv")
  if case in ("grouped_ids", "block_mask_2d"):
    ids = np.stack([ids, make_segment_ids(n, "seg_lt_bkv_compute")])
  segment_ids = sk.SegmentIds(jnp.asarray(ids), jnp.asarray(ids))

  sinks = max_logit_value = None
  if case == "sinks":
    sinks = jnp.linspace(-1.0, 1.0, heads, dtype=jnp.float32)
  if case == "max_logit":
    max_logit_value = jnp.linspace(4.0, 6.0, heads, dtype=jnp.float32)

  # Kernels.  Both factories take the same mask over the full sequence.
  fused_kw = dict(cp_size=cp, cp_axis=axis)
  if case == "numpy_mask":
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    # Causal band with a ragged left edge: partial tiles on both edges.
    dense = (cols <= rows) & (cols > rows - 3000 - (rows % 7) * 50)
    native = sk.make_splash_mha(dense, config=config, save_residuals=True)
    fused = a2a.make_splash_mha_a2a_cp(dense, config=config,
                                       save_residuals=True, **fused_kw)
  elif case == "dynamic_mask":
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    dense = (cols <= rows) & (cols > rows - 3000 - (rows % 7) * 50)
    native = sk.make_dynamic_splash_mha(dense, config=config,
                                        save_residuals=True)
    fused = a2a.make_dynamic_splash_mha_a2a_cp(
        dense, config=config, save_residuals=True, **fused_kw)
  elif case == "block_mask_2d":
    kw = dict(config=config, is_mqa=False, save_residuals=True,
              mask_value=sk.base.DEFAULT_MASK_VALUE, mask_function=None,
              fwd_mask_sparsity=1.0, dkv_mask_sparsity=1.0)
    fwd_info = _grouped_mask_info(sk, jnp, np, ids, bq, bkv, False)
    dkv_info = _grouped_mask_info(
        sk, jnp, np, ids, config.block_q_dkv, config.block_kv_dkv, True)
    native = sk.SplashAttentionKernel(fwd_info, dkv_info, **kw)
    fused = a2a.SplashAttentionA2ACpKernel(fwd_info, dkv_info, **kw,
                                           **fused_kw)
  else:
    mask = (sm.FullMask((n, n)) if case == "native_layout"
            else sm.CausalMask((n, n)))
    native = sk.make_splash_mha(mask, config=config, save_residuals=True)
    fused = a2a.make_splash_mha_a2a_cp(mask, config=config,
                                       save_residuals=True, **fused_kw)

  seq = P(None, axis, None)
  out_specs = (seq, P(axis, None))
  call_kw = {} if max_logit_value is None else {"max_logit_value": max_logit_value}

  def local_rows(x):
    rank = lax.axis_index(axis)
    return a2a._local_groups(x, rank, heads, heads_local)

  def head_slice(x):
    rank = lax.axis_index(axis)
    return lax.dynamic_slice_in_dim(x, rank * heads_local, heads_local)

  @jax.shard_map(mesh=mesh, in_specs=(seq, seq, seq), out_specs=out_specs,
                 check_vma=False)
  def base_fn(q, k, v):
    to_heads = lambda x: lax.all_to_all(x, axis, 0, 1, tiled=True)
    q, k, v = map(to_heads, (q, k, v))
    kernel, seg = native, segment_ids
    if ids.ndim == 2:
      seg = sk.SegmentIds(local_rows(seg.q), local_rows(seg.kv))
    if case == "block_mask_2d":
      kernel = sk.SplashAttentionKernel(
          native.fwd_mask_info._replace(
              block_mask=local_rows(native.fwd_mask_info.block_mask)),
          native.dkv_mask_info._replace(
              block_mask=local_rows(native.dkv_mask_info.block_mask)),
          **native.kwargs)
    kw = {k_: head_slice(v_) for k_, v_ in call_kw.items()}
    out, stats = kernel(q, k, v, seg,
                        None if sinks is None else head_slice(sinks), **kw)
    return lax.all_to_all(out, axis, 1, 0, tiled=True), stats

  @jax.shard_map(mesh=mesh, in_specs=(seq, seq, seq), out_specs=out_specs,
                 check_vma=False)
  def fused_fn(q, k, v):
    return fused(q, k, v, segment_ids, sinks, **call_kw)

  def with_grads(fn):
    @jax.jit
    def run(q, k, v, do):
      (out, stats), vjp = jax.vjp(fn, q, k, v)
      grads = vjp((do, jax.tree.map(jnp.zeros_like, stats)))
      return out, stats, grads
    return run

  specs = [
      jax.ShapeDtypeStruct((heads, n, dqk), jnp.bfloat16),
      jax.ShapeDtypeStruct((kv_heads, n, dqk), jnp.bfloat16),
      jax.ShapeDtypeStruct((kv_heads, n, dv), jnp.bfloat16),
      jax.ShapeDtypeStruct((heads, n, dv), jnp.bfloat16),
  ]
  sharding = NamedSharding(mesh, seq)
  inputs = make_sharded_arrays(jax, specs, (sharding,) * 4, devices)

  def host(tree):
    return jax.tree.map(np.asarray, tree)

  # One side at a time keeps the HBM footprint of the larger cases bounded.
  results = {}
  for side, fn in (("base", base_fn), ("fused", fused_fn)):
    out, stats, grads = jax.block_until_ready(with_grads(fn)(*inputs))
    results[side] = {
        "o": np.asarray(out),
        "logsumexp": np.asarray(stats["logsumexp"]),
        "max_logits": (None if stats.get("max_logits") is None
                       else np.asarray(stats["max_logits"])),
        "dq": np.asarray(grads[0]), "dk": np.asarray(grads[1]),
        "dv": np.asarray(grads[2]),
    }
    del out, stats, grads

  report = {"case": case, "cp": cp, "tokens": n, "heads": heads,
            "kv_heads": kv_heads,
            "ids_sha256": hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()}
  all_equal = True
  for name in ("o", "logsumexp", "max_logits", "dq", "dk", "dv"):
    b, f = results["base"][name], results["fused"][name]
    if b is None or f is None:
      entry = {"present": [b is not None, f is not None]}
      all_equal &= b is None and f is None
    elif b.shape != f.shape or b.dtype != f.dtype:
      entry = {"shape": [list(b.shape), list(f.shape)],
               "dtype": [str(b.dtype), str(f.dtype)]}
      all_equal = False
    else:
      bits = np.uint16 if b.dtype.itemsize == 2 else np.uint32
      differs = b.view(bits) != f.view(bits)
      num = int(np.count_nonzero(differs))
      max_abs = 0.0
      if num:
        max_abs = float(np.max(np.abs(
            b[differs].astype(np.float32) - f[differs].astype(np.float32))))
      entry = {
          "num_bit_diff": num, "max_abs_diff": max_abs,
          "max_abs_ref": float(np.max(np.abs(b.astype(np.float32)))),
          "sha256_base": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(),
          "sha256_fused": hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest(),
      }
      all_equal &= num == 0
    report[name] = entry
  report["bitwise_equal"] = bool(all_equal)
  args.out.joinpath("feature_compare.json").write_text(
      json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
