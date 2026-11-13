"""Abstract checks of the a2a-cp Splash backward input exchanges (CPU tracing only).

Each ``bwd_input_exchange`` must give the fused backward head-layout q/k/v
without re-running the forward kernel under a remat policy that saves only
named residuals. Device numerics are validated separately.
"""

import dataclasses
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.ad_checkpoint import saved_residuals
from jax.sharding import Mesh, PartitionSpec as P
from tokamax._src.ops.experimental.tpu.splash_attention import mla_fused_backward
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_a2a_cp_kernel as a2a
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as sm

CP, HEADS, SEQ = 8, 32, 65536
NAMES = ("mla_q_head", "mla_k_head", "mla_v_head")
RESIDUALS = "mla_splash_residuals"
# (bwd_input_exchange, head_checkpoint_names, names saved by the policy).
CASES = [
    ("head", None, (RESIDUALS,)),
    ("head", NAMES, (RESIDUALS, *NAMES)),
    ("kv_hbm_prefetch", None, (RESIDUALS,)),
    ("a2a", None, (RESIDUALS,)),
]
IDS = ["head", "head-named", "kv_hbm_prefetch", "a2a"]


def config():
  return sk.SplashConfig(
      block_q=4096,
      block_kv=2048,
      block_kv_compute=512,
      block_q_dkv=512,
      block_kv_dkv=4096,
      block_kv_dkv_compute=512,
      use_fused_bwd_kernel=True,
      dq_reduction_steps=3,
      q_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      k_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      bwd_cast_before_transpose=True,
      use_base2_exp=False,
      softmax_scale=0.07216878364870322,
      residual_checkpoint_name=RESIDUALS,
      use_experimental_scheduler=True,
  )


def attention(exchange, names, saved):
  kernel = a2a.make_splash_mha_a2a_cp(
      sm.CausalMask((SEQ, SEQ)),
      config=config(),
      cp_size=CP,
      cp_axis="context",
      bwd_input_exchange=exchange,
      head_checkpoint_names=names,
  )
  mesh = Mesh(np.asarray(jax.devices()[:CP]), ("context",))
  mapped = jax.shard_map(
      lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)),
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(None),),
      out_specs=P(None, "context", None),
      check_vma=False,
  )
  return jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(*saved))


def inputs():
  args = [jax.ShapeDtypeStruct((HEADS, SEQ, d), jnp.bfloat16) for d in (192, 192, 128)]
  return args + [jax.ShapeDtypeStruct((SEQ,), jnp.int32)]


def cycle(fn):
  def run(q, k, v, ids, do):
    out, pullback = jax.vjp(lambda q, k, v: fn(q, k, v, ids), q, k, v)
    return out, *pullback(do)

  return run


def live_forward_kernels(text):
  """Forward custom calls whose results are used in their StableHLO function."""
  live = 0
  for body in re.split(r"\n(?=\s*func\.func)", text):
    lines = body.splitlines()
    for i, line in enumerate(lines):
      if "tpu_custom_call" in line and "fwd_a2a_cp" in line:
        ssa = re.match(r"\s*(%[\w.]+)(?::\d+)?\s*=", line)[1]
        pat = re.compile(re.escape(ssa) + r"(?:#\d+)?(?![\w.])")
        live += any(pat.search(x) for j, x in enumerate(lines) if j != i)
  return live


def backward_all_to_alls(exchange, names):
  """dO always; o unless head re-runs the forward kernel; q for kv_hbm_prefetch; q/k/v for a2a."""
  if exchange == "head":
    # Unnamed: Di uses the re-run forward kernel's head-layout o, as Tokamax f0686ad does.
    return 1 if names is None else 2
  return {"kv_hbm_prefetch": 3, "a2a": 5}[exchange]


requires_devices = pytest.mark.skipif(len(jax.devices()) < CP, reason="Run with 16 logical CPU devices")


@requires_devices
@pytest.mark.parametrize("exchange,names,saved", CASES, ids=IDS)
def test_saved_residuals(exchange, names, saved):
  with jax.ensure_compile_time_eval():
    fn = attention(exchange, names, saved)
  found = sorted(
      (str(aval.dtype), math.prod(aval.shape), origin.startswith("from the argument"))
      for aval, origin in saved_residuals(fn, *inputs())
      if origin != "from a constant"
  )
  arguments = [("bfloat16", HEADS * SEQ * d, True) for d in (192, 192, 128)] + [("int32", SEQ, True)]
  # The local heads' logsumexp, gathered over CP, and the sequence-layout o,
  # except for unnamed head buffers: that backward takes o from the re-run
  # forward kernel, as Tokamax f0686ad does.
  splash = [("float32", HEADS * SEQ, False)]
  if exchange != "head" or names is not None:
    splash.append(("bfloat16", HEADS * SEQ * 128, False))
  head_qkv = [("bfloat16", HEADS * SEQ * d, False) for d in (192, 192, 128)] if names else []
  assert found == sorted(arguments + splash + head_qkv)


@requires_devices
@pytest.mark.parametrize("exchange,names,saved", CASES, ids=IDS)
def test_backward_exchanges_and_forward_recompute(exchange, names, saved):
  with jax.ensure_compile_time_eval():
    run = cycle(attention(exchange, names, saved))
  args = inputs() + [jax.ShapeDtypeStruct((HEADS, SEQ, 128), jnp.bfloat16)]
  program = jax.make_jaxpr(run)(*args)
  outvars = [v.aval.shape for v in program.jaxpr.outvars]
  assert outvars == [(HEADS, SEQ, 128), (HEADS, SEQ, 192), (HEADS, SEQ, 192), (HEADS, SEQ, 128)]
  text = str(program)
  assert text.count("name=mla_fused_dq_dkv_a2a") == 1
  assert text.count("all_to_all[") == backward_all_to_alls(exchange, names)
  hlo = jax.jit(run).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
  # Unnamed head buffers can only be rebuilt by re-running the forward kernel.
  assert live_forward_kernels(hlo) == (2 if exchange == "head" and names is None else 1)
  # dq keeps the kernel's head-dimension-minor layout; XLA chooses the
  # downstream projection layout.
  dq_type = f"tensor<{HEADS}x{SEQ // CP}x192xbf16>"
  assert not any("@LayoutConstraint" in line and dq_type in line for line in hlo.splitlines())


def test_factory_under_jit_checks_concrete_metadata(monkeypatch):
  """MaxText builds the kernel while tracing; the schedule check still reads NumPy metadata."""
  results = []
  validate = mla_fused_backward.validate_metadata

  def recording(*args, **kwargs):
    results.append(validate(*args, **kwargs))
    return results[-1]

  monkeypatch.setattr(mla_fused_backward, "validate_metadata", recording)

  @jax.jit
  def build(x):
    a2a.make_splash_mha_a2a_cp(sm.CausalMask((SEQ, SEQ)), config=config(), cp_size=CP, cp_axis="context")
    return x

  build(jnp.zeros(()))
  assert results and all(r is not None and r["concrete"] for r in results)


@pytest.mark.parametrize(
    "exchange,names",
    [("x", None), ("remat", None), ("kv_hbm_prefetch", NAMES), ("a2a", NAMES), ("head", NAMES[:2]), ("head", (1, 2, 3))],
)
def test_invalid_backward_input_exchange_rejected(exchange, names):
  with pytest.raises(ValueError, match="bwd_input_exchange|head_checkpoint_names"):
    a2a.make_splash_mha_a2a_cp(
        sm.CausalMask((SEQ, SEQ)),
        config=config(),
        cp_size=CP,
        cp_axis="context",
        bwd_input_exchange=exchange,
        head_checkpoint_names=names,
    )


# (CP, heads per sequence, local batch, block_q_dkv, block_kv_dkv). A local batch
# is folded into the head axis, [B, H, S/CP, D] -> [B*H, S/CP, D], with one
# segment-ID row per sequence. Tiny-VL is CP8, 16 heads, two sequences per
# FSDP shard and 2048/2048 dKV tiles; CP2 with four sequences covers several
# sequences on one rank.
FOLDED = [(8, 16, 2, 2048, 2048), (8, 16, 2, 512, 4096), (8, 8, 4, 2048, 2048), (2, 4, 4, 2048, 2048)]


@requires_devices
@pytest.mark.parametrize("cp,heads,batch,bq,bkv", FOLDED)
@pytest.mark.parametrize("exchange", ["head", "kv_hbm_prefetch", "a2a"])
def test_batch_folded_into_heads(cp, heads, batch, bq, bkv, exchange):
  folded = batch * heads
  cfg = dataclasses.replace(config(), block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=512)
  with jax.ensure_compile_time_eval():
    kernel = a2a.make_splash_mha_a2a_cp(
        sm.CausalMask((SEQ, SEQ)), config=cfg, cp_size=cp, cp_axis="context", bwd_input_exchange=exchange
    )
  mesh = Mesh(np.asarray(jax.devices()[:cp]), ("context",))
  mapped = jax.shard_map(
      lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)),
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(None, None),),
      out_specs=P(None, "context", None),
      check_vma=False,
  )
  fn = jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(RESIDUALS))
  args = [jax.ShapeDtypeStruct((folded, SEQ, d), jnp.bfloat16) for d in (192, 192, 128)]
  args += [jax.ShapeDtypeStruct((batch, SEQ), jnp.int32), jax.ShapeDtypeStruct((folded, SEQ, 128), jnp.bfloat16)]
  program = jax.make_jaxpr(cycle(fn))(*args)
  assert [v.aval.shape for v in program.jaxpr.outvars] == [
      (folded, SEQ, 128), (folded, SEQ, 192), (folded, SEQ, 192), (folded, SEQ, 128)
  ]
  text = str(program)
  assert text.count("name=mla_fused_dq_dkv_a2a") == 1
  assert text.count("all_to_all[") == backward_all_to_alls(exchange, None)
  jax.jit(cycle(fn)).trace(*args).lower(lowering_platforms=("tpu",))


@pytest.mark.parametrize("cp,heads,batch", [(8, 16, 3), (16, 4, 3)])
def test_batch_fold_rejects_sequences_straddling_ranks(cp, heads, batch):
  # Three 16-head sequences over CP8 put 6 folded heads on a rank, so a rank
  # would hold parts of two sequences' segment IDs.
  with jax.ensure_compile_time_eval():
    kernel = a2a.make_splash_mha_a2a_cp(sm.CausalMask((SEQ, SEQ)), config=config(), cp_size=cp, cp_axis="context")
  mesh = Mesh(np.asarray(jax.devices()[:cp]), ("context",))
  mapped = jax.shard_map(
      lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)),
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(None, None),),
      out_specs=P(None, "context", None),
      check_vma=False,
  )
  args = [jax.ShapeDtypeStruct((batch * heads, SEQ, d), jnp.bfloat16) for d in (192, 192, 128)]
  with pytest.raises(ValueError, match="straddle|whole groups|must divide"):
    jax.eval_shape(mapped, *args, jax.ShapeDtypeStruct((batch, SEQ), jnp.int32))


# Segment schedules. (block_q, block_kv, block_q_dkv, block_kv_dkv): Plus-VL
# (r = 8), Tiny-VL (r = 1), r = 2 and r = 4.
SEGMENT_TILES = {
    "plus_vl_r8": (4096, 2048, 512, 4096),
    "tiny_vl_r1": (2048, 2048, 2048, 2048),
    "r2": (2048, 2048, 1024, 2048),
    "r4": (2048, 2048, 512, 2048),
}


def segment_layout(name):
  """Segment IDs of one sequence: packed documents, ID 0 as trailing padding."""
  rng = np.random.default_rng(0)
  if name == "packed":
    lengths = [20000, 7000, 16384, 12000, 10152]
  elif name == "short":
    lengths = list(rng.integers(64, 3000, 40))
  elif name == "single":
    return np.ones(SEQ, np.int32)
  elif name == "nonmonotone":
    ids = np.repeat(rng.permutation(16) + 1, SEQ // 16).astype(np.int32)
    return ids
  lengths = [int(x) for x in lengths]
  ids = np.zeros(SEQ, np.int32)
  start = 0
  for doc, length in enumerate(lengths, start=1):
    ids[start : start + length] = doc
    start = min(SEQ, start + length)
    if start == SEQ:
      break
  return ids


def needed_tiles(ids, bq, bkv):
  """[Q tiles, K/V tiles]: some Q token attends some K/V token (same ID, causal)."""
  q_tiles, kv_tiles = SEQ // bq, SEQ // bkv
  needed = np.zeros((q_tiles, kv_tiles), bool)
  pos = np.arange(SEQ)
  for s in np.unique(ids):
    where = pos[ids == s]
    qt, kt = where // bq, where // bkv
    q_last = np.full(q_tiles, -1)
    np.maximum.at(q_last, qt, where)
    kv_first = np.full(kv_tiles, SEQ)
    np.minimum.at(kv_first, kt, where)
    needed |= (q_last[:, None] >= 0) & (kv_first[None, :] < SEQ) & (kv_first[None, :] <= q_last[:, None])
  return needed


def schedule(info):
  n = int(np.asarray(info.num_active_blocks).reshape(-1)[0])
  return np.asarray(info.active_rows)[:n].astype(int), np.asarray(info.active_cols)[:n].astype(int), n


def is_subsequence(part, whole):
  it = iter(whole)
  return all(any(x == y for y in it) for x in part)


@pytest.mark.parametrize("layout", ["packed", "short", "single", "nonmonotone"])
@pytest.mark.parametrize("tiles", list(SEGMENT_TILES))
def test_segment_schedule_keeps_needed_tiles_and_is_safe(tiles, layout):
  bq, bkv, bq_dkv, bkv_dkv = SEGMENT_TILES[tiles]
  cfg = dataclasses.replace(config(), block_q=bq, block_kv=bkv, block_q_dkv=bq_dkv, block_kv_dkv=bkv_dkv)
  with jax.ensure_compile_time_eval():
    kernel = a2a.make_splash_mha_a2a_cp(sm.CausalMask((SEQ, SEQ)), config=cfg, cp_size=CP, cp_axis="context")
    ids = jnp.asarray(segment_layout(layout))
    fwd, dkv = a2a._segment_schedules(
        kernel.fwd_mask_info, kernel.dkv_mask_info, sk.SegmentIds(ids, ids), cfg,
        num_heads=HEADS, cp_size=CP, cp_axis="context",
    )
  ids = np.asarray(ids)
  for full_info, pruned_info, (b_rows, b_cols), transpose in (
      (kernel.fwd_mask_info, fwd, (bq, bkv), False),
      (kernel.dkv_mask_info, dkv, (bq_dkv, bkv_dkv), True),
  ):
    full_rows, full_cols, _ = schedule(full_info)
    rows, cols, n = schedule(pruned_info)
    pairs = list(zip(rows, cols))
    # The pruned schedule keeps the order of the full one.
    assert is_subsequence(pairs, list(zip(full_rows, full_cols)))
    assert np.array_equal(np.asarray(pruned_info.block_mask)[:n], [
        int(np.asarray(full_info.block_mask)[k]) for k in
        [list(zip(full_rows, full_cols)).index(p) for p in pairs]
    ])
    # Every tile with an attending pair is kept, and every row keeps its diagonal.
    needed = needed_tiles(ids, b_rows, b_cols)
    if transpose:
      needed = needed.T  # dKV schedules are [K/V rows, Q tiles].
    kept = set(pairs)
    assert all((j, i) in kept for j, i in zip(full_rows, full_cols) if needed[j, i])
    assert np.array_equal(np.unique(rows), np.unique(full_rows))
  r = bkv_dkv // bq_dkv
  rows, cols, n = schedule(dkv)
  mla_fused_backward.segment_schedule_safety(rows, cols, r)
  if layout in ("packed", "short"):
    assert n < schedule(kernel.dkv_mask_info)[2]


SEGMENT_CASES = [(e, n, s) for e, n, s in CASES]


@requires_devices
@pytest.mark.parametrize("exchange,names,saved", SEGMENT_CASES, ids=IDS)
def test_segment_schedule_traces_and_lowers(exchange, names, saved):
  with jax.ensure_compile_time_eval():
    kernel = a2a.make_splash_mha_a2a_cp(
        sm.CausalMask((SEQ, SEQ)), config=config(), cp_size=CP, cp_axis="context",
        bwd_input_exchange=exchange, head_checkpoint_names=names, segment_schedule=True,
    )
  mesh = Mesh(np.asarray(jax.devices()[:CP]), ("context",))
  mapped = jax.shard_map(
      lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)),
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(None),),
      out_specs=P(None, "context", None),
      check_vma=False,
  )
  fn = jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(*saved))
  args = inputs() + [jax.ShapeDtypeStruct((HEADS, SEQ, 128), jnp.bfloat16)]
  text = str(jax.make_jaxpr(cycle(fn))(*args))
  assert text.count("name=mla_fused_dq_dkv_a2a") == 1
  assert text.count("all_to_all[") == backward_all_to_alls(exchange, names)
  hlo = jax.jit(cycle(fn)).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
  assert live_forward_kernels(hlo) == (2 if exchange == "head" and names is None else 1)


@requires_devices
@pytest.mark.parametrize("exchange", ["head", "kv_hbm_prefetch", "a2a"])
def test_segment_schedule_with_folded_batch_lowers(exchange):
  cp, heads, batch = 8, 16, 2
  folded = batch * heads
  cfg = dataclasses.replace(config(), block_q_dkv=2048, block_kv_dkv=2048, block_kv_dkv_compute=512)
  with jax.ensure_compile_time_eval():
    kernel = a2a.make_splash_mha_a2a_cp(
        sm.CausalMask((SEQ, SEQ)), config=cfg, cp_size=cp, cp_axis="context",
        bwd_input_exchange=exchange, segment_schedule=True,
    )
  mesh = Mesh(np.asarray(jax.devices()[:cp]), ("context",))
  mapped = jax.shard_map(
      lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)),
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(None, None),),
      out_specs=P(None, "context", None),
      check_vma=False,
  )
  fn = jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(RESIDUALS))
  args = [jax.ShapeDtypeStruct((folded, SEQ, d), jnp.bfloat16) for d in (192, 192, 128)]
  args += [jax.ShapeDtypeStruct((batch, SEQ), jnp.int32), jax.ShapeDtypeStruct((folded, SEQ, 128), jnp.bfloat16)]
  jax.jit(cycle(fn)).trace(*args).lower(lowering_platforms=("tpu",))


def test_segment_schedule_needs_causal_mask():
  with pytest.raises(ValueError, match="causal"):
    a2a.make_splash_mha_a2a_cp(
        sm.FullMask((SEQ, SEQ)), config=config(), cp_size=CP, cp_axis="context", segment_schedule=True
    )


@requires_devices
@pytest.mark.parametrize("cp_barrier", [False, True])
@pytest.mark.parametrize("exchange,names,saved", CASES, ids=IDS)
def test_cp_barrier_replaces_the_device_barrier(exchange, names, saved, cp_barrier):
  # Every fused kernel skips the compiler's device barrier only on request.
  with jax.ensure_compile_time_eval():
    kernel = a2a.make_splash_mha_a2a_cp(
        sm.CausalMask((SEQ, SEQ)), config=config(), cp_size=CP, cp_axis="context",
        bwd_input_exchange=exchange, head_checkpoint_names=names, segment_schedule=True,
        cp_barrier=cp_barrier,
    )
  mesh = Mesh(np.asarray(jax.devices()[:CP]), ("context",))
  mapped = jax.shard_map(
      lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)),
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(None),),
      out_specs=P(None, "context", None),
      check_vma=False,
  )
  fn = jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(*saved))
  args = inputs() + [jax.ShapeDtypeStruct((HEADS, SEQ, 128), jnp.bfloat16)]
  hlo = jax.jit(cycle(fn)).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
  calls = [line for line in hlo.splitlines() if "tpu_custom_call" in line]
  assert sum("mla_fused_dq_dkv_a2a" in line for line in calls) == 1
  assert any("fwd_a2a_cp" in line for line in calls)
  assert all(("skip_device_barrier" in line) == cp_barrier for line in calls)
