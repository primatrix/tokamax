"""Device check of the a2a-cp Splash fused MLA backward (TPU only).

Runs at CP8 on 8 TPU devices (v7x 2x2x1). For every tile geometry and
``bwd_input_exchange``, out, dq, dk and dv must match native Splash with the
same tiles and three BF16 dQ partials bit for bit. A local batch folded into
the head axis must match one call per sequence bit for bit, including
max_logits.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_a2a_cp_kernel as a2a
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as sm

CP, HEADS, SEQ = 8, 32, 65536
RESIDUALS = "mla_splash_residuals"
NAMES = ("mla_q_head", "mla_k_head", "mla_v_head")
SEQ_SPEC = P(None, "context", None)
# (block_q, block_kv, block_kv_compute, block_q_dkv, block_kv_dkv, block_kv_dkv_compute, scheduler).
# plus_vl_r8 is the Plus-VL decoder (eight Q tiles per K/V tile); tiny_vl_r1 is
# the Tiny-VL decoder (one Q tile per K/V tile).
TILES = {
    "plus_vl_r8": (4096, 2048, 512, 512, 4096, 512, True),
    "tiny_vl_r1": (2048, 2048, 1024, 2048, 2048, 512, False),
    "r2": (2048, 2048, 1024, 1024, 2048, 512, False),
    "r4": (2048, 2048, 1024, 512, 2048, 512, False),
}
# (tiles, bwd_input_exchange, head_checkpoint_names, names saved by the policy or None for no remat).
CASES = [
    ("plus_vl_r8", "head", None, None),
    ("plus_vl_r8", "head", NAMES, (RESIDUALS, *NAMES)),
    ("plus_vl_r8", "kv_hbm_prefetch", None, (RESIDUALS,)),
    ("plus_vl_r8", "a2a", None, (RESIDUALS,)),
    ("tiny_vl_r1", "kv_hbm_prefetch", None, (RESIDUALS,)),
    ("tiny_vl_r1", "head", NAMES, (RESIDUALS, *NAMES)),
    ("tiny_vl_r1", "a2a", None, (RESIDUALS,)),
    ("r2", "kv_hbm_prefetch", None, (RESIDUALS,)),
    ("r4", "kv_hbm_prefetch", None, (RESIDUALS,)),
]
# Packed documents of unequal length, one row per sequence.
LENGTHS = ([20000, 7000, 16384, 12000, 10152], [30000, 2048, 33488])

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu" or len(jax.devices()) < CP, reason=f"Requires {CP} TPU devices"
)


def config(tiles, dq_reduction_steps=3):
  bq, bkv, bkv_compute, bq_dkv, bkv_dkv, bkv_dkv_compute, scheduler = TILES[tiles]
  return sk.SplashConfig(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkv_compute,
      block_q_dkv=bq_dkv,
      block_kv_dkv=bkv_dkv,
      block_kv_dkv_compute=bkv_dkv_compute,
      use_fused_bwd_kernel=True,
      dq_reduction_steps=dq_reduction_steps,
      q_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      k_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      bwd_cast_before_transpose=True,
      use_base2_exp=False,
      softmax_scale=0.07216878364870322,
      residual_checkpoint_name=RESIDUALS,
      use_experimental_scheduler=scheduler,
  )


def mesh():
  return Mesh(np.asarray(jax.devices()[:CP]), ("context",))


def ids_spec(ids):
  return P(*(None,) * ids.ndim)


def vjp(local, saved, q, k, v, ids, do):
  mapped = jax.shard_map(
      local, mesh=mesh(), in_specs=(SEQ_SPEC,) * 3 + (ids_spec(ids),), out_specs=SEQ_SPEC, check_vma=False
  )
  if saved is not None:
    mapped = jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(*saved))

  @jax.jit
  def run(q, k, v, ids, do):
    out, pullback = jax.vjp(lambda q, k, v: mapped(q, k, v, ids), q, k, v)
    return (out, *pullback(do))

  return run(q, k, v, ids, do)


def native(tiles, dq_reduction_steps, q, k, v, ids, do):
  kernel = sk.make_splash_mha(sm.CausalMask((SEQ, SEQ)), config=config(tiles, dq_reduction_steps), q_seq_shards=1)

  def local(q, k, v, ids):
    to_head = lambda x: lax.all_to_all(x, "context", 0, 1, tiled=True)
    out = kernel(*(to_head(x) for x in (q, k, v)), sk.SegmentIds(ids, ids))
    return lax.all_to_all(out, "context", 1, 0, tiled=True)

  return vjp(local, None, q, k, v, ids, do)


def fused_kernel(tiles, exchange, names):
  return a2a.make_splash_mha_a2a_cp(
      sm.CausalMask((SEQ, SEQ)),
      config=config(tiles),
      cp_size=CP,
      cp_axis="context",
      bwd_input_exchange=exchange,
      head_checkpoint_names=names,
  )


def fused(tiles, exchange, names, saved, q, k, v, ids, do):
  kernel = fused_kernel(tiles, exchange, names)
  return vjp(lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)), saved, q, k, v, ids, do)


def fused_max_logits(tiles, exchange, q, k, v, ids):
  kernel = fused_kernel(tiles, exchange, None)

  def local(q, k, v, ids):
    _, stats = kernel(q, k, v, sk.SegmentIds(ids, ids), save_residuals=True)
    return stats["max_logits"]

  mapped = jax.shard_map(
      local, mesh=mesh(), in_specs=(SEQ_SPEC,) * 3 + (ids_spec(ids),), out_specs=P("context", None), check_vma=False
  )
  return jax.jit(mapped)(q, k, v, ids)


@jax.jit
def compare(a, b):
  bits = lambda x: lax.bitcast_convert_type(x, jnp.uint16)
  a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
  diff = a32 - b32
  return dict(
      mismatches=jnp.sum(bits(a) != bits(b)),
      max_abs=jnp.max(jnp.abs(diff)),
      rel_l2=jnp.sqrt(jnp.sum(diff * diff) / jnp.sum(b32 * b32)),
      nonfinite=jnp.sum(~jnp.isfinite(a32)),
  )


def report(a, b):
  return {key: value.item() for key, value in jax.device_get(compare(a, b)).items()}


@pytest.fixture(scope="module")
def data():
  sharding = NamedSharding(mesh(), SEQ_SPEC)
  keys = jax.random.split(jax.random.key(0), 4)

  def make(key, width):
    return jax.jit(lambda key: jax.random.normal(key, (HEADS, SEQ, width), jnp.bfloat16), out_shardings=sharding)(key)

  q, k, v, do = (make(key, width) for key, width in zip(keys, (192, 192, 128, 128)))
  rows = np.stack([np.repeat(np.arange(len(x), dtype=np.int32), x) for x in LENGTHS])
  ids = jax.device_put(jnp.asarray(rows), NamedSharding(mesh(), P(None, None)))
  return q, k, v, ids, do


_REFERENCES = {}


def references(tiles, data):
  """Native Splash with the same tiles and three BF16 dQ partials."""
  if tiles not in _REFERENCES:
    q, k, v, ids, do = data
    _REFERENCES[tiles] = native(tiles, 3, q, k, v, ids[0], do)
  return _REFERENCES[tiles]


@pytest.mark.parametrize("tiles,exchange,names,saved", CASES)
def test_matches_native_splash(data, tiles, exchange, names, saved):
  q, k, v, ids, do = data
  result = fused(tiles, exchange, names, saved, q, k, v, ids[0], do)
  out = {name: report(a, b) for name, a, b in zip(("out", "dq", "dk", "dv"), result, references(tiles, data))}
  print(json.dumps(dict(tiles=tiles, exchange=exchange, named=names is not None, saved=saved, **out)), flush=True)
  assert all(r["mismatches"] == 0 and r["nonfinite"] == 0 for r in out.values()), out


@pytest.mark.parametrize(
    "exchange,names,saved",
    [
        ("kv_hbm_prefetch", None, (RESIDUALS,)),
        ("head", NAMES, (RESIDUALS, *NAMES)),
    ],
)
def test_batch_folded_into_heads_matches_one_call_per_sequence(data, exchange, names, saved):
  # Tiny-VL: two 16-head sequences per FSDP shard, folded to 32 heads with one
  # segment-ID row per sequence. Each rank's four heads belong to one sequence.
  q, k, v, ids, do = data
  heads, tiles = HEADS // 2, "tiny_vl_r1"
  folded = fused(tiles, exchange, names, saved, q, k, v, ids, do)
  folded_logits = fused_max_logits(tiles, exchange, q, k, v, ids)
  out = {}
  for b in range(2):
    rows = slice(b * heads, (b + 1) * heads)
    single = fused(tiles, exchange, names, saved, q[rows], k[rows], v[rows], ids[b], do[rows])
    for name, a, s in zip(("out", "dq", "dk", "dv"), folded, single):
      out[f"{name}{b}"] = report(a[rows], s)
    single_logits = fused_max_logits(tiles, exchange, q[rows], k[rows], v[rows], ids[b])
    out[f"max_logits{b}"] = report(folded_logits[rows], single_logits)
  print(json.dumps(dict(tiles=tiles, exchange=exchange, folded=2, **out)), flush=True)
  assert all(r["mismatches"] == 0 and r["nonfinite"] == 0 for r in out.values()), out
