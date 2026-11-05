"""Device check of the a2a-cp Splash backward input exchanges (TPU only).

Every ``bwd_input_exchange`` must give the same output and dq/dk/dv as the
native path: XLA all-to-all to the head layout, ``make_splash_mha`` and its
VJP, then the all-to-all back. Runs on 8 TPU devices (v7x 2x2x1) at CP8, with
and without a remat policy that saves only named residuals. Values are
compared bit for bit.
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

CP, LOCAL_HEADS, SEQ = 8, 4, 65536
HEADS = CP * LOCAL_HEADS
RESIDUALS = "mla_splash_residuals"
NAMES = ("mla_q_head", "mla_k_head", "mla_v_head")
SEQ_SPEC = P(None, "context", None)
# (bwd_input_exchange, head_checkpoint_names, names saved by the policy or None for no remat).
CASES = [
    ("head", None, None),
    ("head", None, (RESIDUALS,)),
    ("head", NAMES, (RESIDUALS, *NAMES)),
    ("kv_hbm_prefetch", None, None),
    ("kv_hbm_prefetch", None, (RESIDUALS,)),
    ("a2a", None, None),
    ("a2a", None, (RESIDUALS,)),
]

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu" or len(jax.devices()) < CP, reason=f"Requires {CP} TPU devices"
)


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


def mesh():
  return Mesh(np.asarray(jax.devices()[:CP]), ("context",))


def vjp(local, saved, q, k, v, ids, do):
  mapped = jax.shard_map(
      local, mesh=mesh(), in_specs=(SEQ_SPEC,) * 3 + (P(None),), out_specs=SEQ_SPEC, check_vma=False
  )
  if saved is not None:
    mapped = jax.checkpoint(mapped, prevent_cse=True, policy=jax.checkpoint_policies.save_only_these_names(*saved))

  @jax.jit
  def run(q, k, v, ids, do):
    out, pullback = jax.vjp(lambda q, k, v: mapped(q, k, v, ids), q, k, v)
    return (out, *pullback(do))

  return run(q, k, v, ids, do)


@jax.jit
def compare(a, b):
  bits = lambda x: lax.bitcast_convert_type(x, jnp.uint16)
  diff = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
  return jnp.sum(bits(a) != bits(b)), jnp.max(diff), jnp.any(jnp.isnan(a))


@pytest.fixture(scope="module")
def data():
  sharding = NamedSharding(mesh(), SEQ_SPEC)
  keys = jax.random.split(jax.random.key(0), 4)

  def make(key, width):
    return jax.jit(lambda key: jax.random.normal(key, (HEADS, SEQ, width), jnp.bfloat16), out_shardings=sharding)(key)

  q, k, v, do = (make(key, width) for key, width in zip(keys, (192, 192, 128, 128)))
  # Packed documents of unequal length, as in training.
  lengths = [20000, 7000, 16384, 12000, SEQ - 55384]
  ids = jnp.asarray(np.repeat(np.arange(len(lengths), dtype=np.int32), lengths))
  ids = jax.device_put(ids, NamedSharding(mesh(), P(None)))
  return q, k, v, ids, do


@pytest.fixture(scope="module")
def native(data):
  kernel = sk.make_splash_mha(sm.CausalMask((SEQ, SEQ)), config=config(), q_seq_shards=1)

  def local(q, k, v, ids):
    to_head = lambda x: lax.all_to_all(x, "context", 0, 1, tiled=True)
    out = kernel(*(to_head(x) for x in (q, k, v)), sk.SegmentIds(ids, ids))
    return lax.all_to_all(out, "context", 1, 0, tiled=True)

  return vjp(local, None, *data)


@pytest.mark.parametrize("exchange,names,saved", CASES)
def test_matches_native_splash(data, native, exchange, names, saved):
  kernel = a2a.make_splash_mha_a2a_cp(
      sm.CausalMask((SEQ, SEQ)),
      config=config(),
      cp_size=CP,
      cp_axis="context",
      bwd_input_exchange=exchange,
      head_checkpoint_names=names,
  )
  result = vjp(lambda q, k, v, ids: kernel(q, k, v, sk.SegmentIds(ids, ids)), saved, *data)
  report = {}
  for name, a, b in zip(("out", "dq", "dk", "dv"), result, native):
    count, max_abs, nan = jax.device_get(compare(a, b))
    report[name] = dict(mismatches=int(count), max_abs=float(max_abs), nan=bool(nan))
  print(json.dumps(dict(exchange=exchange, named=names is not None, saved=saved, **report)), flush=True)
  assert all(r["mismatches"] == 0 and not r["nan"] for r in report.values()), report
