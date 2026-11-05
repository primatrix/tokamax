"""Abstract checks of the a2a-cp Splash backward input exchanges (CPU tracing only).

Each ``bwd_input_exchange`` must give the fused backward head-layout q/k/v
without re-running the forward kernel under a remat policy that saves only
named residuals. Device numerics are validated separately.
"""

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
  # Sequence-layout o and the local heads' logsumexp, gathered over CP.
  splash = [("bfloat16", HEADS * SEQ * 128, False), ("float32", HEADS * SEQ, False)]
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
  # dO and Di always; q for kv_hbm_prefetch; q/k/v for a2a.
  assert text.count("all_to_all[") == {"head": 2, "kv_hbm_prefetch": 3, "a2a": 5}[exchange]
  hlo = jax.jit(run).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
  # Unnamed head buffers can only be rebuilt by re-running the forward kernel.
  assert live_forward_kernels(hlo) == (2 if exchange == "head" and names is None else 1)
  # dq keeps the native path's sequence-minor layout (minor-to-major [1, 2, 0]).
  dq_type = f"tensor<{HEADS}x{SEQ // CP}x192xbf16>"
  assert any(
      "@LayoutConstraint" in line and "result_layouts = [dense<[1, 2, 0]>" in line and dq_type in line
      for line in hlo.splitlines()
  )


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
