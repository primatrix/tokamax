# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU interpretation of ViT arithmetic boundaries; no GPU equivalence claim."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as masks


def _config(**kwargs):
  return splash.SplashConfig(
      block_q=128,
      block_kv=256,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=256,
      block_kv_dkv_compute=128,
      softmax_scale=72**-0.5,
      use_base2_exp=True,
      combine_log2_scale=True,
      bwd_scale_after_dot=True,
      gpu_numerics_alignment=True,
      interpret=True,
      **kwargs,
  )


def _dot(x, y):
  return jnp.matmul(x, y, preferred_element_type=jnp.float32)


def _online_reference(q, k, v, config):
  scale = jnp.float32(config.softmax_scale * splash.LOG2E)
  maximum = jnp.full(q.shape[:2] + (1,), -jnp.inf, jnp.float32)
  denominator = jnp.zeros_like(maximum)
  output = jnp.zeros(q.shape[:-1] + (v.shape[-1],), jnp.float32)
  for start in range(0, k.shape[1], config.block_kv_compute):
    stop = start + config.block_kv_compute
    raw = _dot(q, k[:, start:stop].mT)
    next_maximum = jnp.maximum(maximum, raw.max(axis=-1, keepdims=True))
    probability = jnp.exp2(raw * scale - next_maximum * scale)
    correction = jnp.exp2((maximum - next_maximum) * scale)
    denominator = probability.sum(axis=-1, keepdims=True) + correction * denominator
    output = correction * output + _dot(probability.astype(v.dtype), v[:, start:stop])
    maximum = next_maximum
  # LSE is stored in base 2, with the row maximum scaled only at the writer.
  lse_base2 = maximum * scale + jnp.log2(denominator)
  if not config.fuse_reciprocal:
    output = output.astype(q.dtype).astype(jnp.float32)
  return (output / denominator).astype(q.dtype), lse_base2[..., 0]


@pytest.mark.parametrize(
    "reduction,fuse,layout",
    [
        (None, True, splash.QKVLayout.HEAD_DIM_MINOR),
        (3, True, splash.QKVLayout.HEAD_DIM_MINOR),
        (None, False, splash.QKVLayout.SEQ_MINOR),
        (3, False, splash.QKVLayout.SEQ_MINOR),
    ],
)
def test_online_forward_and_merged_fp32_dq(reduction, fuse, layout):
  config = _config(
      dq_reduction_steps=reduction,
      fuse_reciprocal=fuse,
      q_layout=layout,
      k_layout=layout,
      v_layout=layout,
  )
  keys = jax.random.split(jax.random.key(131), 4)
  q = jax.random.normal(keys[0], (1, 128, 72), jnp.bfloat16)
  k = jax.random.normal(keys[1], (1, 768, 72), jnp.bfloat16)
  v = jax.random.normal(keys[2], k.shape, jnp.bfloat16)
  do = jax.random.normal(keys[3], q.shape, jnp.bfloat16)
  kernel = splash.make_splash_mha_single_device(
      masks.FullMask((q.shape[1], k.shape[1])),
      config=config,
      save_residuals=True,
  )
  (output, stats), pullback = jax.vjp(kernel, q, k, v)
  expected, lse_base2 = _online_reference(q, k, v, config)
  if fuse:
    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected))
  else:
    # Unfused normalization crosses a BF16 writer before the division.
    np.testing.assert_allclose(
        output.astype(jnp.float32), expected.astype(jnp.float32), atol=1e-3, rtol=1e-2
    )
  np.testing.assert_allclose(
      stats["logsumexp"], lse_base2 / splash.LOG2E, atol=2e-6, rtol=2e-6
  )
  dq, dk, dv = pullback((do, jax.tree.map(jnp.zeros_like, stats)))
  raw = _dot(k, q.mT)
  probability = jnp.exp2(
      raw * jnp.float32(config.softmax_scale * splash.LOG2E) - lse_base2[:, None, :]
  )
  di = jnp.sum(output.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
  ds = ((_dot(v, do.mT) - di[:, None, :]) * probability).astype(jnp.bfloat16)
  # Preserve the kernel's per-compute-tile dot and FP32 merge before one scale.
  merged = jnp.zeros_like(q, dtype=jnp.float32)
  for start in range(0, k.shape[1], config.block_kv_dkv):
    local = jnp.zeros_like(merged)
    for inner in range(start, start + config.block_kv_dkv, config.block_kv_dkv_compute):
      stop = inner + config.block_kv_dkv_compute
      local += _dot(ds[:, inner:stop].mT, k[:, inner:stop])
    merged += local
  expected_dq = (merged * jnp.float32(config.softmax_scale)).astype(q.dtype)
  np.testing.assert_allclose(
      dq.astype(jnp.float32), expected_dq.astype(jnp.float32), atol=2e-3, rtol=1e-2
  )
  assert all(
      np.isfinite(np.asarray(gradient, np.float32)).all() for gradient in (dq, dk, dv)
  )
  # The output of each backward grid partition stays FP32 until the epilogue.
  backward = str(
      jax.make_jaxpr(
          lambda q, k, v, do: jax.vjp(kernel, q, k, v)[1](
              (do, jax.tree.map(jnp.zeros_like, stats))
          )
      )(q, k, v, do)
  )
  assert "f32[3,1,128,72]" in backward


@pytest.mark.parametrize(
    "changes",
    [
        {"use_base2_exp": False},
        {"combine_log2_scale": False},
        {"softmax_scale": None},
        {"bwd_scale_after_dot": False},
        {"max_logit_const": 0.0},
        {"attn_logits_soft_cap": 50.0},
    ],
)
def test_alignment_rejects_unsupported_arithmetic(changes):
  with pytest.raises(ValueError, match="GPU numerical alignment"):
    dataclasses.replace(_config(), **changes)


def _jaxprs(program):
  program = program.jaxpr if hasattr(program, "jaxpr") else program
  yield program
  for equation in program.eqns:
    for parameter in equation.params.values():
      for child in parameter if isinstance(parameter, (list, tuple)) else (parameter,):
        if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
          yield from _jaxprs(child)


def test_actual_pallas_graph_has_raw_max_bf16_probability_and_final_dq_scale():
  config = _config(dq_reduction_steps=3)
  q = jax.ShapeDtypeStruct((1, 128, 72), jnp.bfloat16)
  kv = jax.ShapeDtypeStruct((1, 1536, 72), jnp.bfloat16)
  kernel = splash.make_splash_mha_single_device(
      masks.FullMask((128, 1536)), config=config
  )
  forward = jax.make_jaxpr(kernel)(q, kv, kv)
  checked = 0
  for program in _jaxprs(forward):
    producers = {var: eqn for eqn in program.eqns for var in eqn.outvars}
    maxima = [eqn for eqn in program.eqns if eqn.primitive.name == "reduce_max"]
    if not maxima:
      continue
    for maximum in maxima:
      assert producers[maximum.invars[0]].primitive.name == "dot_general"
    pv = [eqn for eqn in program.eqns if eqn.primitive.name == "dot_general"][-1]
    assert pv.invars[0].aval.dtype == jnp.bfloat16
    cast = producers[pv.invars[0]]
    assert cast.primitive.name == "convert_element_type"
    assert producers[cast.invars[0]].primitive.name == "exp2"
    checked += 1
  assert checked

  backward = jax.make_jaxpr(lambda x, y, z, do: jax.vjp(kernel, x, y, z)[1](do))(
      q, kv, kv, q
  )
  checked = 0
  for program in _jaxprs(backward):
    for equation in program.eqns:
      if equation.primitive.name != "reduce_sum":
        continue
      source = equation.invars[0]
      if source.aval.shape != (3, 1, 128, 72):
        continue
      assert source.aval.dtype == jnp.float32
      merged = equation.outvars[0]
      consumers = [eqn for eqn in program.eqns if merged in eqn.invars]
      assert len(consumers) == 1 and consumers[0].primitive.name == "mul"
      assert consumers[0].outvars[0].aval.dtype == jnp.float32
      checked += 1
  assert checked


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("alignment", [False, True])
@pytest.mark.parametrize(
    "layout", [splash.QKVLayout.HEAD_DIM_MINOR, splash.QKVLayout.SEQ_MINOR]
)
def test_reused_dq_alias_matches_full_reduction(
    dynamic, alignment, layout, record_property
):
  """Exercise six KV tiles with TPU DMA/alias simulation or a native TPU."""
  native = jax.default_backend() == "tpu"
  # interpret=True uses the generic HLO interpreter, whose separate input/output
  # carries do not propagate aliased output writes back to subsequent input reads
  # (see the IO-aliasing TODO in JAX 0.9.2 hlo_interpreter.py). TPU interpretation
  # models shared buffers, pipelined DMA and synchronization, and checks races.
  interpret = False if native else pltpu.InterpretParams(detect_races=True)
  config = dataclasses.replace(
      _config(),
      gpu_numerics_alignment=alignment,
      interpret=interpret,
      q_layout=layout,
      k_layout=layout,
      v_layout=layout,
  )
  keys = jax.random.split(jax.random.key(131), 4)
  q = jax.random.normal(keys[0], (1, 256, 72), jnp.bfloat16)
  k = jax.random.normal(keys[1], (1, 1536, 72), jnp.bfloat16)
  v = jax.random.normal(keys[2], k.shape, jnp.bfloat16)
  do = jax.random.normal(keys[3], q.shape, jnp.bfloat16)
  qids = np.where(np.arange(256) >= 224, 3, np.where(np.arange(256) < 96, 1, 2))
  kids = np.where((np.arange(1536) < 480) | (np.arange(1536) >= 1024), 1, 2)
  allowed = qids[:, None] == kids[None, :] if dynamic else np.ones((256, 1536), bool)
  full = splash.make_splash_mha_single_device(masks.NumpyMask(allowed), config=config)
  if dynamic:
    reduced = splash.make_dynamic_splash_mha(jnp.asarray(allowed), config=config)
  else:
    reduced = splash.make_splash_mha_single_device(
        masks.NumpyMask(allowed),
        config=dataclasses.replace(config, dq_reduction_steps=3),
    )
  expected, expected_vjp = jax.vjp(full, q, k, v)
  actual, actual_vjp = jax.vjp(reduced, q, k, v)
  results = (actual, *actual_vjp(do))
  references = (expected, *expected_vjp(do))
  if dynamic and alignment:
    np.testing.assert_array_equal(np.asarray(results[0][:, 224:], np.float32), 0)
    np.testing.assert_array_equal(np.asarray(results[1][:, 224:], np.float32), 0)
  record_property("execution_backend", jax.default_backend())
  record_property(
      "kernel_execution", "native_tpu" if native else "tpu_interpret_detect_races"
  )
  for name, result, reference in zip(("output", "dq", "dk", "dv"), results, references):
    result, reference = np.asarray(result, np.float32), np.asarray(
        reference, np.float32
    )
    assert np.isfinite(result).all()
    relative_l2 = np.linalg.norm(result - reference) / np.linalg.norm(reference)
    record_property(name + "_relative_l2", float(relative_l2))
    # Default-off alias buffers retain the existing BF16 partial-write rounding.
    # Alignment keeps those partials FP32; only the reduction tree then differs.
    assert relative_l2 < (0.001 if alignment else 0.01), (name, relative_l2)
