"""CPU-interpreted checks for the opt-in GPU ViT arithmetic boundaries."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


@pytest.mark.parametrize("dq_reduction_steps", [None, 3])
@pytest.mark.parametrize("scale_after_dot", [False, True])
def test_aligned_attention_rounds_pv_and_scales_combined_float32_dq(
    dq_reduction_steps, scale_after_dot
):
  keys = jax.random.split(jax.random.key(31), 4)
  q = jax.random.normal(keys[0], (1, 128, 72), jnp.bfloat16) * 0.25
  k, v = [
      jax.random.normal(key, (1, 1024, 72), jnp.bfloat16) for key in keys[1:3]
  ]
  k *= 0.25
  do = jax.random.normal(keys[3], q.shape, jnp.bfloat16)
  scale = 72**-0.5
  config = splash.SplashConfig(
      block_q=128,
      block_kv=256,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=256,
      block_kv_dkv_compute=128,
      softmax_scale=scale,
      use_base2_exp=True,
      interpret=True,
      vl_gpu_numerics_alignment=True,
      dq_reduction_steps=dq_reduction_steps,
      bwd_scale_after_dot=scale_after_dot,
  )
  mask = mask_lib.FullMask((128, 1024))
  kernel = splash.make_splash_mha_single_device(
      mask, config=config, save_residuals=True
  )
  (output, stats), pullback = jax.vjp(lambda a, b, c: kernel(a, b, c), q, k, v)
  dq, dk, dv = pullback((do, jax.tree.map(jnp.zeros_like, stats)))

  # Replay the declared forward boundary with eight independent 128-key tiles.
  # Denominators consume FP32 P while PV consumes its BF16 buffer value.
  scores = jnp.einsum(
      "hqd,hkd->hqk", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  factor = jnp.float32(scale * splash.LOG2E)

  def reference_forward(scores, v):
    maximum = jnp.full((1, 128, 1), -jnp.inf)
    denominator = jnp.zeros_like(maximum)
    accumulator = jnp.zeros(q.shape, jnp.float32)
    for start in range(0, 1024, 128):
      score = scores[..., start : start + 128]
      next_maximum = jnp.maximum(maximum, score.max(-1, keepdims=True) * factor)
      alpha = jnp.exp2(maximum - next_maximum)
      probability = jnp.exp2(score * factor - next_maximum)
      accumulator = alpha * accumulator + jnp.einsum(
          "hqk,hkd->hqd",
          probability.astype(jnp.bfloat16),
          v[:, start : start + 128],
          preferred_element_type=jnp.float32,
      )
      denominator = alpha * denominator + probability.sum(-1, keepdims=True)
      maximum = next_maximum
    expected = (accumulator * (jnp.float32(1) / denominator)).astype(
        jnp.bfloat16
    )
    return expected, maximum

  expected, maximum = jax.jit(reference_forward)(scores, v)
  # Different FP32 dot/reduction lowering can move BF16 ties in the
  # recurrence; this checks the declared arithmetic, not bitwise equivalence.
  np.testing.assert_allclose(
      output.astype(jnp.float32),
      expected.astype(jnp.float32),
      rtol=1 / 128,
      atol=2e-5,
  )
  baseline = splash.make_splash_mha_single_device(
      mask,
      config=dataclasses.replace(
          config, vl_gpu_numerics_alignment=False, combine_log2_scale=True
      ),
  )(q, k, v)
  assert np.count_nonzero(np.asarray(output) != np.asarray(baseline)) > 128
  np.testing.assert_allclose(
      stats["max_logits"],
      maximum[..., 0] / jnp.float32(splash.LOG2E),
      rtol=1e-6,
  )

  # Use the kernel's actual saved output/LSE, as the GPU backward does.
  p = jnp.exp2(
      scores * factor
      - stats["logsumexp"][..., None] * jnp.float32(splash.LOG2E)
  )
  dp = jnp.einsum("hqd,hkd->hqk", do.astype(jnp.float32), v.astype(jnp.float32))
  di = jnp.sum(
      output.astype(jnp.float32) * do.astype(jnp.float32), -1, keepdims=True
  )
  ds = ((dp - di) * p).astype(jnp.bfloat16)
  contributions = [
      jnp.einsum(
          "hqk,hkd->hqd",
          ds[..., start : start + 128],
          k[:, start : start + 128],
          preferred_element_type=jnp.float32,
      )
      for start in range(0, 1024, 128)
  ]
  # Each memory tile sums two compute tiles before the cross-tile FP32 sum.
  expected_dq = (sum(contributions) * jnp.float32(scale)).astype(q.dtype)
  np.testing.assert_allclose(
      dq.astype(jnp.float32),
      expected_dq.astype(jnp.float32),
      atol=1e-4,
      rtol=1e-2,
  )
  for gradient in (dq, dk, dv):
    assert np.isfinite(np.asarray(gradient, np.float32)).all()
  remat = jax.checkpoint(lambda a, b, c: kernel(a, b, c)[0])
  _, remat_pullback = jax.vjp(remat, q, k, v)
  for actual, recomputed in zip((dq, dk, dv), remat_pullback(do)):
    np.testing.assert_array_equal(actual, recomputed)


def test_alignment_rejects_unbound_fixed_shift_and_soft_cap():
  config = splash.SplashConfig.get_default()
  for changes in (
      {"max_logit_const": 0.0},
      {"attn_logits_soft_cap": 10.0},
      {"use_base2_exp": False},
  ):
    with pytest.raises(ValueError, match="Aligned ViT attention"):
      dataclasses.replace(
          config, vl_gpu_numerics_alignment=True, softmax_scale=0.125, **changes
      )


def test_aligned_sparse_grid_keeps_inactive_dq_partials_zero():
  keys = jax.random.split(jax.random.key(21), 4)
  q = jax.random.normal(keys[0], (1, 256, 72), jnp.bfloat16) * 0.25
  k = jax.random.normal(keys[1], (1, 1024, 72), jnp.bfloat16) * 0.25
  v = jax.random.normal(keys[2], k.shape, jnp.bfloat16)
  do = jax.random.normal(keys[3], q.shape, jnp.bfloat16)
  q_ids = np.repeat([1, 2], [128, 128])
  kv_ids = np.repeat([1, 2, 0], [384, 512, 128])
  mask = mask_lib.NumpyMask(q_ids[:, None] == kv_ids[None, :])
  config = splash.SplashConfig(
      block_q=128,
      block_kv=256,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=256,
      block_kv_dkv_compute=128,
      softmax_scale=72**-0.5,
      vl_gpu_numerics_alignment=True,
      bwd_scale_after_dot=True,
      interpret=True,
  )
  results = []
  for reduction in (None, 3):
    kernel = splash.make_splash_mha_single_device(
        mask, config=dataclasses.replace(config, dq_reduction_steps=reduction)
    )
    output, pullback = jax.vjp(kernel, q, k, v)
    results.append((output, pullback(do)))
  for expected, actual in zip(
      jax.tree.leaves(results[0]), jax.tree.leaves(results[1])
  ):
    np.testing.assert_array_equal(actual, expected)
