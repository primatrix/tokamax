"""CPU interpreted tests of opt-in arithmetic, not native GPU parity tests."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as masks


def config(width=72):
  return splash.SplashConfig(
      block_q=128,
      block_kv=128,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=128,
      interpret=True,
  ).with_gpu_fa4_forward_numerics(softmax_scale=width**-0.5)


@pytest.mark.parametrize(
    "changes",
    [
        {"use_base2_exp": False},
        {"combine_log2_scale": False},
        {"max_logit_const": 0.0},
        {"softmax_scale": None},
        {"softmax_scale": float("nan")},
        {"attn_logits_soft_cap": 10.0},
    ],
)
def test_incompatible_arithmetic_is_rejected(changes):
  with pytest.raises(ValueError):
    dataclasses.replace(config(), **changes)


def test_preset_preserves_tiles_and_does_not_change_default():
  original = splash.SplashConfig.get_default()
  aligned = original.with_gpu_fa4_forward_numerics(softmax_scale=72**-0.5)
  assert not original.gpu_fa4_forward_numerics
  assert aligned.block_q == original.block_q
  assert aligned.block_kv_dkv == original.block_kv_dkv
  assert aligned.max_logit_const is None


@pytest.mark.parametrize("raw_max", [False, True])
@pytest.mark.parametrize("width,length", [(72, 128), (72, 256), (128, 256)])
def test_forward_matches_blockwise_probability_rounding_oracle(
    width, length, raw_max
):
  keys = jax.random.split(jax.random.key(42), 3)
  q, k, v = [
      jax.random.normal(key, (1, length, width), jnp.bfloat16) for key in keys
  ]
  cfg = dataclasses.replace(config(width), fa4_unscaled_max_fma=raw_max)
  mask = masks.FullMask((length, length))
  fn = splash.make_splash_mha_single_device(
      mask, config=cfg, save_residuals=True
  )
  actual, stats = fn(q, k, v)
  # Independent blockwise oracle: denominator stays FP32 while PV alone uses
  # rounded unnormalized probabilities. Two KV tiles exercise online rescaling.
  qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
  m = jnp.full((1, length, 1), -jnp.inf)
  denom = jnp.zeros_like(m)
  numerator = jnp.zeros_like(qf)
  factor = jnp.float32(width**-0.5 * np.log2(np.e))
  for start in range(0, length, 128):
    scores = jnp.einsum("hqd,hkd->hqk", qf, kf[:, start : start + 128])
    if not raw_max:
      scores = scores * factor
    new_m = jnp.maximum(m, jnp.max(scores, axis=-1, keepdims=True))
    p = (
        jnp.exp2(scores * factor - new_m * factor)
        if raw_max
        else jnp.exp2(scores - new_m)
    )
    alpha = jnp.exp2((m - new_m) * factor) if raw_max else jnp.exp2(m - new_m)
    numerator = alpha * numerator + jnp.einsum(
        "hqk,hkd->hqd",
        p.astype(jnp.bfloat16).astype(jnp.float32),
        vf[:, start : start + 128],
    )
    denom = alpha * denom + jnp.sum(p, axis=-1, keepdims=True)
    m = new_m
  expected = (numerator / denom).astype(jnp.bfloat16)
  np.testing.assert_allclose(
      actual.astype(jnp.float32),
      expected.astype(jnp.float32),
      atol=2e-3,
      rtol=1e-2,
  )
  scaled_m = m * factor if raw_max else m
  expected_lse = ((scaled_m + jnp.log2(denom)) / np.log2(np.e))[..., 0]
  np.testing.assert_allclose(
      stats["logsumexp"], expected_lse, atol=2e-5, rtol=2e-5
  )


@pytest.mark.parametrize("raw_max", [False, True])
def test_backward_finite_and_close_to_dense_reference_with_partial_mask(
    raw_max,
):
  keys = jax.random.split(jax.random.key(9), 4)
  q, k, v, do = [
      jax.random.normal(key, (1, 128, 72), jnp.bfloat16) * 0.25 for key in keys
  ]
  allowed = np.tril(np.ones((128, 128), dtype=bool))
  fn = splash.make_splash_mha_single_device(
      masks.NumpyMask(allowed),
      config=dataclasses.replace(config(), fa4_unscaled_max_fma=raw_max),
  )
  out, pullback = jax.vjp(fn, q, k, v)
  gradients = pullback(do)

  def dense(q, k, v):
    scores = (
        jnp.einsum("hqd,hkd->hqk", q.astype(jnp.float32), k.astype(jnp.float32))
        * 72**-0.5
    )
    p = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
    return jnp.einsum("hqk,hkd->hqd", p, v.astype(jnp.float32))

  _, ref_pullback = jax.vjp(dense, q, k, v)
  reference = ref_pullback(do.astype(jnp.float32))
  assert bool(jnp.all(jnp.isfinite(out)))
  for actual, expected in zip(gradients, reference):
    assert bool(jnp.all(jnp.isfinite(actual)))
    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        atol=2e-3,
        rtol=3e-2,
    )


def test_unsupported_dtype_is_rejected():
  fn = splash.make_splash_mha_single_device(
      masks.FullMask((128, 128)), config=config()
  )
  x = jnp.zeros((1, 128, 72), jnp.float32)
  with pytest.raises(ValueError, match="BF16"):
    fn(x, x, x)
