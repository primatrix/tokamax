# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Check the chunked FP32 oracle against independent dense FP64 autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy


@pytest.mark.parametrize("input_scale", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("block_q", [32, 64])
@pytest.mark.parametrize("mask_value,barrier_stages", [(-jnp.inf, False), (-1e30, False), (-jnp.inf, True), (-1e30, True)])
def test_oracle_matches_dense_float64_autodiff(input_scale, block_q, mask_value, barrier_stages):
  q, k, v, do = [
      (jax.random.normal(key, (128, 72), jnp.float32) * input_scale).astype(jnp.bfloat16)
      for key in jax.random.split(jax.random.key(41), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [64, 48, 8, 8]))
  result = accuracy.fp32_attention_and_gradients(
      q, k, v, do, ids, ids, block_q=block_q,
      mask_value=mask_value, barrier_stages=barrier_stages,
  )
  with jax.enable_x64():
    def dense(q64, k64, v64):
      logits = q64 @ k64.T / jnp.sqrt(jnp.float64(q64.shape[-1]))
      logits = jnp.where(ids[:, None] == ids[None, :], logits, -jnp.inf)
      return jax.nn.softmax(logits, axis=-1) @ v64

    q64, k64, v64, do64 = (x.astype(jnp.float64) for x in (q, k, v, do))
    output, pullback = jax.vjp(dense, q64, k64, v64)
    grads = pullback(do64)
    logits = q64 @ k64.T / jnp.sqrt(jnp.float64(72))
    lse = jax.scipy.special.logsumexp(
        jnp.where(ids[:, None] == ids[None, :], logits, -jnp.inf), axis=-1,
    )
    for actual, expected in zip(result, (output, lse, *grads)):
      np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-5)


def test_statistics_keep_oracle_precision():
  expected = jnp.asarray([1.001, -2.003], jnp.float32)
  actual = expected.astype(jnp.bfloat16)
  stats = accuracy.accuracy_statistics(actual, expected)
  assert stats["finite"]
  assert stats["relative_l2"] > 0
  assert stats["max_abs"] > 0


def test_nonfinite_oracle_is_rejected():
  values = [jnp.ones((2,), jnp.float32) for _ in range(5)]
  values[2] = jnp.asarray([1.0, jnp.nan], jnp.float32)
  with pytest.raises(FloatingPointError, match="dq"):
    accuracy.require_finite_oracle(values, head=0)
