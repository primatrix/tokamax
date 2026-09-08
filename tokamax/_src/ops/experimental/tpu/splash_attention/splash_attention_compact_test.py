# Copyright 2026 Tokamax authors.
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

"""Lossless residual packing: actual TPU output, stats and VJP equivalence."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import base as splash_base
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_mask as masks,
)


@pytest.mark.parametrize(
    "base2,fuse,packed,scale,amplitude",
    [
        (True, True, False, 192**-0.5, 0.25),
        (True, True, True, 192**-0.5, 0.25),
        (False, True, False, 192**-0.5, 0.25),
        (False, True, True, 192**-0.5, 2.0),
        (True, False, False, 192**-0.5, 0.25),
        (False, False, True, 192**-0.5, 0.25),
        (True, True, False, None, 0.25),
        (False, True, False, None, 0.25),
    ],
)
def test_compact_residuals_exact(base2, fuse, packed, scale, amplitude):
  if jax.default_backend() != "tpu":
    pytest.skip("This regression test requires real TPU compilation")
  rng = np.random.default_rng(91)

  def array(dim, amp):
    return jnp.asarray(rng.standard_normal((2, 512, dim)) * amp, jnp.bfloat16)

  q, k, v, do = (
      array(192, amplitude),
      array(192, amplitude),
      array(128, 1),
      array(128, 1),
  )
  ids = jnp.arange(512, dtype=jnp.int32) // 173
  segments = splash_base.SegmentIds(ids, ids) if packed else None
  cfg = splash.SplashConfig(
      block_q=128,
      block_kv=256,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=256,
      block_kv_dkv_compute=128,
      use_base2_exp=base2,
      fuse_reciprocal=fuse,
      softmax_scale=scale,
      use_experimental_scheduler=True,
  )
  results = []
  for compact in (False, True):
    config = dataclasses.replace(cfg, compact_residuals=compact)
    mask = masks.CausalMask((512, 512))
    attention = splash.make_splash_mha_single_device(mask, config=config)
    with_stats = splash.make_splash_mha_single_device(
        mask, config=config, save_residuals=True
    )

    def call(q, k, v, attention=attention):
      return attention(q, k, v, segment_ids=segments)

    def fb(q, k, v, do):
      out, pb = jax.vjp(call, q, k, v)
      return out, pb(do)

    result = jax.jit(fb)(q, k, v, do)
    stats = jax.jit(
        lambda q, k, v, with_stats=with_stats: with_stats(q, k, v, segment_ids=segments)
    )(q, k, v)
    results.append(jax.device_get((result, stats)))
  for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("base2", [False, True])
def test_causal_diagonal_skip_exact(packed, base2):
  if jax.default_backend() != "tpu":
    pytest.skip("This regression test requires real TPU compilation")
  rng = np.random.default_rng(193)

  def array(dim, amp):
    return jnp.asarray(rng.standard_normal((2, 512, dim)) * amp, jnp.bfloat16)

  q, k, v, do = array(192, 0.25), array(192, 0.25), array(128, 1), array(128, 1)
  ids = jnp.arange(512, dtype=jnp.int32) // 173
  segments = splash_base.SegmentIds(ids, ids) if packed else None
  base_config = splash.SplashConfig(
      block_q=512,
      block_kv=512,
      block_kv_compute=128,
      block_q_dkv=512,
      block_kv_dkv=512,
      block_kv_dkv_compute=128,
      qk_diag_grid=4,
      use_base2_exp=base2,
      softmax_scale=192**-0.5,
      use_experimental_scheduler=True,
  )
  results = []
  for skip in (False, True):
    config = dataclasses.replace(base_config, qk_diag_skip=skip)
    attention = splash.make_splash_mha_single_device(
        masks.CausalMask((512, 512)), config=config
    )

    def fb(q, k, v, do, attention=attention):
      out, pb = jax.vjp(
          lambda q, k, v: attention(q, k, v, segment_ids=segments), q, k, v
      )
      return out, pb(do)

    results.append(jax.device_get(jax.jit(fb)(q, k, v, do)))
  for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("base2", [False, True])
def test_optimized_training_config_matches_fp32_reference(packed, base2):
  """Checks the complete performance configuration, including dq reduction."""
  if jax.default_backend() != "tpu":
    pytest.skip("This regression test requires real TPU compilation")
  rng = np.random.default_rng(307)

  def array(dim, amp):
    return jnp.asarray(rng.standard_normal((1, 512, dim)) * amp, jnp.bfloat16)

  q = array(192, 0.25)
  k = array(192, 0.25)
  v = array(128, 1.0)
  do = array(128, 1.0)
  ids = jnp.arange(512, dtype=jnp.int32) // 173
  segments = splash_base.SegmentIds(ids, ids) if packed else None
  config = splash.SplashConfig(
      block_q=512,
      block_kv=512,
      block_kv_compute=128,
      block_q_dkv=512,
      block_kv_dkv=512,
      block_kv_dkv_compute=128,
      qk_diag_skip=True,
      qk_diag_grid=4,
      compact_residuals=True,
      dq_reduction_steps=3,
      use_base2_exp=base2,
      softmax_scale=192**-0.5,
      use_experimental_scheduler=True,
  )
  attention = splash.make_splash_mha_single_device(
      masks.CausalMask((512, 512)), config=config
  )

  def reference(q, k, v):
    logits = jnp.einsum(
        "hsd,htd->hst", q.astype(jnp.float32), k.astype(jnp.float32)
    ) * jnp.float32(192**-0.5)
    valid = jnp.arange(512)[:, None] >= jnp.arange(512)[None, :]
    if packed:
      valid &= ids[:, None] == ids[None, :]
    logits = jnp.where(valid[None], logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hst,htd->hsd", probs, v.astype(jnp.float32))

  def value_and_grad(fn, do_dtype):
    out, pullback = jax.vjp(fn, q, k, v)
    return out, pullback(do.astype(do_dtype))

  def optimized_value_and_grad(q, k, v, do):
    out, pullback = jax.vjp(
        lambda q, k, v: attention(q, k, v, segment_ids=segments), q, k, v
    )
    return out, pullback(do)

  actual = jax.jit(optimized_value_and_grad)(q, k, v, do)
  expected = value_and_grad(reference, jnp.float32)
  tolerances = (
      (8e-3, 1e-2),
      (8e-2, 3e-2),
      (7e-2, 3e-2),
      (2e-2, 3e-2),
  )
  for actual_leaf, expected_leaf, (atol, rtol) in zip(
      jax.tree.leaves(actual), jax.tree.leaves(expected), tolerances
  ):
    actual_leaf = actual_leaf.astype(jnp.float32)
    expected_leaf = expected_leaf.astype(jnp.float32)
    assert np.isfinite(np.asarray(actual_leaf)).all()
    np.testing.assert_allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)
