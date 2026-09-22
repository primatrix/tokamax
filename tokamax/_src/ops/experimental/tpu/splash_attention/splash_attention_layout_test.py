# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Public output/VJP regressions for automatic Splash layout selection."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash


def _config(width, **overrides):
  return dataclasses.replace(splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=width**-0.5, max_logit_const=0.0,
      combine_log2_scale=True, bwd_scale_after_dot=True,
      interpret=jax.default_backend() == "cpu",
  ), **overrides)


def _inputs(nq, nk, width, value_width, mode="native"):
  kv_heads = 1 if mode in ("mqa", "gqa") else 2
  shapes = [(2, nq, width), (kv_heads, nk, width),
            (kv_heads, nk, value_width), (2, nq, value_width)]
  arrays = [jax.random.normal(key, shape, jnp.bfloat16)
            for key, shape in zip(jax.random.split(jax.random.key(41), 4), shapes)]
  if mode == "mqa":
    arrays[1], arrays[2] = arrays[1][0], arrays[2][0]
  # Non-monotone IDs, including ID 0, and boundaries inside compute tiles.
  ids = base.SegmentIds(
      jnp.asarray((np.arange(nq) // 37) % 3, jnp.int32),
      jnp.asarray((np.arange(nk) // 29) % 3, jnp.int32),
  )
  return arrays, ids


def _oracle(q, k, v, do, ids, mask, config, sinks=None):
  """Independent float64 dense softmax and analytic derivatives."""
  q, k, v, do = [np.asarray(x, np.float64) for x in (q, k, v, do)]
  mqa = k.ndim == 2
  if mqa:
    k, v = k[None], v[None]
  groups = q.shape[0] // k.shape[0]
  keys, values = np.repeat(k, groups, axis=0), np.repeat(v, groups, axis=0)
  logits = (q @ keys.swapaxes(-1, -2)) * config.softmax_scale
  derivative = 1.0
  if config.attn_logits_soft_cap is not None:
    capped = np.tanh(logits / config.attn_logits_soft_cap)
    derivative = 1 - capped**2
    logits = capped * config.attn_logits_soft_cap
  allowed = mask.copy()
  if ids is not None:
    allowed &= np.asarray(ids.q)[:, None] == np.asarray(ids.kv)[None, :]
  logits = np.where(allowed, logits, -np.inf)
  maximum = np.max(logits, axis=-1, keepdims=True)
  if sinks is not None:
    maximum = np.maximum(maximum, np.asarray(sinks)[:, None, None])
  p = np.exp(logits - maximum)
  denominator = np.sum(p, axis=-1, keepdims=True)
  if sinks is not None:
    denominator += np.exp(np.asarray(sinks)[:, None, None] - maximum)
  lse = (maximum + np.log(denominator))[..., 0]
  p /= denominator
  output = p @ values
  dp = do @ values.swapaxes(-1, -2)
  ds = p * (dp - np.sum(dp * p, axis=-1, keepdims=True)) * derivative
  dq = (ds @ keys) * config.softmax_scale
  dk = (ds.swapaxes(-1, -2) @ q) * config.softmax_scale
  dv = p.swapaxes(-1, -2) @ do
  dk = dk.reshape(k.shape[0], groups, *k.shape[1:]).sum(axis=1)
  dv = dv.reshape(v.shape[0], groups, *v.shape[1:]).sum(axis=1)
  return output, lse, dq, dk[0] if mqa else dk, dv[0] if mqa else dv


def _check(config, arrays, ids, mask, *, mode="native", native=True):
  q, k, v, do = arrays
  factory = (splash.make_splash_mqa_single_device if mode == "mqa"
             else splash.make_splash_mha_single_device)
  kernel = factory(mask, config=config, save_residuals=True)
  assert splash._use_native_layout(
      config, q, k, v, ids, kernel.fwd_mask_info,
      kernel.kwargs["mask_function"], is_mqa=mode == "mqa",
  ) == native
  sinks = jnp.array([0.1, -0.2], jnp.float32) if mode == "sinks" else None
  (output, stats), pullback = jax.vjp(
      lambda q, k, v: kernel(q, k, v, ids, sinks), q, k, v
  )
  gradients = pullback((do, jax.tree.map(jnp.zeros_like, stats)))
  expected, lse, *expected_grads = _oracle(q, k, v, do, ids, mask, config, sinks)
  for actual, wanted in zip((output, *gradients), (expected, *expected_grads)):
    assert np.isfinite(np.asarray(actual)).all()
    assert actual.shape == wanted.shape
    actual = np.asarray(actual, np.float64)
    # BF16 output/gradient rounding, not a relaxed full-model accuracy gate.
    relative_l2 = np.linalg.norm(actual - wanted) / np.linalg.norm(wanted)
    assert relative_l2 < 0.01
  np.testing.assert_allclose(stats["logsumexp"], lse, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("shape", [(256, 512, 64, 64), (512, 256, 72, 96), (256, 512, 128, 128)])
@pytest.mark.parametrize("reduction_steps", [None, 3])
@pytest.mark.parametrize("fuse_reciprocal", [False, True])
def test_native_layout_outputs_and_gradients(shape, reduction_steps, fuse_reciprocal):
  nq, nk, width, value_width = shape
  arrays, ids = _inputs(*shape)
  config = _config(width, dq_reduction_steps=reduction_steps,
                   fuse_reciprocal=fuse_reciprocal)
  _check(config, arrays, ids, np.ones((nq, nk), bool))


def test_native_layout_preserves_block_sparse_mask():
  arrays, ids = _inputs(256, 512, 72, 72)
  mask = np.arange(256)[:, None] // 128 == np.arange(512)[None, :] // 256
  _check(_config(72), arrays, ids, mask)


@pytest.mark.parametrize("mode", [
    "online", "nonzero_shift", "soft_cap", "causal", "no_segments",
    "mqa", "gqa", "head_minor", "sinks", "single_compute_tile",
])
def test_compatible_attention_modes(mode):
  arrays, ids = _inputs(256, 256, 72, 72, mode)
  overrides = {
      "online": dict(max_logit_const=None),
      "nonzero_shift": dict(max_logit_const=2.0),
      "soft_cap": dict(attn_logits_soft_cap=3.0),
      "head_minor": dict(q_layout=splash.QKVLayout.HEAD_DIM_MINOR),
      "single_compute_tile": dict(block_kv_dkv_compute=256),
  }.get(mode, {})
  mask = np.ones((256, 256), bool)
  if mode == "causal":
    mask = np.tril(mask)
    # Self-attention segments ensure no entirely masked query rows.
    ids = base.SegmentIds(ids.q, ids.q)
  if mode == "no_segments":
    ids = None
  _check(_config(72, **overrides), arrays, ids, mask, mode=mode,
         native=mode in ("sinks", "single_compute_tile"))
