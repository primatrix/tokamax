# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

"""Masked dP overflow must not poison finite packed-attention gradients."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import (
  splash_attention_kernel as s,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
  splash_attention_mask as masks,
)


@pytest.mark.parametrize(
  "interpret,n",
  [(True, 256), (False, 256), (False, 8192)],
  ids=["interpret", "native-small", "native-production"],
)
@pytest.mark.parametrize("exponent", [50, 60, 61, 62])
def test_masked_dp_overflow_keeps_packed_gradients_finite(interpret, n, exponent):
  if not interpret and jax.default_backend() != "tpu":
    pytest.skip("Native TPU required")
  b = min(n, 2048)
  bc = min(n, 512)
  cfg = dataclasses.replace(
    s.SplashConfig.get_default(),
    interpret=interpret,
    block_q=b,
    block_kv=b,
    block_kv_compute=bc,
    block_q_dkv=b,
    block_kv_dkv=b,
    block_kv_dkv_compute=bc,
    use_base2_exp=False,
    softmax_scale=192**-0.5,
    q_layout=s.QKVLayout.SEQ_MINOR,
    k_layout=s.QKVLayout.SEQ_MINOR,
    v_layout=s.QKVLayout.SEQ_MINOR,
    use_experimental_scheduler=True,
  )
  fn = s.make_splash_mha_single_device(masks.CausalMask((n, n)), config=cfg)
  q = k = jnp.zeros((1, n, 192), jnp.bfloat16)
  # Every value and cotangent is finite. Only masked cross-segment dP is huge:
  # V_A=2**e, dO_A=0; V_B=0, dO_B=2**e.
  amp = float(2**exponent)
  v = jnp.zeros((1, n, 128), jnp.bfloat16).at[:, : n // 2].set(amp)
  do = jnp.zeros_like(v).at[:, n // 2 :].set(amp)
  ids = jnp.arange(n, dtype=jnp.int32) // (n // 2)
  segments = s.SegmentIds(ids, ids)

  def calc(q, k, v, do):
    o, pb = jax.vjp(lambda q, k, v: fn(q, k, v, segment_ids=segments), q, k, v)
    return o, *pb(do)

  result = jax.jit(calc)(q, k, v, do)
  jax.block_until_ready(result)
  o, dq, dk, dv = [np.asarray(x, dtype=np.float32) for x in result]
  print(
    json.dumps(
      {
        "n": n,
        "interpret": interpret,
        "exponent": exponent,
        "masked_dp_exact": 128 * amp * amp,
        "nan": {
          name: int(np.isnan(a).sum())
          for name, a in zip(["o", "dq", "dk", "dv"], [o, dq, dk, dv])
        },
        "inf": {
          name: int(np.isinf(a).sum())
          for name, a in zip(["o", "dq", "dk", "dv"], [o, dq, dk, dv])
        },
      }
    ),
    flush=True,
  )
  for a in (o, dq, dk, dv):
    assert np.isfinite(a).all()
  np.testing.assert_allclose(o[:, : n // 2], amp, rtol=0.01, atol=0)
  np.testing.assert_array_equal(o[:, n // 2 :], 0)
  np.testing.assert_array_equal(dq, 0)
  np.testing.assert_array_equal(dk, 0)
  np.testing.assert_array_equal(dv[:, : n // 2], 0)
  # Exact real-arithmetic VJP: causal uniform averages within the second segment.
  inv = 1 / np.arange(1, n // 2 + 1, dtype=np.float64)
  ref = amp * np.cumsum(inv[::-1])[::-1]
  np.testing.assert_allclose(
    dv[0, n // 2 :], np.broadcast_to(ref[:, None], (n // 2, 128)), rtol=0.03, atol=0
  )


@pytest.mark.parametrize("interpret", [True, False], ids=["interpret", "native"])
def test_nonzero_probability_overflow_is_not_sanitized(interpret):
  if not interpret and jax.default_backend() != "tpu":
    pytest.skip("Native TPU required")
  cfg = dataclasses.replace(
    s.SplashConfig.get_default(),
    interpret=interpret,
    use_base2_exp=False,
    softmax_scale=128**-0.5,
  )
  fn = s.make_splash_mha_single_device(masks.CausalMask((256, 256)), config=cfg)
  q = k = jnp.zeros((1, 256, 128), jnp.bfloat16)
  v = jnp.full_like(q, float(2**61))
  do = v

  def calc(q, k, v, do):
    o, pb = jax.vjp(fn, q, k, v)
    return o, *pb(do)

  o, dq, _dk, _dv = jax.jit(calc)(q, k, v, do)
  assert np.isfinite(np.asarray(o, dtype=np.float32)).all()
  assert not np.isfinite(np.asarray(dq, dtype=np.float32)).all()
