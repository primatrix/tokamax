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

"""CPU-interpreted regression test for the forward KV loop control."""

import dataclasses

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


def test_rolled_kv_loop_preserves_causal_output_and_stats():
  seq_len = 512
  q, k, v = [
      random.uniform(
          key, (1, seq_len, 128), jnp.bfloat16, minval=-0.25, maxval=0.25
      )
      for key in random.split(random.key(24), 3)
  ]
  # Visit both fully visible and partially masked KV tiles, with two
  # compute-loop iterations in each tile.
  mask = mask_lib.CausalMask((seq_len, seq_len))
  config = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128, interpret=True
  )
  default_kernel = splash.make_splash_mha_single_device(
      mask, config=config, save_residuals=True
  )
  rolled_kernel = splash.make_splash_mha_single_device(
      mask,
      config=dataclasses.replace(config, fwd_kv_unroll=False),
      save_residuals=True,
  )
  expected = default_kernel(q, k, v)
  actual = rolled_kernel(q, k, v)
  for lhs, rhs in zip(
      jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
  ):
    np.testing.assert_array_equal(lhs, rhs)
