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
"""Tests for the experimental Pallas TPU KDA implementation."""

import inspect
from unittest import mock

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda import pallas_tpu
from tokamax._src.ops.experimental.kda import pallas_tpu_types
from tokamax._src.ops.experimental.kda.cp_utils import CPContext


def _call_parameter_snapshot(fn, args, kwargs):
  """Returns comparable call metadata without retaining large TPU arrays."""
  bound = inspect.signature(fn).bind(*args, **kwargs)
  bound.apply_defaults()

  def snapshot(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
      return (
          tuple(value.shape),
          str(value.dtype),
          type(value).__name__,
          isinstance(value, jax.core.Tracer),
      )
    return value

  return {
      name: jax.tree.map(snapshot, value)
      for name, value in bound.arguments.items()
  }


class PallasTpuKimiDeltaAttentionConfigTest(absltest.TestCase):

  def test_rejects_unsupported_chunk_size_at_construction(self):
    with self.assertRaisesRegex(ValueError, "only supports chunk_size=64"):
      pallas_tpu.PallasTpuKimiDeltaAttention(chunk_size=128)


class PallasTpuKimiDeltaAttentionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @pytest.mark.long
  def test_chunk_kda_fwd_bwd_matches_xla_t8192(self):
    heads, batch, seq_len, dim = 16, 1, 8192, 128
    q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
    shape = (heads, batch, seq_len, dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
    g = jnp.full(shape, -0.01, dtype=jnp.bfloat16)
    beta = jnp.full((heads, batch, seq_len), 0.5, dtype=jnp.bfloat16)
    kwargs = dict(
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        chunk_size=64,
    )

    def run(implementation, q, k, v, g, beta):
      def loss_fn(q, k, v, g, beta):
        output, final_state = api.kimi_delta_attention(
            q, k, v, g, beta, implementation=implementation, **kwargs
        )
        loss = jnp.sum(jnp.square(output.astype(jnp.float32))) / (
            heads * dim
        )
        return loss, (output, final_state)

      return jax.value_and_grad(
          loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True
      )(q, k, v, g, beta)

    custom_fwd_calls = []
    kernel_bwd_calls = []
    original_custom_fwd = pallas_tpu.chunk_kda_fwd_custom
    original_kernel_bwd = pallas_tpu.chunk_kda_bwd_custom

    def capture_custom_fwd(*args, **call_kwargs):
      custom_fwd_calls.append(
          _call_parameter_snapshot(original_custom_fwd, args, call_kwargs)
      )
      return original_custom_fwd(*args, **call_kwargs)

    def capture_kernel_bwd(*args, **call_kwargs):
      kernel_bwd_calls.append(
          _call_parameter_snapshot(original_kernel_bwd, args, call_kwargs)
      )
      return original_kernel_bwd(*args, **call_kwargs)

    with (
        mock.patch.object(
            pallas_tpu, "chunk_kda_fwd_custom", new=capture_custom_fwd
        ),
        mock.patch.object(
            pallas_tpu, "chunk_kda_bwd_custom", new=capture_kernel_bwd
        ),
    ):
      primal_output = api.kimi_delta_attention(
          q, k, v, g, beta, implementation="pallas_tpu", **kwargs
      )
      self.assertEmpty(kernel_bwd_calls)
      (loss, (output, final_state)), grads = run(
          "pallas_tpu", q, k, v, g, beta
      )

    self.assertLen(custom_fwd_calls, 2)
    primal_custom_params, autodiff_custom_params = custom_fwd_calls
    self.assertFalse(primal_custom_params.pop("return_residuals"))
    self.assertTrue(autodiff_custom_params.pop("return_residuals"))
    self.assertDictEqual(primal_custom_params, autodiff_custom_params)

    self.assertLen(kernel_bwd_calls, 1)
    bwd_params = kernel_bwd_calls[0]
    residuals = bwd_params.pop("residuals")
    self.assertIsInstance(residuals, pallas_tpu_types.KdaResiduals)
    self.assertLen(residuals, 21)
    self.assertLen(bwd_params.pop("grad_outputs"), 2)
    self.assertTrue(bwd_params.pop("use_qk_l2norm_in_kernel"))
    self.assertIsNone(bwd_params.pop("N_max"))
    self.assertFalse(bwd_params.pop("has_initial_state"))
    for name, value in bwd_params.items():
      self.assertEqual(value, autodiff_custom_params[name])

    (ref_loss, (ref_output, ref_final_state)), ref_grads = run(
        "xla", q, k, v, g, beta
    )

    chex.assert_trees_all_close(
        primal_output, (output, final_state), atol=0, rtol=0
    )
    chex.assert_trees_all_close(
        (loss, output, final_state),
        (ref_loss, ref_output, ref_final_state),
        atol=0.03,
        rtol=0.03,
    )
    for name, grad, ref_grad in zip(
        ("dq", "dk", "dv", "dg", "dbeta"), grads, ref_grads
    ):
      with self.subTest(name):
        chex.assert_trees_all_close(
            grad, ref_grad, atol=0.05, rtol=0.05
        )

  def test_varlen_preprocessing_fwd_bwd_matches_xla(self):
    heads, batch, seq_len, dim = 16, 1, 128, 128
    q_key, k_key, v_key, g_key = jax.random.split(jax.random.key(1), 4)
    shape = (heads, batch, seq_len, dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
    g = 0.1 * jax.random.normal(g_key, shape, dtype=jnp.bfloat16)
    beta = jnp.full((heads, batch, seq_len), 0.5, dtype=jnp.bfloat16)
    segment_ids = jnp.concatenate(
        [jnp.ones(48, jnp.int32), jnp.full(80, 2, jnp.int32)]
    )[None]
    initial_state = jnp.zeros(
        (batch, 2, heads, dim, dim), dtype=jnp.float32
    )
    kwargs = dict(
        A_log=jnp.zeros((heads,), dtype=jnp.float32),
        dt_bias=jnp.zeros((heads * dim,), dtype=jnp.float32),
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        segment_ids=segment_ids,
        safe_gate=True,
        lower_bound=-0.01,
        chunk_size=64,
        N_max=2,
    )

    def run(implementation):
      def loss_fn(q, k, v, g, beta):
        output, final_state = api.kimi_delta_attention(
            q, k, v, g, beta, implementation=implementation, **kwargs
        )
        loss = jnp.mean(jnp.square(output.astype(jnp.float32)))
        return loss, (output, final_state)

      return jax.value_and_grad(
          loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True
      )(q, k, v, g, beta)

    (loss, (output, final_state)), grads = run("pallas_tpu")
    (ref_loss, (ref_output, ref_final_state)), ref_grads = run("xla")

    chex.assert_trees_all_close(
        (loss, output, final_state),
        (ref_loss, ref_output, ref_final_state),
        atol=0.05,
        rtol=0.05,
    )
    chex.assert_trees_all_close(
        grads, ref_grads, atol=0.08, rtol=0.08
    )

  def test_fwd_bwd_with_recompute_matches_xla(self):
    heads, batch, seq_len, dim = 16, 1, 128, 128
    q_key, k_key, v_key = jax.random.split(jax.random.key(2), 3)
    shape = (heads, batch, seq_len, dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
    g = jnp.full(shape, -0.01, dtype=jnp.bfloat16)
    beta = jnp.full((heads, batch, seq_len), 0.5, dtype=jnp.bfloat16)

    def run(implementation):
      def loss_fn(q, k, v, g, beta):
        output, _ = api.kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            implementation=implementation,
            use_qk_l2norm_in_kernel=True,
            disable_recompute=False,
            chunk_size=64,
        )
        return jnp.mean(jnp.square(output.astype(jnp.float32)))

      return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
          q, k, v, g, beta
      )

    loss, grads = run("pallas_tpu")
    ref_loss, ref_grads = run("xla")

    chex.assert_trees_all_close(loss, ref_loss, atol=0.05, rtol=0.05)
    chex.assert_trees_all_close(grads, ref_grads, atol=0.08, rtol=0.08)

  def test_two_device_cp_fwd_bwd_matches_xla(self):
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Requires at least two TPU devices.")

    heads, batch, local_seq_len, dim = 2, 1, 64, 128
    seq_len = 2 * local_seq_len
    shape = (heads, batch, seq_len, dim)
    q_key, k_key, v_key = jax.random.split(jax.random.key(3), 3)
    q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
    g = jnp.full(shape, -0.01, dtype=jnp.bfloat16)
    beta = jnp.full((heads, batch, seq_len), 0.5, dtype=jnp.bfloat16)
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    mesh = Mesh(np.asarray(devices[:2]), ("context",))
    cp_context = CPContext(mesh=mesh, axis_name="context")

    def local_forward(q, k, v, g, beta, local_segment_ids):
      output, _ = api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          implementation="pallas_tpu",
          segment_ids=local_segment_ids,
          cp_context=cp_context,
          chunk_size=64,
          N_max=1,
      )
      return output

    with mesh:
      cp_forward = jax.jit(
          jax.shard_map(
              local_forward,
              mesh=mesh,
              in_specs=(P(None, None, "context", None),) * 4
              + (P(None, None, "context"), P(None, "context")),
              out_specs=P(None, None, "context", None),
              check_vma=False,
          )
      )

      def cp_loss(q, k, v, g, beta):
        output = cp_forward(q, k, v, g, beta, segment_ids)
        return jnp.mean(jnp.square(output.astype(jnp.float32))), output

      (loss, output), grads = jax.value_and_grad(
          cp_loss, argnums=(0, 1, 2, 3, 4), has_aux=True
      )(q, k, v, g, beta)

    def ref_loss(q, k, v, g, beta):
      output, _ = api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          implementation="xla",
          segment_ids=segment_ids,
          chunk_size=64,
          N_max=1,
      )
      return jnp.mean(jnp.square(output.astype(jnp.float32))), output

    (ref_loss_value, ref_output), ref_grads = jax.value_and_grad(
        ref_loss, argnums=(0, 1, 2, 3, 4), has_aux=True
    )(q, k, v, g, beta)

    chex.assert_trees_all_close(
        (loss, output), (ref_loss_value, ref_output), atol=0.05, rtol=0.05
    )
    chex.assert_trees_all_close(grads, ref_grads, atol=0.08, rtol=0.08)

  def test_jit_fwd_bwd_metadata_trace(self):
    heads, batch, seq_len, dim = 1, 1, 64, 128
    shape = (heads, batch, seq_len, dim)
    q = jnp.full(shape, 0.01, dtype=jnp.bfloat16)
    k = jnp.full(shape, 0.01, dtype=jnp.bfloat16)
    v = jnp.full(shape, 0.01, dtype=jnp.bfloat16)
    g = jnp.full(shape, -0.01, dtype=jnp.bfloat16)
    beta = jnp.full((heads, batch, seq_len), 0.5, dtype=jnp.bfloat16)
    initial_state = jnp.zeros(
        (batch, 1, heads, dim, dim), dtype=jnp.float32
    )

    def loss_fn(q, k, v, g, beta, initial_state):
      output, final_state = api.kimi_delta_attention(
          q,
          k,
          v,
          g,
          beta,
          implementation="pallas_tpu",
          initial_state=initial_state,
          output_final_state=True,
          chunk_size=64,
      )
      return jnp.mean(jnp.square(output.astype(jnp.float32))) + 0 * jnp.sum(
          final_state
      )

    value_and_grad_fn = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
    )
    value_and_grad_fn.trace(q, k, v, g, beta, initial_state)


if __name__ == "__main__":
  absltest.main()
