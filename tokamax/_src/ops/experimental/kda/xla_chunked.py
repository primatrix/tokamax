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
"""Experimental chunk-wise XLA implementation of Kimi Delta Attention."""

import dataclasses
from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import base
from typing_extensions import override


_Config = TypeVar("_Config")


def _pad_sequence(x: jax.Array, size: int) -> jax.Array:
  seq_len = x.shape[2]
  padded_len = ((seq_len + size - 1) // size) * size
  pad = padded_len - seq_len
  if pad == 0:
    return x
  widths = [(0, 0)] * x.ndim
  widths[2] = (0, pad)
  return jnp.pad(x, widths)


@dataclasses.dataclass(frozen=True)
class XlaChunkedKimiDeltaAttention(base.KimiDeltaAttention):
  """Chunk-wise XLA KDA implementation.

  This implementation is mathematically equivalent to the recurrent XLA
  reference but groups timesteps into blocks and solves intra-block delta-rule
  dependencies with a lower-triangular forward substitution.
  """

  chunk_size: int = 64

  @jaxtyping.jaxtyped
  @override
  def _fwd(
      self,
      q: Float[Array, "H B T K"],
      k: Float[Array, "H B T K"],
      v: Float[Array, "H B T V"],
      g: Float[Array, "H B T K"],
      beta: Float[Array, "H B T"],
      *,
      A_log: Float[Array, "H"] | None,
      dt_bias: Float[Array, "H*K"] | None,
      scale: float,
      initial_state: Float[Array, "B N H K V"] | None,
      output_final_state: bool,
      use_qk_l2norm_in_kernel: bool,
      use_gate_in_kernel: bool,
      segment_ids: Int[Array, "B T"] | None,
      safe_gate: bool,
      lower_bound: float | None,
      disable_recompute: bool,
      return_intermediate_states: bool,
      cp_context: object | None,
      transpose_state_layout: bool,
      chunk_size: int,
      N_max: int | None,
      return_residuals: bool,
      config: _Config,
  ) -> tuple[base.Output, base.Residuals]:
    """Computes KDA by grouping timesteps into fixed-size chunks."""
    del config, return_residuals  # Unused.

    if (
        segment_ids is not None
        or use_qk_l2norm_in_kernel
        or use_gate_in_kernel
        or return_intermediate_states
        or transpose_state_layout
        or (cp_context is not None and getattr(cp_context, "is_cp_enabled", False))
        or (initial_state is not None and initial_state.shape[1] != 1)
    ):
      output = base.KimiDeltaAttention()(
          q=q,
          k=k,
          v=v,
          g=g,
          beta=beta,
          A_log=A_log,
          dt_bias=dt_bias,
          scale=scale,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
          use_gate_in_kernel=use_gate_in_kernel,
          segment_ids=segment_ids,
          safe_gate=safe_gate,
          lower_bound=lower_bound,
          disable_recompute=disable_recompute,
          return_intermediate_states=return_intermediate_states,
          cp_context=cp_context,
          transpose_state_layout=transpose_state_layout,
          chunk_size=chunk_size,
          N_max=N_max,
      )
      return output, None

    del A_log, dt_bias, safe_gate, lower_bound, disable_recompute, N_max

    if chunk_size <= 0:
      raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")

    heads, batch, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    padded_len = ((seq_len + chunk_size - 1) // chunk_size) * chunk_size
    num_chunks = padded_len // chunk_size
    acc_dtype = base._accumulator_dtype(q.dtype)  # pylint: disable=protected-access
    output_dtype = q.dtype

    q = _pad_sequence(q, chunk_size)
    k = _pad_sequence(k, chunk_size)
    v = _pad_sequence(v, chunk_size)
    g = _pad_sequence(g, chunk_size)
    beta = _pad_sequence(beta, chunk_size)

    def reshape_chunks(x):
      x = x.astype(acc_dtype)
      return x.reshape(heads, batch, num_chunks, chunk_size, -1)

    q = reshape_chunks(q) * scale
    k = reshape_chunks(k)
    v = reshape_chunks(v)
    g = jnp.cumsum(reshape_chunks(g), axis=3)
    beta = beta.astype(acc_dtype)
    beta = beta.reshape(heads, batch, num_chunks, chunk_size)

    triangular_mask = jnp.triu(
        jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_)
    )
    attention = jnp.zeros((*q.shape[:-1], chunk_size), dtype=acc_dtype)
    for i in range(chunk_size):
      k_i = k[..., i, :]
      g_i = g[..., i : i + 1, :]
      attention = attention.at[..., i].set(
          jnp.einsum("...ck,...k->...c", k * jnp.exp(g - g_i), k_i)
      )
    attention = attention * beta[..., None]
    attention = jnp.where(triangular_mask, 0, -attention)

    for i in range(1, chunk_size):
      update = jnp.einsum(
          "...m,...mj->...j", attention[..., i, :], attention[..., :, :i]
      )
      attention = attention.at[..., i, :i].set(
          attention[..., i, :i] + update
      )
    attention = (attention + jnp.eye(chunk_size, dtype=acc_dtype)) * beta[
        ..., None, :
    ]

    w = attention @ (jnp.exp(g) * k)
    u = attention @ v

    state = jnp.zeros((heads, batch, key_dim, value_dim), dtype=acc_dtype)
    if initial_state is not None:
      state = state + jnp.transpose(
          initial_state[:, 0].astype(acc_dtype), (1, 0, 2, 3)
      )

    output = jnp.zeros_like(v)
    strict_upper_mask = jnp.triu(
        jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1
    )
    for i in range(num_chunks):
      q_i = q[:, :, i]
      k_i = k[:, :, i]
      u_i = u[:, :, i]
      g_i = g[:, :, i]
      w_i = w[:, :, i]

      intra_attention = jnp.einsum(
          "...ck,...jk->...cj",
          q_i * jnp.exp(g_i),
          k_i * jnp.exp(-g_i),
      )
      intra_attention = jnp.where(strict_upper_mask, 0, intra_attention)

      corrected_values = u_i - w_i @ state
      output = output.at[:, :, i].set(
          jnp.einsum("...ck,...kv->...cv", q_i * jnp.exp(g_i), state)
          + intra_attention @ corrected_values
      )

      g_last = g_i[:, :, -1]
      state = state * jnp.exp(g_last)[..., None]
      state = state + jnp.einsum(
          "...ck,...cv->...kv",
          jnp.exp(g_last[:, :, None, :] - g_i) * k_i,
          corrected_values,
      )

    output = output.reshape(heads, batch, padded_len, value_dim)
    output = output[:, :, :seq_len].astype(output_dtype)
    if output_final_state:
      final_state = jnp.transpose(state, (1, 0, 2, 3))[:, None]
    else:
      final_state = None
    return (output, final_state), None
