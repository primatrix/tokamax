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
"""Experimental Pallas TPU implementation of Kimi Delta Attention."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.pallas_tpu_bwd import (
    PallasTpuKimiDeltaAttentionVjp,
)
from tokamax._src.ops.experimental.kda.pallas_tpu_fwd import (
    chunk_kda_fwd_custom,
    chunk_kda_fwd,
)
from tokamax._src.ops.experimental.kda.utils import (
    as_public_final_state,
    derive_cp_context,
    segment_ids_to_cu_seqlens,
)
from typing_extensions import override


def _chunk_kda_fwd_no_residuals(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    A_log: jax.Array | None,
    dt_bias: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    segment_ids: jax.Array | None,
    safe_gate: bool,
    lower_bound: float | None,
    disable_recompute: bool,
    cp_context: object | None,
    chunk_size: int,
    N_max: int | None,
) -> base.Output:
  """Runs Pallas forward without materialising Tokamax VJP residuals."""
  cp_context, cu_seqlens = derive_cp_context(
      q=q,
      segment_ids=segment_ids,
      initial_state=initial_state,
      output_final_state=output_final_state,
      cp_context=cp_context,
      chunk_size=chunk_size,
      N_max=N_max,
  )
  if cu_seqlens is None:
    cu_seqlens, _ = segment_ids_to_cu_seqlens(
        segment_ids,
        initial_state=initial_state,
        chunk_size=chunk_size,
        N_max=N_max,
        seq_len=q.shape[2],
  )

  output, final_state, *_ = chunk_kda_fwd(
      q,
      k,
      v,
      g,
      beta,
      A_log=A_log,
      dt_bias=dt_bias,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,  # chunk_kda_fwd_custom need
      use_gate_in_kernel=use_gate_in_kernel,
      segment_ids=segment_ids,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      disable_recompute=disable_recompute,
      cp_context=cp_context,
      chunk_size=chunk_size,
      cu_seqlens=cu_seqlens,
  )
  return output.astype(q.dtype), as_public_final_state(
      final_state, segment_ids=segment_ids
  )


@dataclasses.dataclass(frozen=True)
class PallasTpuKimiDeltaAttention(base.KimiDeltaAttention):
  """Pallas TPU KDA backend.

  This adapter preserves Tokamax's experimental head-first KDA contract:
  inputs are `[H, B, T, D]` and recurrent states are `[B, N, H, K, V]`.
  """

  chunk_size: int = 64

  def __post_init__(self):
    if self.vjp is None:
      object.__setattr__(self, "vjp", PallasTpuKimiDeltaAttentionVjp())

  @override
  def supported_on(self, device: jax.Device) -> bool:
    return device.platform == "tpu"

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
      cp_context: object | None,
      chunk_size: int,
      N_max: int | None,
      return_residuals: bool,
      config: Any,
  ) -> tuple[base.Output, base.Residuals]:
    del config

    if q.dtype not in (jnp.bfloat16, jnp.float32):
      raise NotImplementedError(
          "`pallas_tpu` currently supports bfloat16 and float32 inputs only."
      )
    del self
    if chunk_size != 64:
      raise NotImplementedError("`pallas_tpu` currently supports chunk_size=64.")
    if segment_ids is None and q.shape[2] % chunk_size != 0:
      raise NotImplementedError(
          "`pallas_tpu` requires the sequence length to be divisible by "
          f"`chunk_size`; got T={q.shape[2]}, chunk_size={chunk_size}."
      )

    if return_residuals:
      return chunk_kda_fwd_custom(
          q,
          k,
          v,
          g,
          beta,
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
          cp_context=cp_context,
          chunk_size=chunk_size,
          N_max=N_max,
      )

    output, final_state = _chunk_kda_fwd_no_residuals(
        q,
        k,
        v,
        g,
        beta,
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
        cp_context=cp_context,
        chunk_size=chunk_size,
        N_max=N_max,
    )

    return (output.astype(q.dtype), final_state), None
