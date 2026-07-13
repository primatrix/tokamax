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
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.cp_utils import CPContext
from tokamax._src.ops.experimental.kda.pallas_tpu_bwd import (
    chunk_kda_bwd_custom,
)
from tokamax._src.ops.experimental.kda.pallas_tpu_fwd import (
    chunk_kda_fwd_custom,
    chunk_kda_fwd,
)
from tokamax._src.ops.experimental.kda.utils import (
    as_public_final_state,
    derive_cp_context,
    l2norm_fwd,
    normalize_initial_state,
    segment_ids_to_cu_seqlens,
)
from typing_extensions import override


_NONDIFF_ARGNUMS = (7, 9, 10, 11, 13, 14, 15, 16, 17, 18)


@functools.partial(jax.custom_vjp, nondiff_argnums=_NONDIFF_ARGNUMS)
def chunk_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids: jax.Array | None = None,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    disable_recompute: bool = True,
    cp_context: CPContext | None = None,
    chunk_size: int = 64,
    N_max: int | None = None,
):
  """Head-first chunk KDA with pallas-kernel custom VJP."""
  H, B, T, K = q.shape
  V = v.shape[-1]
  initial_state = normalize_initial_state(
      initial_state, batch=B, heads=H, key_dim=K, value_dim=V
  )
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
        seq_len=T,
  )
  actual_scale = scale if scale is not None else K**-0.5
  if use_qk_l2norm_in_kernel:
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

  output, final_state, *_ = chunk_kda_fwd(
      q,
      k,
      v,
      g,
      beta,
      A_log=A_log,
      dt_bias=dt_bias,
      scale=actual_scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=False,
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


chunk_kda.defvjp(chunk_kda_fwd_custom, chunk_kda_bwd_custom)


@dataclasses.dataclass(frozen=True)
class PallasTpuKimiDeltaAttention(base.KimiDeltaAttention):
  """Pallas TPU KDA backend.

  This adapter preserves Tokamax's experimental head-first KDA contract:
  inputs are `[H, B, T, D]` and recurrent states are `[B, N, H, K, V]`.
  """

  chunk_size: int = 64

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
    del config, return_residuals  # Unused.

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

    output, final_state = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        segment_ids,
        safe_gate,
        lower_bound,
        disable_recompute,
        cp_context,
        chunk_size,
        N_max,
    )

    return (output.astype(q.dtype), final_state), None
