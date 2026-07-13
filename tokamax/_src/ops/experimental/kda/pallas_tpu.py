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
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    _derive_cp_metadata_from_segment_ids,
)
from tokamax._src.ops.experimental.kda.pallas_tpu_bwd import chunk_kda_bwd
from tokamax._src.ops.experimental.kda.pallas_tpu_fwd import (
    _align_seqs,
    _unalign_output,
    chunk_kda_fwd as pallas_chunk_kda_fwd,
)
from tokamax._src.ops.experimental.kda.utils import (
    align_segment_ids,
    cdiv,
    compute_padded_cu_seqlens,
    prepare_chunk_indices,
    segment_ids_to_seqlens,
)
from typing_extensions import override


_NONDIFF_ARGNUMS = (7, 9, 10, 11, 13, 14, 15, 16, 17, 18)


def _l2norm_fwd(x: jax.Array, eps: float = 1e-6):
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + eps)
  return (x_f * rstd[..., None]).astype(x.dtype), rstd.astype(jnp.float32)


def _l2norm_bwd(y: jax.Array, rstd: jax.Array, dy: jax.Array):
  y_f = y.astype(jnp.float32)
  dy_f = dy.astype(jnp.float32)
  rstd_f = rstd.astype(jnp.float32)
  dot_dy_y = jnp.sum(dy_f * y_f, axis=-1)
  dx = dy_f * rstd_f[..., None] - dot_dy_y[..., None] * y_f * rstd_f[..., None]
  return dx.astype(y.dtype)


def _check_segment_ids(segment_ids: jax.Array | None, batch: int, seq_len: int):
  if segment_ids is None:
    return
  if segment_ids.shape != (batch, seq_len):
    raise ValueError(
        "`segment_ids` must have shape [B, T]; got "
        f"{segment_ids.shape}, expected {(batch, seq_len)}."
    )


def _normalize_initial_state(
    initial_state: jax.Array | None,
    *,
    batch: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> jax.Array | None:
  if initial_state is None:
    return None
  if initial_state.ndim != 5:
    raise ValueError(
        "`initial_state` must have shape [B, N, H, K, V]; got "
        f"{initial_state.shape}."
    )
  if initial_state.shape[0] != batch:
    raise ValueError(
        f"`initial_state` batch dimension must be {batch}; got {initial_state.shape}."
    )
  if initial_state.shape[2:] != (heads, key_dim, value_dim):
    raise ValueError(
        "`initial_state` trailing dimensions must be "
        f"{(heads, key_dim, value_dim)}; got {initial_state.shape[2:]}."
    )
  return initial_state


def _as_public_final_state(
    final_state: jax.Array | None,
    *,
    segment_ids: jax.Array | None,
) -> jax.Array | None:
  if final_state is None:
    return None
  if final_state.ndim == 4 and segment_ids is None:
    return final_state[:, None]
  return final_state


def _derive_cp_context(
    *,
    q: jax.Array,
    segment_ids: jax.Array | None,
    initial_state: jax.Array | None,
    output_final_state: bool,
    cp_context: CPContext | None,
    chunk_size: int,
    N_max: int | None,
) -> tuple[CPContext | None, jax.Array | None]:
  cu_seqlens = None
  if cp_context is None or not cp_context.is_cp_enabled:
    return cp_context, cu_seqlens

  if initial_state is not None:
    raise ValueError("`initial_state` is not supported when CP is enabled.")
  if output_final_state:
    raise ValueError("`output_final_state` is not supported when CP is enabled.")
  if segment_ids is None:
    raise ValueError("CP requires rank-local `segment_ids` with shape [B, T].")

  n_max = N_max if N_max is not None else cdiv(q.shape[2], chunk_size)
  cu_locals, chain_metas = [], []
  for b in range(segment_ids.shape[0]):
    cu_b, meta_b = _derive_cp_metadata_from_segment_ids(
        segment_ids[b],
        cp_context.axis_name,
        n_max=n_max,
    )
    cu_locals.append(cu_b)
    chain_metas.append(meta_b)
  cu_seqlens = jnp.stack(cu_locals, axis=0)
  chain_meta = {k: jnp.stack([m[k] for m in chain_metas]) for k in chain_metas[0]}
  cp_context = dataclasses.replace(
      cp_context,
      is_first_rank=chain_meta["is_first_rank"],
      is_last_rank=chain_meta["is_last_rank"],
      pre_num_ranks=chain_meta["pre_num_ranks"],
      post_num_ranks=chain_meta["post_num_ranks"],
  )
  return cp_context, cu_seqlens


def _segment_ids_to_cu_seqlens(
    segment_ids: jax.Array | None,
    *,
    initial_state: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
    seq_len: int,
) -> tuple[jax.Array | None, int | None]:
  if segment_ids is None:
    return None, N_max
  if N_max is None:
    N_max = initial_state.shape[1] if initial_state is not None else cdiv(seq_len, chunk_size)
  return segment_ids_to_seqlens(segment_ids, max_segs=N_max), N_max


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
  _check_segment_ids(segment_ids, B, T)
  initial_state = _normalize_initial_state(
      initial_state, batch=B, heads=H, key_dim=K, value_dim=V
  )
  cp_context, cu_seqlens = _derive_cp_context(
      q=q,
      segment_ids=segment_ids,
      initial_state=initial_state,
      output_final_state=output_final_state,
      cp_context=cp_context,
      chunk_size=chunk_size,
      N_max=N_max,
  )
  if cu_seqlens is None:
    cu_seqlens, _ = _segment_ids_to_cu_seqlens(
        segment_ids,
        initial_state=initial_state,
        chunk_size=chunk_size,
        N_max=N_max,
        seq_len=T,
    )
  actual_scale = scale if scale is not None else K**-0.5
  if use_qk_l2norm_in_kernel:
    q, _ = _l2norm_fwd(q)
    k, _ = _l2norm_fwd(k)

  output, final_state, *_ = pallas_chunk_kda_fwd(
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
  return output.astype(q.dtype), _as_public_final_state(
      final_state, segment_ids=segment_ids
  )


def _chunk_kda_fwd_custom(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    segment_ids=None,
    safe_gate=True,
    lower_bound=None,
    disable_recompute=True,
    cp_context=None,
    chunk_size=64,
    N_max=None,
):
  H, B, T, K = q.shape
  V = v.shape[-1]
  _check_segment_ids(segment_ids, B, T)
  initial_state = _normalize_initial_state(
      initial_state, batch=B, heads=H, key_dim=K, value_dim=V
  )

  cp_context, cu_seqlens = _derive_cp_context(
      q=q,
      segment_ids=segment_ids,
      initial_state=initial_state,
      output_final_state=output_final_state,
      cp_context=cp_context,
      chunk_size=chunk_size,
      N_max=N_max,
  )
  if cu_seqlens is None:
    cu_seqlens, N_max = _segment_ids_to_cu_seqlens(
        segment_ids,
        initial_state=initial_state,
        chunk_size=chunk_size,
        N_max=N_max,
        seq_len=T,
    )
  actual_scale = scale if scale is not None else K**-0.5

  ori_cu_seqlens = cu_seqlens
  aligned_cu = None
  if cu_seqlens is not None:
    [q_a, k_a, v_a, g_a], [beta_a], aligned_cu, _ = _align_seqs(
        [q, k, v, g],
        [beta],
        cu_seqlens,
        align=chunk_size,
    )
    aligned_cu = compute_padded_cu_seqlens(ori_cu_seqlens, chunk_size)
    if use_gate_in_kernel:
      T_a = g_a.shape[2]
      orig_lens = jnp.diff(cu_seqlens, axis=-1)
      aligned_starts = aligned_cu[..., :-1]
      pos = jnp.arange(T_a)
      for b in range(cu_seqlens.shape[0]):
        in_range = (pos[None, :] >= aligned_starts[b, :, None]) & (
            pos[None, :] < (aligned_starts[b] + orig_lens[b])[:, None]
        )
        valid_mask = in_range.any(axis=0)
        g_a = g_a.at[:, b].set(
            jnp.where(valid_mask[None, :, None], g_a[:, b], -1e4)
        )
  else:
    q_a, k_a, v_a, g_a, beta_a = q, k, v, g, beta

  segment_ids_aligned = None
  if cu_seqlens is not None and segment_ids is not None:
    segment_ids_aligned = jnp.stack(
        [
            align_segment_ids(segment_ids[b], N_max, chunk_size)
            for b in range(segment_ids.shape[0])
        ]
    )

  if use_qk_l2norm_in_kernel:
    q_hat, rstd_q = _l2norm_fwd(q_a)
    k_hat, rstd_k = _l2norm_fwd(k_a)
  else:
    q_hat, k_hat = q_a, k_a
    rstd_q = rstd_k = None

  (
      output,
      final_state,
      g_cumsum,
      Aqk,
      Akk,
      _w,
      _u,
      _qg,
      _kg,
      _v_new,
      h,
      initial_state,
  ) = pallas_chunk_kda_fwd(
      q_hat,
      k_hat,
      v_a,
      g_a,
      beta_a,
      A_log=A_log,
      dt_bias=dt_bias,
      scale=actual_scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=False,
      use_gate_in_kernel=use_gate_in_kernel,
      cu_seqlens=aligned_cu,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      disable_recompute=disable_recompute,
      cp_context=cp_context,
      chunk_size=chunk_size,
      _skip_align=True,
  )

  if aligned_cu is not None:
    output = _unalign_output(output, ori_cu_seqlens, aligned_cu, T)

  g_org = g_a if use_gate_in_kernel else None
  g_cumsum = checkpoint_name(g_cumsum, "kda_residuals")
  Aqk = checkpoint_name(Aqk, "kda_residuals")
  Akk = checkpoint_name(Akk, "kda_residuals")
  if disable_recompute and h is not None:
    h = checkpoint_name(h, "kda_residuals")

  residuals = (
      q_hat,
      k_hat,
      v_a,
      beta_a,
      g_cumsum,
      Aqk,
      Akk,
      initial_state,
      g_org,
      A_log,
      dt_bias,
      h,
      jnp.zeros((), dtype=g.dtype),
      rstd_q,
      rstd_k,
      ori_cu_seqlens,
      aligned_cu,
      segment_ids_aligned,
      segment_ids,
      initial_state is not None,
  )
  return (
      output.astype(q.dtype),
      _as_public_final_state(final_state, segment_ids=segment_ids),
  ), residuals


def _chunk_kda_bwd_custom(
    scale,
    output_final_state,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    safe_gate,
    lower_bound,
    disable_recompute,
    cp_context,
    chunk_size,
    N_max,
    residuals,
    grad_outputs,
):
  del output_final_state
  do, dht = grad_outputs
  (
      q,
      k,
      v,
      beta,
      g_cumsum,
      Aqk,
      Akk,
      initial_state,
      g_org,
      A_log,
      dt_bias,
      h,
      g_dtype_marker,
      rstd_q,
      rstd_k,
      ori_cu_seqlens,
      aligned_cu,
      segment_ids_aligned,
      segment_ids,
      has_initial_state,
  ) = residuals

  cu_seqlens_bwd = None
  chunk_indices_bwd = None
  T_orig = None
  if ori_cu_seqlens is not None:
    T_orig = do.shape[2]
    [do], [], _, _ = _align_seqs([do], [], ori_cu_seqlens, align=chunk_size)
    cu_seqlens_bwd = compute_padded_cu_seqlens(ori_cu_seqlens, chunk_size)
    chunk_indices_bwd = prepare_chunk_indices(
        cu_seqlens_bwd, chunk_size, max_T=q.shape[2]
    )

  if cp_context is not None and cp_context.is_cp_enabled:
    if segment_ids is None:
      raise ValueError("backward CP requires rank-local `segment_ids`.")
    n_max = N_max if N_max is not None else cdiv(q.shape[2], chunk_size)
    chain_metas = []
    for b in range(segment_ids.shape[0]):
      _, meta_b = _derive_cp_metadata_from_segment_ids(
          segment_ids[b],
          cp_context.axis_name,
          n_max=n_max,
      )
      chain_metas.append(meta_b)
    chain_meta = {k: jnp.stack([m[k] for m in chain_metas]) for k in chain_metas[0]}
    needed = ("post_num_ranks", "is_last_rank", "pre_num_ranks", "is_first_rank")
    if any(getattr(cp_context, name) is None for name in needed):
      cp_fields = {field.name for field in dataclasses.fields(cp_context)}
      updates = {
          key: value
          for key, value in chain_meta.items()
          if key in cp_fields and getattr(cp_context, key) is None
      }
      cp_context = dataclasses.replace(cp_context, **updates)

  bwd_kwargs = dict(
      g=g_cumsum,
      g_org=g_org,
      cu_seqlens=cu_seqlens_bwd,
      chunk_indices=chunk_indices_bwd,
      chunk_size=chunk_size,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      use_gate_in_kernel=use_gate_in_kernel,
      A_log=A_log,
      dt_bias=dt_bias,
      disable_recompute=disable_recompute,
      cp_context=cp_context,
      segment_ids=segment_ids_aligned if segment_ids_aligned is not None else segment_ids,
  )
  if disable_recompute and h is not None:
    bwd_kwargs["h"] = h

  actual_scale = scale if scale is not None else q.shape[-1] ** -0.5
  dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
      q,
      k,
      v,
      beta,
      Aqk,
      Akk,
      actual_scale,
      initial_state,
      do,
      dht,
      N_max=N_max,
      **bwd_kwargs,
  )

  if use_qk_l2norm_in_kernel:
    dq = _l2norm_bwd(q, rstd_q, dq)
    dk = _l2norm_bwd(k, rstd_k, dk)

  if ori_cu_seqlens is not None:
    dq = _unalign_output(dq, ori_cu_seqlens, aligned_cu, T_orig)
    dk = _unalign_output(dk, ori_cu_seqlens, aligned_cu, T_orig)
    dv = _unalign_output(dv, ori_cu_seqlens, aligned_cu, T_orig)
    dg = _unalign_output(dg, ori_cu_seqlens, aligned_cu, T_orig)
    db = _unalign_output(db, ori_cu_seqlens, aligned_cu, T_orig)

  if dh0 is not None and dh0.ndim == 4 and has_initial_state:
    dh0 = dh0[:, None]

  return (
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      dg.astype(g_dtype_marker.dtype),
      db.astype(beta.dtype),
      dA,
      dbias,
      dh0,
      None,
  )


chunk_kda.defvjp(_chunk_kda_fwd_custom, _chunk_kda_bwd_custom)


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
