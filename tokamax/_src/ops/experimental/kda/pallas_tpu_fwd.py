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
"""Pallas TPU forward kernels for experimental KDA."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member

from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    _merge_initial_state,
    all_gather_into_tensor,
)
from tokamax._src.ops.experimental.kda.utils import (
    align_segment_ids,
    align_up,
    as_public_final_state,
    assert_shape,
    assert_shape_or_none,
    cdiv,
    compute_padded_cu_seqlens,
    derive_cp_context,
    exp2,
    get_interpret,
    get_tpu_config,
    l2norm_fwd,
    normalize_initial_state,
    prepare_chunk_indices,
    segment_ids_to_cu_seqlens,
    segment_ids_to_seqlens,
)

_RCP_LN2 = 1.0 / math.log(2)


# =============================================================================
# Mini-batch sizing
# =============================================================================

def estimate_mini_batch(
    per_tile_bytes: int,
    total: int,
    *,
    max_mb: int = 16,
    vmem_budget: int | None = None,
    align_minor: int | None = None,
) -> int:
  """Estimate the optimal mini-batch size to maximise VMEM utilisation.

  This mirrors the backward auto-tune pattern used throughout the KDA kernels:
  compute the largest MB that fits within the hardware VMEM budget, cap it,
  and then adjust downward so that ``total`` is evenly divisible by ``MB``.

  Args:
      per_tile_bytes: Estimated VMEM footprint (bytes) for **one** tile/head.
      total: Number of tiles (or heads) to partition.
      max_mb: Upper bound on MB (default 16).
      vmem_budget: VMEM budget in bytes. ``None`` (default) queries
          ``get_tpu_config().vmem_limit_bytes`` at call time.
      align_minor: TPU block_align_minor constraint. ``None`` (default)
          queries ``get_tpu_config().block_align_minor``.  When provided,
          the function prefers MB values that satisfy this alignment.

  Returns:
      Mini-batch size ``MB`` such that ``total % MB == 0`` (best-effort).
  """
  if vmem_budget is None or align_minor is None:
    hw = get_tpu_config()
    if vmem_budget is None:
      vmem_budget = hw.vmem_limit_bytes
    if align_minor is None:
      align_minor = hw.block_align_minor

  per_tile_bytes = max(1, per_tile_bytes)
  MB = max(1, vmem_budget // per_tile_bytes)
  MB = max(1, min(MB, total, max_mb))

  # Try to find an MB that divides total evenly.
  while total % MB != 0 and MB > 1:
    MB -= 1

  return MB


# =============================================================================
# Gate cumsum used by the float32 fallback
# =============================================================================

def _chunk_local_cumsum_matmul(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Chunk-local cumsum via triangular matmul (best for ``head_first=True``).

  Args:
      g:            [B, T, H, S] or [H, B, T, S] — input gates.
      chunk_size:   block size (must be power of 2).
      reverse:      if True, compute reverse (suffix) cumsum.
      scale:        optional multiplicative scale applied to the output.
      head_first:   if True, ``g`` is [H, B, T, S]; otherwise [B, T, H, S].
      output_dtype: dtype of the output tensor (default float32).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum.
  """
  BT = chunk_size
  out_dtype = output_dtype or g.dtype

  if head_first:
    H, B, T, S = g.shape
  else:
    B, T, H, S = g.shape

  NT = (T + BT - 1) // BT
  T_padded = NT * BT
  pad_t = T_padded - T

  if head_first:
    g_work = jnp.pad(g, ((0, 0), (0, 0), (0, pad_t), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(H, B, NT, BT, S).astype(jnp.float32)
  else:
    g_work = jnp.pad(g, ((0, 0), (0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(B, NT, BT, H, S).astype(jnp.float32)

  if reverse:
    cum_mask = jnp.triu(jnp.ones((BT, BT), dtype=jnp.float32))
  else:
    cum_mask = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))

  if head_first:
    o_chunked = jnp.einsum(
      "ij,hbnjs->hbnis",
      cum_mask,
      g_chunked,
      precision=jax.lax.Precision.HIGHEST,
    )
    o = o_chunked.reshape(H, B, T_padded, S)[:, :, :T, :]
  else:
    o_chunked = jnp.einsum(
      "ij,bnjhs->bnihs",
      cum_mask,
      g_chunked,
      precision=jax.lax.Precision.HIGHEST,
    )
    o = o_chunked.reshape(B, T_padded, H, S)[:, :T, :, :]

  if scale is not None:
    o = o * scale

  return o.astype(out_dtype)


@functools.partial(
  jax.jit,
  static_argnames=["chunk_size", "reverse", "scale", "head_first", "output_dtype"],
)
def chunk_local_cumsum_vector(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Chunk-local cumulative sum of gates via triangular matmul.

  Args:
      g:              [B, T, H, S] or [H, B, T, S] — input gates.
      chunk_size:     block size along T (must be power of 2).
      reverse:        if True, compute reverse (suffix) cumsum within each chunk.
      scale:          optional multiplicative scale applied to the output.
      head_first:     if True, ``g`` layout is [H, B, T, S]; otherwise [B, T, H, S].
      output_dtype:   dtype of the output tensor (default float32).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum of the input gates.
  """
  # =================== assert kernel requirements start ===================
  assert g.ndim == 4, f"g must be 4-D, got {g.ndim}-D"
  assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
    "chunk_size must be power of 2"
  )
  # =================== assert kernel requirements done ====================

  return _chunk_local_cumsum_matmul(
    g,
    chunk_size,
    reverse,
    scale,
    head_first,
    output_dtype,
  )


# =============================================================================
# Context-parallel boundary pre-process
# =============================================================================

def _pre_process_kernel(
  seqlens_ref,        # scalar prefetch: cu_seqlens [N+1]
  chunk_to_seq_ref,   # scalar prefetch: chunk -> seq mapping [NT]
  last_seg_idx_ref,   # scalar prefetch: real last seg idx [1], int32
  k_ref,    # [MB, 1, BT, K_PADSIZE]
  w_ref,    # [MB, 1, BT, K_PADSIZE]
  u_ref,    # [MB, 1, BT, V_ALIGNED]
  gk_ref,   # [MB, 1, BT, K_PADSIZE]
  S_ext_ref,  # [MB, 1, K_PADSIZE, V_ALIGNED]
  M_ref,      # [MB, 1, K_PADSIZE, K_PADSIZE]
  h_scratch,  # [MB, K_PADSIZE, V_ALIGNED]
  m_scratch,  # [MB, K_PADSIZE, K_PADSIZE]
  *,
  BT,
  MB,
):
  """Fused (S_ext, M) pre-process Pallas kernel for KDA CP forward.

  Pattern B (design doc §3.5): MB heads as batch dim of ``dot_general``
  (no Python unroll). For each program point (h_group, i_c):
    - seq_idx = chunk_to_seq[i_c] — which segment this chunk belongs to.
    - last_seg_idx = scalar prefetch from launcher — the **REAL** last
      segment idx (largest i where ``cu_seqlens[i+1] > cu_seqlens[i]``).
      Computed in the launcher rather than reading ``chunk_to_seq[NT-1]``
      because ``prepare_chunk_indices``'s ``jnp.repeat(total_repeat_length=...)``
      pads trailing slots with the **last input value** (= the last
      trailing-padding seg id when ``cu_seqlens`` has N_max padding),
      which would mis-identify the real last seg.
    - Only chunks of the LAST segment contribute to (S_ext, M); others
      no-op (BlockSpec DMA still happens but compute is skipped).

  Updates per chunk (last segment only):
    M_c    = Diag(exp2(gk_last_c)) - K_c^T @ W_c        # [MB, K, K]
    dS_c   = K_c^T @ U_c                                # [MB, K, V]
    h_acc  = M_c @ h_acc + dS_c                         # [MB, K, V]
    m_acc  = M_c @ m_acc                                # [MB, K, K]
  """
  i_c = pl.program_id(1)
  seq_idx = chunk_to_seq_ref[i_c]
  last_seg_idx = last_seg_idx_ref[0]
  bos = seqlens_ref[seq_idx]
  eos = seqlens_ref[seq_idx + 1]
  t0 = i_c * BT

  K = k_ref.shape[-1]
  V = u_ref.shape[-1]
  eye_k = jnp.eye(K, dtype=jnp.float32)

  # ── Init scratch at LAST segment's first chunk ──
  @pl.when((t0 == bos) & (seq_idx == last_seg_idx))
  def _init():
    h_scratch[...] = jnp.zeros((MB, K, V), dtype=jnp.float32)
    m_scratch[...] = jnp.broadcast_to(eye_k, (MB, K, K))

  # ── Update only on LAST segment chunks ──
  @pl.when(seq_idx == last_seg_idx)
  def _update():
    K_all = k_ref[:,0,:].astype(jnp.float32)                       # [MB, BT, K]
    W_all = w_ref[:,0,:].astype(jnp.float32)                       # [MB, BT, K]
    U_all = u_ref[:,0,:].astype(jnp.float32)                       # [MB, BT, V]
    gk_last_all = gk_ref[:, 0, BT - 1].astype(jnp.float32)     # [MB, K]

    # M_c per head: Diag(exp2(gk_last)) - K^T @ W
    decay_all = jnp.exp2(jnp.maximum(gk_last_all, -126.0))     # [MB, K]
    diag_all = decay_all[..., None] * eye_k                    # [MB, K, K]

    # K^T @ W: contract BT (axis 1), batch MB (axis 0) -> [MB, K, K]
    KW_all = jax.lax.dot_general(
      K_all, W_all,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    )
    M_all = diag_all - KW_all                                  # [MB, K, K]

    # dS_c = K^T @ U: contract BT, batch MB -> [MB, K, V]
    dS_all = jax.lax.dot_general(
      K_all, U_all,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    )

    # h_acc = M @ h_acc + dS: M [MB, K_out, K_in] @ h [MB, K_in, V] -> [MB, K_out, V]
    # contract K_in (M.axis 2 / h.axis 1), batch MB
    h_new = jax.lax.dot_general(
      M_all, h_scratch[...],
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    ) + dS_all                                                 # [MB, K, V]
    m_new = jax.lax.dot_general(
      M_all, m_scratch[...],
      (((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
    )                                                          # [MB, K, K]
    h_scratch[...] = h_new
    m_scratch[...] = m_new

  # ── Write outputs at LAST segment's last chunk ──
  @pl.when((t0 + BT >= eos) & (seq_idx == last_seg_idx))
  def _store():
    S_ext_ref[:,0,:] = h_scratch[...].astype(S_ext_ref.dtype)
    M_ref[:,0,:] = m_scratch[...].astype(M_ref.dtype)


def _pre_process_pallas(
  k: jax.Array,    # [H, B=1, T_local, K]
  w: jax.Array,    # [H, B=1, T_local, K]
  u: jax.Array,    # [H, B=1, T_local, V]
  gk: jax.Array,   # [H, B=1, T_local, K]
  cu_seqlens: jax.Array,
  chunk_indices: jax.Array,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
  """Pallas launcher for (S_ext, M) pre-process. Returns shape
  ``([1, H, K, V], [1, H, K, K])`` both fp32.

  Assumes ``T_local`` is already a multiple of ``chunk_size`` (caller's
  ``_align_seqs`` has run) and ``chunk_indices = prepare_chunk_indices(...)``
  has been pre-computed by the caller.
  """
  H, B, T_local, K = k.shape
  V = u.shape[-1]
  BT = chunk_size

  # Handle B > 1 by looping over batch elements with the B=1 kernel.
  if B > 1:
    s_list, m_list = [], []
    for b in range(B):
      # chunk_indices: [B, NT, 2] → [NT, 2]
      ci_b = chunk_indices[b] if chunk_indices.ndim == 3 else chunk_indices
      cu_b = cu_seqlens[b] if cu_seqlens.ndim == 2 else cu_seqlens
      s_b, m_b = _pre_process_pallas(
        k=k[:, b:b+1], w=w[:, b:b+1], u=u[:, b:b+1], gk=gk[:, b:b+1],
        cu_seqlens=cu_b, chunk_indices=ci_b, chunk_size=BT,
      )
      s_list.append(s_b)
      m_list.append(m_b)
    return jnp.concatenate(s_list, axis=1), jnp.concatenate(m_list, axis=1)

  # B=1 path: squeeze cu_seqlens to 1D and chunk_indices to 2D if needed
  if cu_seqlens.ndim == 2:
    assert cu_seqlens.shape[0] == 1, f"B=1 but cu_seqlens has shape {cu_seqlens.shape}"
    cu_seqlens = cu_seqlens[0]
  if chunk_indices.ndim == 3:
    assert chunk_indices.shape[0] == 1, f"B=1 but chunk_indices has shape {chunk_indices.shape}"
    chunk_indices = chunk_indices[0]
  NT = len(chunk_indices)
  assert T_local % BT == 0
  assert K <= 256, "pre_process does not support K > 256"

  K_PADSIZE = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))

  # ── Auto MB: maximise VMEM utilisation, capped at min(H, 16) ──
  # scratch per head: h_acc [K, V] + m_acc [K, K], all fp32.
  per_head_scratch = (K_PADSIZE * V_ALIGNED + K_PADSIZE * K_PADSIZE) * 4
  vmem_budget = 8 * 1024 * 1024
  MB = max(1, vmem_budget // per_head_scratch)
  MB = min(MB, H, 16)
  while H % MB != 0 and MB > 1:
    MB -= 1

  # ── Pad K/V then T (reserve trailing BT for safe gather) ──

  def _pad_kdim_then_t(x, dim_pad):
    if dim_pad > 0:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim_pad)))
    return x  # [H, B, T_alloc, D]

  k_t = _pad_kdim_then_t(k.astype(jnp.float32), K_PADSIZE - K)
  w_t = _pad_kdim_then_t(w.astype(jnp.float32), K_PADSIZE - K)
  u_t = _pad_kdim_then_t(u.astype(jnp.float32), V_ALIGNED - V)
  gk_t = _pad_kdim_then_t(gk.astype(jnp.float32), K_PADSIZE - K)

  chunk_to_seq = chunk_indices[:, 0].astype(jnp.int32)

  # ── Real last seg idx (largest i where cu_seqlens[i+1] > cu_seqlens[i]) ──
  # MUST NOT derive from chunk_to_seq[NT-1] inside the kernel: when
  # cu_seqlens is N_max-padded with trailing zero-length segments,
  # prepare_chunk_indices's `jnp.repeat(total_repeat_length=...)` pads
  # the trailing chunk_to_seq slots with the LAST input seg id (= the
  # trailing-padding seg id), which would point at a phantom segment.
  # Computing real_last_seg_idx here from cu_seqlens directly is robust
  # to any N_max padding scheme.
  seg_lens = jnp.diff(cu_seqlens)  # [N]
  seg_indices = jnp.arange(seg_lens.shape[0], dtype=jnp.int32)
  real_last_seg_idx = jnp.maximum(
    jnp.max(jnp.where(seg_lens > 0, seg_indices, jnp.int32(-1))),
    jnp.int32(0),
  ).astype(jnp.int32)
  real_last_seg_idx_arr = real_last_seg_idx[None]  # [1] for scalar prefetch

  def _in_index_map(h, c, seqlens_ref, chunk_to_seq_ref, last_seg_idx_ref):
    return (h, 0, c, 0)

  def _out_index_map(h, c, seqlens_ref, chunk_to_seq_ref, last_seg_idx_ref):
    return (h, 0, 0, 0)

  bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_in_index_map)
  bspec_u = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_in_index_map)
  S_ext_spec = pl.BlockSpec(
    [MB, 1, K_PADSIZE, V_ALIGNED], index_map=_out_index_map
  )
  M_spec = pl.BlockSpec(
    [MB, 1, K_PADSIZE, K_PADSIZE], index_map=_out_index_map
  )

  S_ext_shape = jax.ShapeDtypeStruct(
    [H, 1, K_PADSIZE, V_ALIGNED], jnp.float32
  )
  M_shape = jax.ShapeDtypeStruct(
    [H, 1, K_PADSIZE, K_PADSIZE], jnp.float32
  )

  scratch_shapes = [
    pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32),  # h_acc
    pltpu.VMEM((MB, K_PADSIZE, K_PADSIZE), jnp.float32),  # m_acc
  ]
  grid = (H // MB, NT)
  interpret = get_interpret()

  S_ext_pad, M_pad = pl.pallas_call(
    functools.partial(_pre_process_kernel, BT=BT, MB=MB),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=3,
      grid=grid,
      in_specs=[bspec_k, bspec_k, bspec_u, bspec_k],
      out_specs=[S_ext_spec, M_spec],
      scratch_shapes=scratch_shapes,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "arbitrary"),
    ),
    out_shape=[S_ext_shape, M_shape],
    interpret=interpret,
  )(cu_seqlens.astype(jnp.int32), chunk_to_seq, real_last_seg_idx_arr,
    k_t, w_t, u_t, gk_t)

  # Trim K/V padding back to original shape.
  S_ext = S_ext_pad[..., :K, :V]
  M = M_pad[..., :K, :K]
  return S_ext, M


def chunk_gated_delta_rule_fwd_h_pre_process(
  k: jax.Array,  # [B, T_local, H, K]
  w: jax.Array,  # [B, T_local, H, K]
  u: jax.Array,  # [B, T_local, H, V]
  gk: jax.Array,  # [B, T_local, H, K] -- per-element gate in log2 space
  cu_seqlens: jax.Array,
  chunk_indices: jax.Array | None = None,
  chunk_size: int = 64,
  use_exp2: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Compute ``(S_ext, M)`` for the LAST rank-local segment under CP.

  Used by KDA forward context parallel (design-doc §2.2). Each rank runs
  this assuming ``S_in = 0``; the resulting tensors are then all-gathered
  and merged via ``_merge_initial_state`` to recover the true ``S_in`` for
  the FIRST segment on every rank.

  Why only the LAST segment? Segments fully within a rank start fresh and
  finish within this rank — they need no cross-rank communication. Only
  the last segment of rank ``r`` may continue into rank ``r+1``, so only
  its boundary state matters. Mirrors FLA passing
  ``cu_seqlens=cu_seqlens[-2:]`` in
  ``fla/ops/cp/chunk_delta_h.py::chunk_gated_delta_rule_fwd_h_pre_process``.

  Implementation: single fused Pallas TPU kernel ``_pre_process_kernel``
  (design-doc ``cp_pre_process_pallas.aligned.zh.md`` §3) that computes
  ``(S_ext, M)`` in one pass — shares ``K_c, W_c, gk_c`` VMEM loads and
  ``M_c`` compute across both updates. Pattern B (batched ``dot_general``
  with MB as batch dim) replaces the prior two-step path
  (``chunk_gated_delta_rule_fwd_h`` + JAX ``fori_loop`` over K×K matmuls).

  The kernel uses ``chunk_to_seq[NT - 1]`` to identify the LAST real
  segment (scalar prefetch SMEM read) — semantically equivalent to
  ``max idx where seg_lens > 0`` since ``prepare_chunk_indices`` skips
  empty trailing segments.

  No host-side ``cu_seqlens_cpu`` parameter is required: the kernel
  operates on a traced ``cu_seqlens``, so it works inside ``jit`` /
  ``shard_map`` where host-side Python int indexing into a traced array
  is impossible.

  Args:
    k: ``[H, B, T_local, K]`` -- gated keys (``kg`` in KDA notation).
    w: ``[H, B, T_local, K]`` -- WY-representation correction weights.
    u: ``[H, B, T_local, V]`` -- delta-corrected values from intra-chunk.
    gk: ``[H, B, T_local, K]`` -- chunk-local cumsum of the log-space gate.
      Must be fp32 (or fp32-promotable).
    cu_seqlens: ``[N_local + 1]`` int32 -- rank-local cumulative seq lengths.
      Each segment's length must already be a multiple of ``chunk_size``
      (i.e. the caller's ``_align_seqs`` has run). Can be a traced jnp
      array; only the leading-dim shape ``N_local + 1`` is read at trace
      time.
    chunk_indices: ``[NT, 2]`` int32 -- optional precomputed chunk indices
      (output of ``prepare_chunk_indices(cu_seqlens, chunk_size)``).
      Pass it when the caller has already computed it (avoids redundant
      work). ``None`` → compute internally.
    chunk_size: Tile size ``BT`` (default 64).
    use_exp2: ``True`` for KDA (gates are in log2 space).

  Returns:
    Tuple ``(S_ext, M)``:
      - ``S_ext``: ``[B, H, K, V]`` fp32 -- accumulated state from the last
        segment, assuming ``S_in = 0``.
      - ``M``:     ``[B, H, K, K]`` fp32 -- chain transition matrix for the
        last segment.
  """
  H, B, T_local, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  N_local = cu_seqlens.shape[-1] - 1

  assert_shape(k, (H, B, T_local, K), "k")
  assert_shape(w, (H, B, T_local, K), "w")
  assert_shape(u, (H, B, T_local, V), "u")
  assert_shape(gk, (H, B, T_local, K), "gk")
  assert use_exp2, "KDA pre-process requires use_exp2=True (gates are log2 space)"
  assert K <= 256, (
    "current pre-process does not support head dimension larger than 256."
  )
  assert T_local % BT == 0, (
    f"T_local={T_local} must be divisible by chunk_size={BT}; "
    f"caller must pre-align via _align_seqs."
  )
  assert N_local >= 1, (
    f"cu_seqlens must have at least 2 entries (one segment); got "
    f"shape {cu_seqlens.shape}"
  )

  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T_local)

  S_ext, M = _pre_process_pallas(
    k=k, w=w, u=u, gk=gk,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    chunk_size=BT,
  )

  assert S_ext.shape == (H, B, K, V), (
    f"S_ext shape mismatch: expected {(H, B, K, V)}, got {S_ext.shape}"
  )
  assert M.shape == (H, B, K, K), (
    f"M shape mismatch: expected {(H, B, K, K)}, got {M.shape}"
  )
  assert S_ext.dtype == jnp.float32, f"S_ext must be fp32, got {S_ext.dtype}"
  assert M.dtype == jnp.float32, f"M must remain fp32, got {M.dtype}"
  return S_ext, M


# =============================================================================
# Gate activation
# =============================================================================

def kda_gate_chunk_cumsum(
  g: jax.Array,
  A_log: jax.Array,
  chunk_size: int,
  scale: float | None = None,
  dt_bias: jax.Array | None = None,
  output_dtype: jnp.dtype | None = jnp.float32,
  lower_bound: float | None = None,
) -> jax.Array:
  """Fused KDA gate activation + chunk-local cumulative sum.

  Applies the KDA gate activation to raw gate inputs, then computes
  chunk-local cumsum. The two gate variants are:

  Standard (lower_bound is None):
      g_act = -exp(A_log[h]) * softplus(g + dt_bias)

  Lower-bound (lower_bound is not None):
      g_act = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias))

  Then chunk-local cumsum is applied, optionally scaled.

  Mirrors ``kda_gate_chunk_cumsum`` from ``fla.ops.kda.gate``.

  Args:
      g:          [H, B, T, K] -- raw gate input.
      A_log:      [H] -- log of the diagonal decay parameter A, one per head.
      chunk_size: int -- chunk size BT. Must be power of 2.
      scale:      float or None -- multiplicative scale after cumsum
                  (typically RCP_LN2 = 1/ln2 to convert to log2 space).
      dt_bias:    [H*K] or None -- optional bias added to g before activation.
      output_dtype: dtype for output (default float32).
      lower_bound: float or None -- if set, use sigmoid variant instead of
                   softplus.

  Returns:
      g_out: [H, B, T, K] -- chunk-local cumsum of activated gates.
  """
  H, B, T, K = g.shape
  assert_shape(g, (H, B, T, K), "g")
  assert A_log.shape == (H,), f"A_log shape {A_log.shape} != ({H},)"

  g_f32 = g.astype(jnp.float32)

  if dt_bias is not None:
    g_f32 = g_f32 + dt_bias.astype(jnp.float32).reshape(H, 1, 1, K)

  A = A_log.astype(jnp.float32)

  if lower_bound is None:
    g_act = -jnp.exp(A).reshape(H, 1, 1, 1) * jax.nn.softplus(g_f32)
  else:
    g_act = lower_bound * jax.nn.sigmoid(jnp.exp(A).reshape(H, 1, 1, 1) * g_f32)

  return chunk_local_cumsum_vector(
    g_act,
    chunk_size=chunk_size,
    scale=scale,
    head_first=True,
    output_dtype=output_dtype or jnp.float32,
  )


# =============================================================================
# Exact intra-chunk fallback for float32
# =============================================================================

def _solve_unit_lower_triangular(A, b):
  """Solve (I + A) x = b exactly, where A is strictly lower triangular.

  Uses block forward substitution with block size 16 for TPU MXU
  utilization. Within each 16-row diagonal block, rows are solved
  sequentially; between blocks, a single matmul propagates the
  solution downward.

  Args:
      A: (N, N) strictly lower triangular matrix (float32).
      b: (N, D) right-hand side matrix (float32).

  Returns:
      x: (N, D) exact solution matrix.
  """
  N, D = b.shape
  BS = 16
  num_blocks = N // BS
  A = A.astype(jnp.float32)
  b = b.astype(jnp.float32)

  blocks = jnp.split(b, num_blocks, axis=0)

  for i in range(num_blocks):
    start = i * BS
    end = (i + 1) * BS

    A_ii = A[start:end, start:end]
    x_block = blocks[i]

    rows = [x_block[r] for r in range(BS)]
    for j in range(BS):
      if j > 0:
        vec = A_ii[j, :j][None, :]
        mat = jnp.stack(rows[:j])
        correction = jax.lax.dot_general(
          vec,
          mat,
          (((1,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
        ).squeeze(axis=0)
        rows[j] = rows[j] - correction

    x_block = jnp.stack(rows)
    blocks[i] = x_block

    if i < num_blocks - 1:
      rest_start = (i + 1) * BS
      x_rest = jnp.concatenate(blocks[i + 1 :], axis=0)
      A_rest = A[rest_start:, start:end]

      update = jax.lax.dot_general(
        A_rest,
        x_block,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
      )
      x_rest = x_rest - update

      remaining = num_blocks - 1 - i
      new_blocks = jnp.split(x_rest, remaining, axis=0)
      for idx, nb in enumerate(new_blocks):
        blocks[i + 1 + idx] = nb

  return jnp.concatenate(blocks, axis=0)


def _solve_unit_lower_triangular_batched(A, b):
  """Solve (I + A) x = b exactly for batched inputs.

  Like ``_solve_unit_lower_triangular`` but with a leading batch dimension.
  Uses block forward substitution with block size 16 for TPU MXU
  utilization.

  Args:
      A: (MB, N, N) strictly lower triangular matrix (float32).
      b: (MB, N, D) right-hand side matrix (float32).

  Returns:
      x: (MB, N, D) exact solution matrix.
  """
  MB, N, D = b.shape
  BS = 16
  num_blocks = N // BS
  A = A.astype(jnp.float32)
  b = b.astype(jnp.float32)

  blocks = [b[:, i * BS : (i + 1) * BS, :] for i in range(num_blocks)]

  for i in range(num_blocks):
    start = i * BS
    end = (i + 1) * BS

    A_ii = A[:, start:end, start:end]  # [MB, BS, BS]
    x_block = blocks[i]  # [MB, BS, D]

    rows = [x_block[:, r, :] for r in range(BS)]  # list of [MB, D]
    for j in range(BS):
      if j > 0:
        vec = A_ii[:, j, :j]  # [MB, j]
        mat = jnp.stack(rows[:j], axis=1)  # [MB, j, D]
        # [MB, 1, j] @ [MB, j, D] → [MB, 1, D] → squeeze
        correction = jnp.matmul(vec[:, None, :], mat).squeeze(1)
        rows[j] = rows[j] - correction

    x_block = jnp.stack(rows, axis=1)  # [MB, BS, D]
    blocks[i] = x_block

    if i < num_blocks - 1:
      x_rest = jnp.concatenate(blocks[i + 1 :], axis=1)  # [MB, rest, D]
      A_rest = A[:, (i + 1) * BS :, start:end]  # [MB, rest, BS]

      update = jnp.matmul(
        A_rest,
        x_block,
        preferred_element_type=jnp.float32,
      )
      x_rest = x_rest - update

      remaining = num_blocks - 1 - i
      for idx in range(remaining):
        blocks[i + 1 + idx] = x_rest[:, idx * BS : (idx + 1) * BS, :]

  return jnp.concatenate(blocks, axis=1)


def _kda_fwd_intra_kernel(
  q_ref,
  k_ref,
  g_ref,
  beta_ref,
  v_ref,
  u_out_ref,
  w_out_ref,
  qg_out_ref,
  kg_out_ref,
  Aqk_out_ref,
  Akk_inv_out_ref,
  *,
  chunk_size: int,
  head_dim: int,
  value_dim: int,
  scale: float,
  disable_recompute: bool,
  safe_gate: bool,
):
  """Pallas kernel body for exact intra-chunk solve.

  Uses sub-block (BC=16) factored matmul for Aqk/Akk computation,
  matching the CPU reference's stabilization and einsum structure,
  then solves (I + L)x = b via block forward substitution.

  When safe_gate=False, uses sub-block first element (g[0]) as
  reference point (matches CPU default). When safe_gate=True, uses
  sub-block midpoint (g[BC//2]) to halve the max exponent, preventing
  exp2 overflow for large gate magnitudes (|gate| > 5.5/step).

  All refs have leading singleton dims from BlockSpec: [1, 1, 1, BT, D].
  """
  dtype = q_ref.dtype
  q = q_ref[0, 0, 0]  # (BT, K)
  k = k_ref[0, 0, 0]  # (BT, K)
  g = g_ref[0, 0, 0]  # (BT, K) -- cumsum gate in log2 space
  beta = beta_ref[0, 0, 0]  # (BT, 1)
  v = v_ref[0, 0, 0]  # (BT, V)

  BT = chunk_size
  BC = 16
  NC = BT // BC

  g_f32 = g.astype(jnp.float32)
  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  beta_f32 = beta.astype(jnp.float32)

  causal_bc = jnp.tril(jnp.ones((BC, BC), dtype=jnp.float32))
  strict_bc = jnp.tril(jnp.ones((BC, BC), dtype=jnp.float32), k=-1)
  zeros_bc = jnp.zeros((BC, BC), dtype=jnp.float32)

  # --- Sub-block factored Aqk/Akk via per-block max-subtraction ---
  # Mathematical identity (exact, no precision loss):
  #     Aqk[r,c] = sum_k q[r,k] * k[c,k] * exp2(g_i[r,k] - g_j[c,k])
  #             = exp2(g_max[r,c]) * sum_k q[r,k] * k[c,k] * exp2(g_diff - g_max)
  # where g_diff[r,c,k] = g_i[r,k] - g_j[c,k] and g_max[r,c] = max_k g_diff[r,c,k].
  # The inner exp2 is in (0, 1] -> always representable in fp32, no inf.
  # The outer exp2(g_max) only becomes inf when the true mathematical value
  # already exceeds fp32 range (faithful representation, not spurious NaN).
  #
  # On the diagonal sub-block (i_sc == j_sc), anti-causal entries (r < c)
  # would otherwise dominate g_max with arbitrarily large positives, then
  # be zeroed by causal_bc -- but the spurious large g_max would shrink
  # causal entries via exp2(g_diff - g_max). So we mask anti-causal g_diff
  # to a very small value before the max reduction on the diagonal block.
  NEG_INF_FLOAT = jnp.float32(-1e30)
  Aqk_rows = []
  L_rows = []
  for i_sc in range(NC):
    i_s = i_sc * BC
    q_i = q_f32[i_s : i_s + BC]  # (BC, K)
    k_i = k_f32[i_s : i_s + BC]
    g_i = g_f32[i_s : i_s + BC]
    beta_i = beta_f32[i_s : i_s + BC]  # (BC, 1)

    Aqk_blks = []
    L_blks = []
    for j_sc in range(NC):
      if j_sc > i_sc:
        Aqk_blks.append(zeros_bc)
        L_blks.append(zeros_bc)
      else:
        j_s = j_sc * BC
        k_j = k_f32[j_s : j_s + BC]
        g_j = g_f32[j_s : j_s + BC]

        # g_diff[r, c, k] = g_i[r, k] - g_j[c, k]; shape (BC, BC, K)
        g_diff = g_i[:, None, :] - g_j[None, :, :]

        # On diagonal block, mask anti-causal positions to a very small
        # value (1) so they don't pollute g_max for the causal positions,
        # and (2) so decay stays finite there even though they will be
        # zeroed out below by causal_bc.
        if i_sc == j_sc:
          g_diff = jnp.where(causal_bc[:, :, None] > 0, g_diff, NEG_INF_FLOAT)

        # (BC, BC, 1) -- per-(r, c) max over K. Always finite because
        # mask above clamps anti-causal entries (or none are masked
        # in the off-diagonal case).
        g_max = jnp.max(g_diff, axis=-1, keepdims=True)

        # Inner exponent in (-inf, 0] -> exp2 in (0, 1], always finite
        # (no inf, no NaN). For masked anti-causal positions, g_diff and
        # g_max are both NEG_INF_FLOAT so g_diff - g_max = 0 -> decay = 1
        # (a finite value that will be zeroed by causal_bc anyway).
        decay = jnp.exp2(g_diff - g_max)  # (BC, BC, K)
        exp_max = jnp.exp2(g_max[..., 0])  # (BC, BC)

        # Aqk_blk[r, c] = scale * exp_max[r, c]
        #               * sum_k q_i[r, k] * decay[r, c, k] * k_j[c, k]
        Aqk_blk = (
          scale * exp_max * jnp.sum(q_i[:, None, :] * decay * k_j[None, :, :], axis=-1)
        )

        # Akk_blk[r, c] = beta_i[r] * exp_max[r, c]
        #               * sum_k k_i[r, k] * decay[r, c, k] * k_j[c, k]
        Akk_blk = (
          beta_i * exp_max * jnp.sum(k_i[:, None, :] * decay * k_j[None, :, :], axis=-1)
        )

        if i_sc == j_sc:
          # Use `where` instead of `* mask` to avoid `inf * 0 = NaN` when
          # exp_max overflows on long sequences.
          Aqk_blk = jnp.where(causal_bc > 0, Aqk_blk, jnp.float32(0.0))
          Akk_blk = jnp.where(strict_bc > 0, Akk_blk, jnp.float32(0.0))

        Aqk_blks.append(Aqk_blk)
        L_blks.append(Akk_blk)

    Aqk_rows.append(jnp.concatenate(Aqk_blks, axis=1))
    L_rows.append(jnp.concatenate(L_blks, axis=1))

  Aqk = jnp.concatenate(Aqk_rows, axis=0).astype(dtype)  # (BT, BT)
  L = jnp.concatenate(L_rows, axis=0)  # (BT, BT)

  # --- Exact solve: (I + L) x = [v*beta, k*exp2(g)*beta, I] ---
  v_beta = v.astype(jnp.float32) * beta_f32  # (BT, V)
  k_eg_beta = k_f32 * jnp.exp2(g_f32) * beta_f32  # (BT, K)
  identity = jnp.eye(BT, dtype=jnp.float32)  # (BT, BT)

  combined_b = jnp.concatenate([v_beta, k_eg_beta, identity], axis=-1)
  combined_x = _solve_unit_lower_triangular(L, combined_b)

  u = combined_x[:, :value_dim]  # (BT, V)
  w = combined_x[:, value_dim : value_dim + head_dim]  # (BT, K)
  A_inv = combined_x[:, value_dim + head_dim :]  # (BT, BT)

  # --- kg = k * exp2(g_last - g) ---
  # g_last <= g[i] for all i (monotonically non-increasing), so exponent <= 0.
  g_last = g_f32[BT - 1 : BT, :]  # (1, K)
  kg = k_f32 * jnp.exp2(g_last - g_f32)  # (BT, K)

  # --- qg = q * exp2(g) (optional) ---
  if disable_recompute:
    qg = q_f32 * jnp.exp2(g_f32)  # (BT, K)
  else:
    qg = jnp.zeros_like(q_f32)

  # --- Store outputs ---
  u_out_ref[0, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[0, 0, 0] = w.astype(w_out_ref.dtype)
  qg_out_ref[0, 0, 0] = qg.astype(qg_out_ref.dtype)
  kg_out_ref[0, 0, 0] = kg.astype(kg_out_ref.dtype)
  Aqk_out_ref[0, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
  Akk_inv_out_ref[0, 0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)


def kda_fwd_intra(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  gk: jax.Array,
  beta: jax.Array,
  scale: float,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
  chunk_indices: jax.Array | None = None,
  safe_gate: bool = True,
  disable_recompute: bool = False,
):
  if cu_seqlens is None:
    return pallas_kda_fwd_intra(
      q=q,
      k=k,
      v=v,
      gk=gk,
      beta=beta,
      scale=scale,
      cu_seqlens=cu_seqlens,
      chunk_size=chunk_size,
      chunk_indices=chunk_indices,
      safe_gate=safe_gate,
      disable_recompute=disable_recompute,
    )
  return kda_fwd_intra_varlen(
    q=q,
    k=k,
    v=v,
    gk=gk,
    beta=beta,
    scale=scale,
    cu_seqlens=cu_seqlens,
    chunk_size=chunk_size,
    chunk_indices=chunk_indices,
    safe_gate=safe_gate,
    disable_recompute=disable_recompute,
  )


def _kda_fwd_intra_varlen_kernel(
  q_ref,
  k_ref,
  g_ref,
  beta_ref,
  v_ref,
  u_out_ref,
  w_out_ref,
  qg_out_ref,
  kg_out_ref,
  Aqk_out_ref,
  Akk_inv_out_ref,
  *,
  chunk_size,
  head_dim,
  value_dim,
  scale,
  disable_recompute,
  safe_gate,
):
  dtype = q_ref.dtype
  q = q_ref[0, 0, 0]
  k = k_ref[0, 0, 0]
  g = g_ref[0, 0, 0]
  beta = beta_ref[0, 0, 0]
  v = v_ref[0, 0, 0]

  BT = chunk_size
  BC = 16
  NC = BT // BC

  g_f32 = g.astype(jnp.float32)
  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  beta_f32 = beta.astype(jnp.float32)

  # Build Aqk and L directly using exp2(g[i] - g[j]).
  # For causal (i >= j): g_cumsum[i] <= g_cumsum[j], so g[i]-g[j] <= 0,
  # giving exp2 in (0, 1].  This avoids the split-normalization overflow
  # that occurs with exp2(g-gn) when per-step gate changes exceed ~127.
  causal_bt = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
  strict_bt = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32), k=-1)

  # g_diff[i, j, k] = g[i, k] - g[j, k];  shape [BT, BT, K]
  g_diff = g_f32[:, None, :] - g_f32[None, :, :]
  # Mask anti-causal entries to -126 before exp2 to prevent overflow;
  # they will be zeroed by causal_bt / strict_bt anyway.
  g_diff = jnp.where(causal_bt[:, :, None] > 0, g_diff, -126.0)
  decay = exp2(jnp.maximum(g_diff, -126.0))  # [BT, BT, K]

  # Aqk[i, j] = scale * sum_k q[i,k] * k[j,k] * decay[i,j,k]
  Aqk = scale * jnp.sum(q_f32[:, None, :] * decay * k_f32[None, :, :], axis=-1)
  # Use `where` instead of `* mask` to avoid `inf * 0 = NaN`.
  Aqk = jnp.where(causal_bt > 0, Aqk, jnp.float32(0.0)).astype(dtype)

  # L[i, j] = beta[i] * sum_k k[i,k] * k[j,k] * decay[i,j,k]   (i > j)
  L = jnp.sum(k_f32[:, None, :] * decay * k_f32[None, :, :], axis=-1) * beta_f32
  L = jnp.where(strict_bt > 0, L, jnp.float32(0.0))

  v_beta = v.astype(jnp.float32) * beta_f32
  k_eg_beta = k_f32 * exp2(g_f32) * beta_f32
  identity = jnp.eye(BT, dtype=jnp.float32)

  combined_b = jnp.concatenate([v_beta, k_eg_beta, identity], axis=-1)
  combined_x = _solve_unit_lower_triangular(L, combined_b)

  u = combined_x[:, :value_dim]
  w = combined_x[:, value_dim : value_dim + head_dim]
  A_inv = combined_x[:, value_dim + head_dim :]

  g_last = g_f32[BT - 1 : BT, :]
  kg = k_f32 * exp2(g_last - g_f32)

  qg = q_f32 * exp2(g_f32) if disable_recompute else jnp.zeros_like(q_f32)

  u_out_ref[0, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[0, 0, 0] = w.astype(w_out_ref.dtype)
  qg_out_ref[0, 0, 0] = qg.astype(qg_out_ref.dtype)
  kg_out_ref[0, 0, 0] = kg.astype(kg_out_ref.dtype)
  Aqk_out_ref[0, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
  Akk_inv_out_ref[0, 0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "safe_gate",
    "disable_recompute",
  ],
)
def kda_fwd_intra_varlen(
  q,
  k,
  v,
  gk,
  beta,
  scale,
  cu_seqlens,
  chunk_size=64,
  chunk_indices=None,
  safe_gate=True,
  disable_recompute=False,
):
  assert cu_seqlens is not None, "cu_seqlens must be provided for varlen"
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert BT >= 16 and BT % 16 == 0

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(gk, (H, B, T, K), "gk")
  assert_shape(beta, (H, B, T), "beta")

  NC = T // BT
  q_r = q.reshape(H, B, NC, BT, K)
  k_r = k.reshape(H, B, NC, BT, K)
  g_r = gk.reshape(H, B, NC, BT, K)
  beta_r = beta.reshape(H, B, NC, BT, 1)
  v_r = v.reshape(H, B, NC, BT, V)

  NC_max = NC  # chunk_indices may be 3D [B, NT, 2] for B>1; use NC directly
  grid = (H, B, NC_max)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, n: (i, j, n, 0, 0), block_shape=(1, 1, 1, BT, last_dim)
    )

  (u_r, w_r, qg_r, kg_r, Aqk_r, Akk_inv_r) = pl.pallas_call(
    functools.partial(
      _kda_fwd_intra_varlen_kernel,
      chunk_size=BT,
      head_dim=K,
      value_dim=V,
      scale=scale,
      disable_recompute=disable_recompute,
      safe_gate=safe_gate,
    ),
    interpret=get_interpret(),
    out_shape=[
      jax.ShapeDtypeStruct((H, B, NC_max, BT, V), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC_max, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC_max, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC_max, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC_max, BT, BT), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC_max, BT, BT), q.dtype),
    ],
    in_specs=[
      _make_spec(K),
      _make_spec(K),
      _make_spec(K),
      _make_spec(1),
      _make_spec(V),
    ],
    out_specs=[
      _make_spec(V),
      _make_spec(K),
      _make_spec(K),
      _make_spec(K),
      _make_spec(BT),
      _make_spec(BT),
    ],
    grid=grid,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel")
    ),
  )(q_r, k_r, g_r, beta_r, v_r)


  qg_out = qg_r if disable_recompute else None

  # Reshape 5D [H,B,NC,BT,X] back to 4D [H,B,T,X] for downstream consumers
  def _r4(x):
    if x is None:
      return None
    return x.reshape(H, B, -1, x.shape[-1])

  return _r4(w_r), _r4(u_r), _r4(qg_out), _r4(kg_r), _r4(Aqk_r), _r4(Akk_inv_r)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "safe_gate",
    "disable_recompute",
  ],
)
def pallas_kda_fwd_intra(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  gk: jax.Array,
  beta: jax.Array,
  scale: float,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
  chunk_indices: jax.Array | None = None,
  safe_gate: bool = True,
  disable_recompute: bool = False,
) -> tuple[
  jax.Array,
  jax.Array,
  jax.Array | None,
  jax.Array,
  jax.Array,
  jax.Array,
]:
  """KDA intra-chunk forward using exact block forward substitution.

  Within each chunk, builds the key-key interaction matrix Akk (strictly
  lower-triangular), solves (I + L)x = b exactly via block forward
  substitution (block size 16), and uses the result to compute
  delta-corrected values (u), correction weights (w), and the
  attention / inverse matrices (Aqk, Akk).

  The gates gk must already be in log2 space (chunk-local cumsum scaled
  by 1/ln2), so all exponentials use exp2.

  Args:
      q:     [H, B, T, K] -- query vectors.
      k:     [H, B, T, K] -- key vectors.
      v:     [H, B, T, V] -- value vectors. V may differ from K.
      gk:    [H, B, T, K] -- chunk-local cumsum of gates in log2 space.
      beta:  [H, B, T]    -- per-token scalar mixing coefficient for the
                              delta-rule update.
      scale: float         -- attention scale factor (typically 1/sqrt(K)).
      chunk_size: int      -- chunk size BT (default 64). T must be
                              divisible by chunk_size.
      safe_gate: bool      -- if True, use sub-block midpoint (g[BC//2])
                                 as reference to halve max exponent and
                                 prevent exp2 overflow for large gates.
                                 If False, use first element (g[0]) to
                                 match CPU reference default.
      disable_recompute: bool -- if True, also outputs qg = q * exp2(gk);
                                 if False, qg is None.

  Returns:
      w:    [H, B, T, K]       -- correction weights = A_inv @ (k*beta*exp2(g)).
      u:    [H, B, T, V]       -- delta-corrected values = A_inv @ (v*beta).
      qg:   [H, B, T, K] or None -- q * exp2(gk), only if disable_recompute.
      kg:   [H, B, T, K]       -- k * exp2(g_last - gk); g_last = last
                                  token's gate in chunk.
      Aqk:  [H, B, NC, BT, BT] -- query-key attention matrix per chunk
                                  (5D), with causal mask (i >= j) and
                                  scale applied.
      Akk:  [H, B, NC, BT, BT] -- exact (I + L)^{-1} matrix per chunk (5D).
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  NC = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(gk, (H, B, T, K), "gk")
  assert_shape(beta, (H, B, T), "beta")

  # --- Reshape to [B, H, NC, BT, D] for per-chunk Pallas grid ---
  q_r = q.reshape(H, B, NC, BT, K)
  k_r = k.reshape(H, B, NC, BT, K)
  g_r = gk.reshape(H, B, NC, BT, K)
  beta_r = beta.reshape(H, B, NC, BT, 1)
  v_r = v.reshape(H, B, NC, BT, V)

  grid = (H, B, NC)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, j, l, 0, 0),
      block_shape=(1, 1, 1, BT, last_dim),
    )

  (u_r, w_r, qg_r, kg_r, Aqk_r, Akk_inv_r) = pl.pallas_call(
    functools.partial(
      _kda_fwd_intra_kernel,
      chunk_size=BT,
      head_dim=K,
      value_dim=V,
      scale=scale,
      disable_recompute=disable_recompute,
      safe_gate=safe_gate,
    ),
    interpret=get_interpret(),
    out_shape=[
      jax.ShapeDtypeStruct((H, B, NC, BT, V), k.dtype),  # u
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),  # w
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),  # qg
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),  # kg
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),  # Aqk
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),  # Akk_inv
    ],
    in_specs=[
      _make_spec(K),  # q
      _make_spec(K),  # k
      _make_spec(K),  # g
      _make_spec(1),  # beta
      _make_spec(V),  # v
    ],
    out_specs=[
      _make_spec(V),  # u
      _make_spec(K),  # w
      _make_spec(K),  # qg
      _make_spec(K),  # kg
      _make_spec(BT),  # Aqk
      _make_spec(BT),  # Akk_inv
    ],
    grid=grid,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel"),
    ),
  )(q_r, k_r, g_r, beta_r, v_r)

  # --- Reshape back to [H, B, T, D] ---
  w_out = w_r.reshape(H, B, T, K)
  u_out = u_r.reshape(H, B, T, V)
  kg_out = kg_r.reshape(H, B, T, K)

  qg_out: jax.Array | None
  if disable_recompute:
    qg_out = qg_r.reshape(H, B, T, K)
  else:
    qg_out = None

  return w_out, u_out, qg_out, kg_out, Aqk_r, Akk_inv_r


# =============================================================================
# Fused gate cumsum and intra-chunk solve
# =============================================================================

def _fused_gate_intra_kernel(
  q_ref,
  k_ref,
  g_ref,
  beta_ref,
  v_ref,
  A_log_ref,
  dt_bias_ref,
  u_out_ref,
  w_out_ref,
  qg_out_ref,
  kg_out_ref,
  Aqk_out_ref,
  Akk_inv_out_ref,
  g_cumsum_out_ref,
  *,
  chunk_size: int,
  head_dim: int,
  value_dim: int,
  scale: float,
  cumsum_scale: float,
  fuse_cumsum: bool,
  disable_recompute: bool,
  safe_gate: bool,
  use_gate_in_kernel: bool,
  lower_bound: float | None,
  mini_batch: int = 1,
):
  """Fused Pallas kernel: gate activation + cumsum + BC=16 Aqk/L + Neumann inversion.

  When fuse_cumsum=True, g_ref contains raw gate values (not cumsum'd).
  The kernel applies gate activation (if use_gate_in_kernel) and chunk-local
  prefix sum via tril matmul before proceeding to the Aqk/L construction.

  For bfloat16 inputs, uses Neumann series inversion.
  For float32 inputs, falls back to exact forward substitution.

  All refs have leading dims from BlockSpec: [1, MB, 1, BT, D].
  A_log_ref: [1, MB, 1, 1, 1] — per-head scalar.
  dt_bias_ref: [1, MB, 1, 1, K] — per-head bias vector.

  MB heads are processed simultaneously via batch-vectorized ops.

  Args:
      mini_batch: int — number of heads processed per grid point (MB).
  """
  dtype = q_ref.dtype
  BT = chunk_size
  BC = 16
  NC = BT // BC
  K = head_dim
  V = value_dim
  MB = mini_batch

  # Load all MB heads at once
  q = q_ref[:, 0, 0]        # [MB, BT, K]
  k = k_ref[:, 0, 0]        # [MB, BT, K]
  g = g_ref[:, 0, 0]        # [MB, BT, K]
  beta = beta_ref[:, 0, 0]  # [MB, BT, 1]
  v = v_ref[:, 0, 0]        # [MB, BT, V]

  # --- Gate activation + cumsum ---
  g_f32 = g.astype(jnp.float32)

  if use_gate_in_kernel:
    dt_b = dt_bias_ref[:, 0, 0, 0]        # [MB, K]
    g_f32 = g_f32 + dt_b[:, None, :]       # [MB, BT, K]
    A_val = A_log_ref[:, 0, 0, 0, 0]       # [MB]
    if lower_bound is None:
      g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
    else:
      g_f32 = lower_bound * jax.nn.sigmoid(
        jnp.exp(A_val)[:, None, None] * g_f32
      )

  if fuse_cumsum:
    tril = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    # tril[BT,BT] @ g_f32[MB,BT,K] → [MB,BT,K]: contract on BT, batch on MB
    g_cumsum = jax.lax.dot_general(
      tril, g_f32, (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2) * cumsum_scale  # [BT,MB,K] → [MB,BT,K]
  else:
    g_cumsum = g_f32

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  beta_f32 = beta.astype(jnp.float32)

  # --- BC=16 sub-block factored Aqk/L (fla safe-gate style, j-batched) ---
  # Per i_sc, batch the j_sc loop into a single (MB, 2*BC, K) x (MB, BT, K) matmul:
  #   q_eg[m,r,k]   = q_i[m,r,k] * exp2(g_i[m,r,k] - gn_ref[m,k])
  #   k_eg[m,r,k]   = k_i[m,r,k] * exp2(g_i[m,r,k] - gn_ref[m,k])    (Akk left)
  #   k_eng[m,t,k]  = k[m,t,k]   * exp2(gn_ref[m,k] - g_cumsum[m,t,k]) (full BT)
  # so Aqk[m,r,t]  = sum_k q_i[m,r,k] * k[m,t,k] * exp2(g_i[m,r,k] - g_cumsum[m,t,k]).
  # Stack q_eg and k_eg along row dim and do one matmul -> split Aqk/Akk.
  #
  # Anti-causal rows (t >= (i_sc+1)*BC) have positive `gn_ref - g_cumsum[t]`
  # for negative-gate cumsum, which can overflow exp2. Mask those diffs to 0
  # BEFORE exp2, then zero the resulting rows so they contribute 0 to matmul.
  #
  # Use broadcasted_iota for 2D indices to avoid Mosaic-unsupported i1
  # shape casts (e.g. (BT,) -> (BT, 1) on bool tensors).
  ref_idx = BC // 2 if safe_gate else 0
  row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (BT, K), dimension=0)
  row_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), dimension=0)
  col_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (BC, BT), dimension=1)

  Aqk_rows = []
  L_rows = []
  for i_sc in range(NC):
    i_s = i_sc * BC
    q_i = q_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    k_i = k_f32[:, i_s : i_s + BC]       # [MB, BC, K]
    g_i = g_cumsum[:, i_s : i_s + BC]    # [MB, BC, K]
    beta_i = beta_f32[:, i_s : i_s + BC]  # [MB, BC, 1]

    gn_ref = g_i[:, ref_idx : ref_idx + 1, :]   # [MB, 1, K] — query sub-block ref
    diff_i = g_i - gn_ref                        # [MB, BC, K], <= 0
    exp_diff_i = jnp.exp2(diff_i)
    q_eg = q_i * exp_diff_i     # [MB, BC, K]
    k_eg = k_i * exp_diff_i     # [MB, BC, K]

    # j-side: full BT rows. Mask anti-causal rows BEFORE exp2 (avoids
    # overflow when gn_ref - g_cumsum[t] is large positive). The 2D mask
    # is built from broadcasted_iota to bypass the i1 reshape limitation.
    valid_j = (row_iota_bt_k < (i_s + BC)).astype(jnp.float32)   # [BT, K]
    diff_j_safe = (gn_ref - g_cumsum) * valid_j[None]              # [MB, BT, K]
    k_eng_full = k_f32 * jnp.exp2(diff_j_safe) * valid_j[None]    # [MB, BT, K]

    # Stack q_eg / k_eg and do one matmul: [MB, 2*BC, K] x [MB, K, BT] -> [MB, 2*BC, BT]
    qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)  # [MB, 2*BC, K]
    qk_dot = jax.lax.dot_general(
      qk_eg, k_eng_full, (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
    )  # [MB, 2*BC, BT]
    Aqk_row = qk_dot[:, :BC] * scale      # [MB, BC, BT]
    Akk_row = qk_dot[:, BC:] * beta_i     # [MB, BC, BT]

    # Apply diagonal-block masks (causal for Aqk, strict-lower for Akk).
    # All masks built from 2D iota; no i1 shape casts needed.
    in_diag = (col_iota_bc_bt >= i_s) & (col_iota_bc_bt < i_s + BC)  # [BC, BT]
    col_local = col_iota_bc_bt - i_s                                   # [BC, BT]
    causal_diag = row_iota_bc_bt >= col_local                          # [BC, BT]
    strict_diag = row_iota_bc_bt > col_local                           # [BC, BT]
    aqk_keep = (~in_diag) | causal_diag
    akk_keep = (~in_diag) | strict_diag
    Aqk_row = jnp.where(aqk_keep[None], Aqk_row, jnp.float32(0.0))
    Akk_row = jnp.where(akk_keep[None], Akk_row, jnp.float32(0.0))

    Aqk_rows.append(Aqk_row)
    L_rows.append(Akk_row)

  Aqk = jnp.concatenate(Aqk_rows, axis=1).astype(dtype)  # [MB, BT, BT]
  L = jnp.concatenate(L_rows, axis=1)                     # [MB, BT, BT]

  # --- Solve (I + L) x = rhs ---
  v_beta = v.astype(jnp.float32) * beta_f32          # [MB, BT, V]
  k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32  # [MB, BT, K]
  I_bt = jnp.eye(BT, dtype=jnp.float32)               # [BT, BT]

  # Batched dot helper: [MB, M, K] @ [MB, K, N] → [MB, M, N]
  _dot_batch = lambda a, b: jax.lax.dot_general(
    a, b, (((2,), (1,)), ((0,), (0,))),
    preferred_element_type=jnp.float32,
  )

  use_neumann = dtype != jnp.float32

  if use_neumann:
    # --- Neumann series inversion ---
    BC_inv = 8
    NC_inv = BT // BC_inv
    inv_dtype = jnp.float32

    L_inv = L.astype(inv_dtype)  # [MB, BT, BT]

    _idx = jnp.arange(BT, dtype=jnp.int32)
    _block_id = _idx // BC_inv
    _same_block = (_block_id[:, None] == _block_id[None, :]).astype(inv_dtype)
    L_diag = L_inv * _same_block[None]  # [MB, BT, BT]
    F = L_inv - L_diag                   # [MB, BT, BT]

    neg_Ld = -L_diag
    S = I_bt[None] + neg_Ld              # [MB, BT, BT]
    Mk = neg_Ld
    num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
    for _ in range(num_diag_steps):
      Mk = _dot_batch(Mk, Mk)
      S = _dot_batch(S, I_bt[None] + Mk)
    P = S

    rhs = jnp.concatenate([
      v_beta.astype(inv_dtype),
      k_eg_beta.astype(inv_dtype),
    ], axis=-1)  # [MB, BT, V+K]

    if NC_inv == 1:
      # P already equals (I + L)^{-1} (no off-diagonal blocks).
      result = _dot_batch(P, rhs)
      A_inv = P
    else:
      # Fuse `P @ [F | rhs]` to share a matmul; split out G and P_rhs.
      F_and_rhs = jnp.concatenate([F, rhs], axis=-1)  # [MB, BT, BT+V+K]
      P_merged = _dot_batch(P, F_and_rhs)              # [MB, BT, BT+V+K]
      G = P_merged[:, :, :BT]                          # [MB, BT, BT]
      P_rhs = P_merged[:, :, BT:]                      # [MB, BT, V+K]

      # Compute inv_I_G = (I + G)^{-1} = sum_{k=0}^{NC_inv-1} (-G)^k via
      # Horner doubling. Build the matrix first (small (MB,BT,BT) matmuls),
      # then apply once to P_rhs and once to P (small + medium matmuls).
      # For NC_inv=8: 4 × (MB,BT,BT,BT) mm + 1 × (MB,BT,BT,V+K) mm + 1 × (MB,BT,BT,BT) mm
      # vs sequential: 6 × (MB,BT,BT,BT) mm + 1 × (MB,BT,BT,V+K+BT) mm.
      H_mat = -G
      inv_I_G = I_bt[None] + H_mat                     # (I + H)
      Hk = H_mat
      log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
      for step in range(log2_NC_inv - 1):
        Hk = _dot_batch(Hk, Hk)                        # H^(2^(step+1))
        inv_I_G = inv_I_G + _dot_batch(inv_I_G, Hk)    # inv_I_G @ (I + Hk)

      result = _dot_batch(inv_I_G, P_rhs)              # [MB, BT, V+K]
      A_inv = _dot_batch(inv_I_G, P)                   # [MB, BT, BT] = (I+L)^{-1}
  else:
    # --- Exact forward substitution (fp32) ---
    I_bt_batch = jnp.broadcast_to(I_bt, (MB, BT, BT))
    combined_b = jnp.concatenate(
      [v_beta, k_eg_beta, I_bt_batch], axis=-1
    )  # [MB, BT, V+K+BT]
    full_result = _solve_unit_lower_triangular_batched(L, combined_b)
    result = full_result[:, :, : V + K]
    A_inv = full_result[:, :, V + K :]

  u = result[:, :, :V]         # [MB, BT, V]
  w = result[:, :, V : V + K]  # [MB, BT, K]

  # --- kg, qg ---
  g_last = g_cumsum[:, BT - 1 : BT, :]  # [MB, 1, K]
  kg = k_f32 * exp2(g_last - g_cumsum)
  qg = q_f32 * exp2(g_cumsum) if disable_recompute else jnp.zeros_like(q_f32)

  # --- Store all MB heads ---
  u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
  qg_out_ref[:, 0, 0] = qg.astype(qg_out_ref.dtype)
  kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
  Aqk_out_ref[:, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
  Akk_inv_out_ref[:, 0, 0] = A_inv.astype(Akk_inv_out_ref.dtype)
  g_cumsum_out_ref[:, 0, 0] = g_cumsum


def _compute_intra_fused_mini_batch(H, BT, K, V, dtype=None):
  """Auto-compute mini-batch size for intra fused kernel.

  Each head needs VMEM for intermediates: BT*K + BT*V + 2*BT*BT (Aqk, L).
  Uses hardware VMEM budget via ``estimate_mini_batch``.
  """
  elem_size = dtype.itemsize if dtype is not None else 4
  per_head = (BT * K + BT * V + 2 * BT * BT) * elem_size
  return estimate_mini_batch(per_head, H, max_mb=16)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "safe_gate",
    "disable_recompute",
    "cumsum_scale",
    "use_gate_in_kernel",
    "lower_bound",
    "mini_batch",
  ],
)
def pallas_kda_fwd_intra_fused(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  scale: float,
  chunk_size: int = 64,
  safe_gate: bool = True,
  disable_recompute: bool = False,
  cumsum_scale: float = _RCP_LN2,
  A_log: jax.Array | None = None,
  dt_bias: jax.Array | None = None,
  use_gate_in_kernel: bool = False,
  lower_bound: float | None = None,
  mini_batch: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array,
           jax.Array, jax.Array, jax.Array]:
  """Fused gate cumsum + intra-chunk solve with mini-batch.

  Processes MB heads per grid point to amortize DMA overhead.
  Variable-length inputs must already be chunk-aligned by the caller.

  Args:
      q:     [H, B, T, K] -- query vectors (head-first layout).
      k:     [H, B, T, K] -- key vectors.
      v:     [H, B, T, V] -- value vectors.
      g:     [H, B, T, K] -- raw gate input.
      beta:  [H, B, T]    -- per-token scalar mixing coefficient.
      scale: float         -- attention scale factor.
      chunk_size: int      -- chunk size BT.
      mini_batch: int or None -- heads per grid point (auto if None).

  Returns:
      7-tuple: (w, u, qg, kg, Aqk, Akk, g_cumsum) in head-first
      ``[H, B, T, ...]`` layout.  qg is None when ``disable_recompute=False``.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  NC = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")

  if use_gate_in_kernel:
    assert A_log is not None, "A_log required when use_gate_in_kernel=True"

  if mini_batch is None:
    MB = _compute_intra_fused_mini_batch(H, BT, K, V, dtype=q.dtype)
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # [H, B, T, K] -> [H, B, NC, BT, K]
  q_r = q.reshape(H, B, NC, BT, K)
  k_r = k.reshape(H, B, NC, BT, K)
  g_r = g.reshape(H, B, NC, BT, K)
  beta_r = beta.reshape(H, B, NC, BT, 1)
  v_r = v.reshape(H, B, NC, BT, V)

  if use_gate_in_kernel:
    A_log_r = A_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
    if dt_bias is not None:
      dt_bias_r = dt_bias.astype(jnp.float32).reshape(H, 1, 1, 1, K)
    else:
      dt_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
  else:
    A_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
    dt_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)

  grid = (H // MB, B, NC)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, j, l, 0, 0),
      block_shape=(MB, 1, 1, BT, last_dim),
    )

  def _make_per_head_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, l: (i, 0, 0, 0, 0),
      block_shape=(MB, 1, 1, 1, last_dim),
    )

  (u_r, w_r, qg_r, kg_r, Aqk_r, Akk_inv_r, g_cumsum_r) = pl.pallas_call(
    functools.partial(
      _fused_gate_intra_kernel,
      chunk_size=BT,
      head_dim=K,
      value_dim=V,
      scale=scale,
      cumsum_scale=cumsum_scale,
      fuse_cumsum=True,
      disable_recompute=disable_recompute,
      safe_gate=safe_gate,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      mini_batch=MB,
    ),
    interpret=get_interpret(),
    out_shape=[
      jax.ShapeDtypeStruct((H, B, NC, BT, V), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), jnp.float32),
    ],
    in_specs=[
      _make_spec(K),
      _make_spec(K),
      _make_spec(K),
      _make_spec(1),
      _make_spec(V),
      _make_per_head_spec(1),
      _make_per_head_spec(K),
    ],
    out_specs=[
      _make_spec(V),
      _make_spec(K),
      _make_spec(K),
      _make_spec(K),
      _make_spec(BT),
      _make_spec(BT),
      _make_spec(K),
    ],
    grid=grid,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel"),
    ),
  )(q_r, k_r, g_r, beta_r, v_r, A_log_r, dt_bias_r)

  # --- Reshape back to [H, B, T, D] (head-first) ---
  w_out = w_r.reshape(H, B, T, K)
  u_out = u_r.reshape(H, B, T, V)
  kg_out = kg_r.reshape(H, B, T, K)
  qg_out = (
    qg_r.reshape(H, B, T, K)
    if disable_recompute else None
  )
  Aqk_flat = Aqk_r.reshape(H, B, NC * BT, BT)
  Akk_flat = Akk_inv_r.reshape(H, B, NC * BT, BT)
  g_cumsum_out = g_cumsum_r.reshape(H, B, T, K)

  return w_out, u_out, qg_out, kg_out, Aqk_flat, Akk_flat, g_cumsum_out


def kda_fwd_intra_fused(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  scale: float,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
  chunk_indices: jax.Array | None = None,
  safe_gate: bool = True,
  disable_recompute: bool = False,
  cumsum_scale: float = _RCP_LN2,
  A_log: jax.Array | None = None,
  dt_bias: jax.Array | None = None,
  use_gate_in_kernel: bool = False,
  lower_bound: float | None = None,
):
  """Fused gate cumsum + intra-chunk solve entry.

  **fp32 fallback**: fp32 uses separate S1+S2 (no BC=16 fusion) to avoid
  numerical issues in near-zero gate scenarios where BC=16 reference point
  normalization amplifies rounding errors.

  Args:
      g: [H, B, T, K] -- raw gate input (NOT cumsum'd).
      Other args: same as ``kda_fwd_intra`` + gate activation params.

  Returns:
      7-tuple: (w, u, qg, kg, Aqk, Akk, g_cumsum).
  """
  assert chunk_size == 64, f"Expected chunk_size=64, got {chunk_size}"
  # fp32 fallback: use separate S1+S2 to avoid BC=16 numerical issues
  if q.dtype == jnp.float32:
    if use_gate_in_kernel:
      assert A_log is not None, "A_log must not be None when use_gate_in_kernel=True"
      g_cumsum = kda_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        chunk_size=chunk_size,
        scale=_RCP_LN2,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
      )
    else:
      g_cumsum = kda_gate_cumsum(
        g=g,
        scale=_RCP_LN2,
        chunk_size=chunk_size,
        head_first=True,
      )

    # S2: intra-chunk solve (uses original kernel without BC=16)
    w, u, qg, kg, Aqk, Akk = kda_fwd_intra(
      q=q, k=k, v=v, gk=g_cumsum, beta=beta,
      scale=scale, cu_seqlens=cu_seqlens,
      chunk_size=chunk_size, chunk_indices=chunk_indices,
      safe_gate=safe_gate, disable_recompute=disable_recompute,
    )

    # Reshape Aqk/Akk from 5D [H,B,NC,BT,BT] to 4D [H,B,T,BT]
    # (fp32 fallback returns raw Pallas output; bf16 paths do this in their wrappers)
    Aqk = Aqk.reshape(q.shape[0], q.shape[1], -1, Aqk.shape[-1])
    Akk = Akk.reshape(q.shape[0], q.shape[1], -1, Akk.shape[-1])
    return w, u, qg, kg, Aqk, Akk, g_cumsum

  # The caller has already BT-aligned varlen inputs, so the same contiguous
  # fused kernel handles both fixed-length and variable-length batches.
  return pallas_kda_fwd_intra_fused(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=scale, chunk_size=chunk_size,
    safe_gate=safe_gate, disable_recompute=disable_recompute,
    cumsum_scale=cumsum_scale,
    A_log=A_log, dt_bias=dt_bias,
    use_gate_in_kernel=use_gate_in_kernel,
    lower_bound=lower_bound,
  )


def kda_gate_cumsum(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float = _RCP_LN2,
  head_first: bool = True,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Chunk-local cumulative sum of pre-activated KDA gates.

  For the case where gates are already activated (use_gate_in_kernel=False),
  converts g from natural log space to log2 space (g / ln2) via
  chunk-local cumsum with scale=RCP_LN2.

  Args:
      g:          [H, B, T, K] -- per-element gate in natural log space
                  (head-first layout).
      chunk_size: int -- chunk size BT. T must be divisible by chunk_size.
      scale:      float -- scale factor applied after cumsum (default 1/ln2).

  Returns:
      g_out: [H, B, T, K] (float32) -- chunk-local cumsum in log2 space.
  """
  H, B, T, K = g.shape
  assert_shape(g, (H, B, T, K), "g")
  assert T % chunk_size == 0, f"T={T} must be divisible by chunk_size={chunk_size}"

  return chunk_local_cumsum_vector(
    g,
    chunk_size=chunk_size,
    reverse=reverse,
    scale=scale,
    head_first=head_first,
    output_dtype=output_dtype,
  )


# =============================================================================
# Fused state propagation and output
# =============================================================================

def _chunk_kda_fwd_h_o_varlen_kernel(
  seqlens_ref,       # scalar prefetch: cu_seqlens [N+1]
  chunk_to_seq_ref,  # scalar prefetch: chunk -> seq mapping [NT]
  # Stage 3 inputs
  w_ref,      # [MB, 1, BT, K_PADSIZE]
  u_ref,      # [MB, 1, BT, V_ALIGNED]
  kg_ref,     # [MB, 1, BT, K_PADSIZE]
  gk_ref,     # [MB, 1, BT, K_PADSIZE]  -- g_cumsum
  # Stage 4 inputs
  q_ref,      # [MB, 1, BT, K_PADSIZE]
  A_ref,      # [MB, 1, BT, BT]
  # Optional initial state
  h0_ref,     # [1, MB, K_PADSIZE, V_ALIGNED] or None
  # Outputs
  o_ref,      # [MB, 1, BT, V_ALIGNED]
  ht_ref,     # [1, MB, K_PADSIZE, V_ALIGNED] or None
  h_out_ref,    # [MB, 1, 1, K_PADSIZE, V_ALIGNED] or None  -- per-chunk pre-update h
  v_new_out_ref,  # [MB, 1, BT, V_ALIGNED] or None         -- per-token v_new
  # Scratch
  scratch_ref,  # [MB, K_PADSIZE, V_ALIGNED]
  *,
  BT,
  scale,
  USE_INITIAL_STATE,
  STORE_FINAL_STATE,
  STORE_H,
  STORE_V_NEW,
  MB,
  OUTPUT_PRECISION,
):
  """Fused Stage 3+4 Pallas kernel body for varlen with H-dim mini-batch.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, i_c):
    - MB heads [h_group*MB .. h_group*MB+MB) are processed per grid point
      via batched matmuls over the MB dimension.
    - i_b is the batch index, i_c is the chunk index within that batch.
    - seq_idx = chunk_to_seq[i_b, i_c] identifies which sequence this chunk
      belongs to within batch i_b.
    - At t0 == bos: init scratch (h0 or zeros) for this sequence
    - At t0 + BT >= eos: store final state for this sequence
  """
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  seq_idx = chunk_to_seq_ref[i_b, i_c]

  bos = seqlens_ref[i_b, seq_idx]
  eos = seqlens_ref[i_b, seq_idx + 1]
  t0 = i_c * BT

  K = w_ref.shape[3]
  V = u_ref.shape[3]

  # === Init state (first chunk of THIS sequence) — uniform across all MB heads ===
  @pl.when(t0 == bos)
  def _():
    scratch_ref[:] = jnp.zeros([MB, K, V], dtype=jnp.float32)
    if USE_INITIAL_STATE:
      scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]

  # === Stage 3+4 work — batched over MB heads ===
  # h: pre-update state for all MB heads in this tile.
  b_h = scratch_ref[:]   # [MB, K, V]

  # Spill pre-update h (saved residual for bwd; tagged with
  # checkpoint_name("kda_residuals") at the custom_vjp boundary).
  if STORE_H:
    h_out_ref[:, 0, 0] = b_h.astype(h_out_ref.dtype)  # [MB, K, V]

  b_w = w_ref[:,0,:]   # [MB, BT, K]
  b_u = u_ref[:,0,:]   # [MB, BT, V]

  # Stage 3 delta correction: v_new = u - w @ h
  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # HIGHEST precision: v_new feeds directly into the recursive state update.
  b_v_new = b_u.astype(jnp.float32) - jnp.matmul(
    b_w.astype(jnp.float32), b_h,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [MB, BT, V]

  # Spill v_new (used by bwd Stage 0 when disable_recompute=True).
  if STORE_V_NEW:
    v_new_out_ref[:,0,:] = b_v_new.astype(v_new_out_ref.dtype)

  # Stage 4 inter-chunk output: scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref))
  # Reference-point stabilization mirrors the production varlen kernel's
  # g_ref = g[0] choice for bit-identical numerics.
  b_q = q_ref[:,0,:]                              # [MB, BT, K]
  b_g = gk_ref[:,0,:].astype(jnp.float32)         # [MB, BT, K]
  b_A = A_ref[:,0,:]                               # [MB, BT, BT]

  b_g_ref_row = b_g[:, 0:1, :]                # [MB, 1, K]
  b_qg = b_q.astype(jnp.float32) * jnp.exp2(
    jnp.maximum(b_g - b_g_ref_row, -126.0)
  )                                            # [MB, BT, K]
  b_h_scaled = b_h * jnp.exp2(
    jnp.maximum(b_g_ref_row[:, 0, :], -126.0)
  )[:, :, None]                                # [MB, K, V]

  # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
  # Output-only GEMM: bf16 inputs use DEFAULT (no extra precision to preserve),
  # fp32 inputs use HIGHEST to maintain full mantissa fidelity.
  b_o = jnp.matmul(
    b_qg, b_h_scaled,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  ) * scale                                    # [MB, BT, V]

  # Stage 4 intra-chunk: A @ v_new
  # Apply lower-triangular mask: Aqk is causal by construction, but masking
  # here matches the original per-head loop behaviour and guards against
  # any tiny upper-triangle fp noise from the Neumann intra-chunk solve.
  m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]  # [BT, BT]
  b_A_f32 = jnp.where(m_s[None, :, :], b_A.astype(jnp.float32), 0.0)
  # [MB, BT, BT] @ [MB, BT, V] -> [MB, BT, V]
  b_o = b_o + jnp.matmul(
    b_A_f32, b_v_new,
    precision=OUTPUT_PRECISION,
    preferred_element_type=jnp.float32,
  )                                            # [MB, BT, V]

  o_ref[:,0,:] = b_o.astype(o_ref.dtype)

  # Stage 3 state update: h = decay(h) + kg^T @ v_new
  # HIGHEST precision: accumulates directly into the recursive hidden state.
  b_gk_last = gk_ref[:,0,:][:, BT - 1, :].astype(jnp.float32)  # [MB, K]
  b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None]          # [MB, K, V] decay

  b_kg = kg_ref[:,0,:]   # [MB, BT, K]
  # [MB, K, BT] @ [MB, BT, V] -> [MB, K, V]
  b_h_new = b_h_new + jnp.matmul(
    b_kg.astype(jnp.float32).transpose(0, 2, 1), b_v_new,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  scratch_ref[:] = b_h_new

  # === Final state (last chunk of THIS sequence) ===
  @pl.when(t0 + BT >= eos)
  def _():
    if STORE_FINAL_STATE:
      ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)  # [MB, K, V]


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "scale",
    "chunk_size",
    "store_h",
    "store_v_new",
    "store_intermediates",
    "mini_batch",
  ],
)
def chunk_kda_fwd_h_o_varlen(
  w: jax.Array,       # [B, T, H, K]
  u: jax.Array,       # [B, T, H, V]
  kg: jax.Array,      # [B, T, H, K]
  gk: jax.Array,      # [B, T, H, K]  -- g_cumsum
  q: jax.Array,       # [B, T, H, K]
  A: jax.Array,       # [B, T, H, BT]
  cu_seqlens: jax.Array,   # [N+1]
  chunk_indices: jax.Array | None = None,
  initial_state: jax.Array | None = None,  # [N, H, K, V]
  output_final_state: bool = False,
  scale: float = 1.0,
  chunk_size: int = 64,
  store_h: bool = False,
  store_v_new: bool = False,
  store_intermediates: bool | None = None,
  mini_batch: int | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
  """Fused Stage 3+4 for variable-length sequences.

  Eliminates the intermediate h [H, B, NT, K, V] and v_new [H, B, T, V]
  tensors from HBM by keeping h in VMEM scratch and v_new in registers
  inside a single Pallas kernel. Mirrors the fixed-length
  ``chunk_kda_fwd_h_o`` (commit b099ab6b) but adds varlen scaffolding
  (scalar prefetch + per-sequence real_NT bounds) following the same
  pattern as ``_chunk_gated_delta_rule_fwd_varlen_kernel``.

  Args:
    w:     [H, B, T, K] -- correction weights from intra-chunk.
    u:     [H, B, T, V] -- delta-corrected values from intra-chunk.
    kg:    [H, B, T, K] -- gated keys from intra-chunk.
    gk:    [H, B, T, K] -- g_cumsum (chunk-local cumsum, log2 space).
    q:     [H, B, T, K] -- query vectors.
    A:     [H, B, T, BT] -- intra-chunk attention matrix (Aqk).
    cu_seqlens:    [B, N+1] -- cumulative seq lengths (per-batch, BT-aligned).
    chunk_indices: [B, NT, 2] or None -- precomputed chunk indices (per-batch).
    initial_state: [B, N, H, K, V] or None -- per-sequence initial state.
    output_final_state: whether to return per-sequence final state.
    scale: attention scale factor.
    chunk_size: chunk size BT.
    store_h: spill per-chunk pre-update ``h`` to HBM (used by bwd
        save-h fast path; tagged with checkpoint_name("kda_residuals")
        at the custom_vjp boundary).
    store_v_new: spill per-token ``v_new`` to HBM (used by bwd Stage 0
        when ``disable_recompute=True``).
    store_intermediates: deprecated alias. When True, enables both
        ``store_h`` and ``store_v_new``. Prefer the split flags.
    mini_batch: int or None. Number of heads per grid point for DMA
        amortisation. When None (default), auto-computed to maximise
        VMEM utilisation (capped at min(H, 16)).

  Returns:
    o:           [H, B, T, V] -- outputs (head-first layout).
    final_state: [B, N, H, K, V] or None.
    h_per_chunk: [H, B, NT, K, V] or None -- pre-update h, only when
                 ``store_h`` (or back-compat ``store_intermediates``) is True.
    v_new:       [H, B, T, V] or None     -- delta-corrected v, only when
                 ``store_v_new`` (or back-compat ``store_intermediates``) is True.
  """
  # Back-compat shim: old single-flag callers map to both stores.
  if store_intermediates is not None:
    store_h = store_h or store_intermediates
    store_v_new = store_v_new or store_intermediates
  H, B, T, K = q.shape
  V = u.shape[-1]
  BT = chunk_size

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert cu_seqlens is not None
  # Ensure cu_seqlens is 2D [B, N+1] for kernel block specs
  if cu_seqlens.ndim == 1:
    cu_seqlens = jnp.broadcast_to(cu_seqlens[None, :], (B, cu_seqlens.shape[0]))
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(u, (H, B, T, V), "u")
  assert_shape(kg, (H, B, T, K), "kg")
  assert_shape(gk, (H, B, T, K), "gk")
  assert_shape(q, (H, B, T, K), "q")
  assert_shape(A, (H, B, T, BT), "A")

  N = cu_seqlens.shape[-1] - 1
  # Varlen initial_state must be 5D (B, N, H, K, V)
  if initial_state is not None:
    assert initial_state.ndim == 5, (
      f"Varlen initial_state must be 5D (B, N, H, K, V), got ndim={initial_state.ndim}"
    )
    assert_shape_or_none(initial_state, (B, N, H, K, V), "initial_state")
  assert K <= 256, "current kernel does not support K > 256."

  hw = get_tpu_config()
  K_PADSIZE = int(align_up(K, hw.block_align_major))
  V_ALIGNED = int(align_up(V, hw.block_align_major))

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = q.dtype.itemsize
    # scratch per head: h_state[K_PADSIZE*V_ALIGNED] in f32 + i/o buffers
    in_bytes = (BT * K_PADSIZE + BT * V_ALIGNED + BT) * elem_size
    out_bytes = BT * V_ALIGNED * elem_size
    scratch_bytes = K_PADSIZE * V_ALIGNED * 4  # float32 accumulator
    per_head = in_bytes + out_bytes + scratch_bytes
    MB = estimate_mini_batch(per_head, H, max_mb=16)
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # Generate chunk_to_seq mapping from chunk_indices
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
  NT = chunk_indices.shape[-2]
  chunk_to_seq = chunk_indices[..., 0].astype(jnp.int32)  # [NT] or [B, NT]
  # Kernel block specs index with [b, c], so ensure 2D [B, NT].
  if chunk_to_seq.ndim == 1:
    chunk_to_seq = jnp.broadcast_to(chunk_to_seq[None, :], (B, NT))

  # grid = (H // MB, NT) with NT = T // BT, so chunk index c ranges
  # 0..NT-1 and the maximum T offset accessed is (NT-1)*BT+BT = T.
  # No extra trailing chunk is ever touched, so T_alloc == T suffices.
  T_alloc = T

  # Pad K (last dim) to K_PADSIZE if needed, then transpose to
  # [B=1, H, T, K_PADSIZE]. No T-dim padding required (see T_alloc above).
  # Inputs stay in their original dtype (typically bf16); the kernel casts
  # tile-level slices to f32 on the fly in VMEM, avoiding a full-tensor
  # bf16->f32 cast in HBM that doubles memory and DMA bandwidth.
  def _pad_kdim_then_t(x, dim_pad):
    if dim_pad > 0:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim_pad)))
    return x

  w_t  = _pad_kdim_then_t(w,  K_PADSIZE - K)
  kg_t = _pad_kdim_then_t(kg, K_PADSIZE - K)
  gk_t = _pad_kdim_then_t(gk, K_PADSIZE - K)
  q_t  = _pad_kdim_then_t(q,  K_PADSIZE - K)
  u_t  = _pad_kdim_then_t(u,  V_ALIGNED - V)

  # A is [B, T, H, BT]; transpose to [B=1, H, T, BT]. No T padding needed.
  A_t = A  # [H, B, T, BT]

  # h0: [B, N, H, K, V] -> pad K and V -> [B, N, H, K_PADSIZE, V_ALIGNED]
  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # Index maps with MB heads per grid point. Scalar prefetch order:
  # (seqlens_ref, chunk_to_seq_ref).
  # Inputs are [H, B, T_alloc, X]; BlockSpec slices MB heads at h*MB.
  def _t_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  def _A_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  bspec_v = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  bspec_a = pl.BlockSpec([MB, 1, BT, BT],        index_map=_A_index_map)
  bspec_h0 = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if h0 is not None else None
  )

  # Output specs.
  o_spec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  ht_spec = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        b, chunk_to_seq_ref[b, c], h, 0, 0
      ),
    )
    if output_final_state else None
  )
  # Per-chunk h spill (pre-update, used by bwd save-h fast path). Layout
  # [H, B, NT, K_PADSIZE, V_ALIGNED]
  h_out_spec = (
    pl.BlockSpec(
      [MB, 1, 1, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (
        h, b, c, 0, 0
      ),
    )
    if store_h else None
  )
  # Per-token v_new spill, [H, B, T_alloc, V_ALIGNED].
  v_new_out_spec = (
    pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    if store_v_new else None
  )

  o_shape = jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
  ht_shape = (
    jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state else None
  )
  # h_per_chunk dtype mirrors u (matches unfused chunk_gated_delta_rule_fwd_h
  # which produces h in u.dtype, typically bf16/fp32).
  h_out_shape = (
    jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], u.dtype)
    if store_h else None
  )
  v_new_out_shape = (
    jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], u.dtype)
    if store_v_new else None
  )

  scratch = pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32)
  grid = (H // MB, B, NT)
  interpret = get_interpret()

  # bf16 inputs: DEFAULT precision is lossless (operands already have ~7-bit
  # mantissa); fp32 inputs: HIGHEST preserves full 23-bit mantissa fidelity.
  # State-update matmuls always use HIGHEST regardless (recursive accumulation).
  _output_prec = (
    jax.lax.Precision.DEFAULT
    if q.dtype == jnp.bfloat16
    else jax.lax.Precision.HIGHEST
  )

  o_out, ht_out, h_out, v_new_out = pl.pallas_call(
    functools.partial(
      _chunk_kda_fwd_h_o_varlen_kernel,
      BT=BT,
      scale=scale,
      USE_INITIAL_STATE=(h0 is not None),
      STORE_FINAL_STATE=output_final_state,
      STORE_H=store_h,
      STORE_V_NEW=store_v_new,
      MB=MB,
      OUTPUT_PRECISION=_output_prec,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=[
        bspec_k,   # w
        bspec_v,   # u
        bspec_k,   # kg
        bspec_k,   # gk
        bspec_k,   # q
        bspec_a,   # A
        bspec_h0,  # h0
      ],
      out_specs=[o_spec, ht_spec, h_out_spec, v_new_out_spec],
      scratch_shapes=[scratch],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      disable_bounds_checks=True,
    ),
    out_shape=[o_shape, ht_shape, h_out_shape, v_new_out_shape],
    interpret=interpret,
  )(cu_seqlens.astype(jnp.int32), chunk_to_seq, w_t, u_t, kg_t, gk_t, q_t, A_t, h0)

  # Post-process: o is [H, 1, T, V_ALIGNED]
  if V_ALIGNED > V:
    o_out = o_out[..., :V]
  o_out = o_out.astype(u.dtype)

  if output_final_state and ht_out is not None:
    if V_ALIGNED > V:
      ht_out = ht_out[..., :V]
    if K_PADSIZE > K:
      ht_out = ht_out[..., :K, :]

    # Handle empty sequences: sequences with no chunks never execute kernel code,
    # so their final_state is uninitialized. Fill them with initial_state or zeros.
    seq_lens = jnp.diff(cu_seqlens, axis=-1)
    empty_mask = (seq_lens == 0)  # [B, N]
    if initial_state is not None:
      # For empty sequences, final_state should equal initial_state
      fill_value = initial_state[:, :, :, :K, :V]
    else:
      # For empty sequences without initial_state, final_state should be zeros
      fill_value = jnp.zeros((B, N, H, K, V), dtype=ht_out.dtype)
    # Use where to selectively replace empty sequence states
    ht_out = jnp.where(empty_mask[:, :, None, None, None], fill_value, ht_out)
  else:
    ht_out = None

  # Post-process spilled intermediates (independently controlled).
  if store_h:
    # h_out: [1, NT, H, K_PADSIZE, V_ALIGNED] -> trim K/V padding back.
    if V_ALIGNED > V:
      h_out = h_out[..., :V]
    if K_PADSIZE > K:
      h_out = h_out[..., :K, :]

  else:
    h_out = None
  if store_v_new:
    if V_ALIGNED > V:
      v_new_out = v_new_out[..., :V]
  else:
    v_new_out = None


  return o_out, ht_out, h_out, v_new_out


# =============================================================================
# Variable-length alignment helpers
# =============================================================================

def _align_seqs(tensors_4d, tensors_3d, cu_seqlens, align):
  """Align (pad) each variable-length sequence to a multiple of ``align``.

  Supports both single-batch (cu_seqlens [N+1]) and batched
  (cu_seqlens [B, N+1]) modes.  In batched mode, each batch element is
  aligned independently and all results are padded to the maximum
  aligned T across batches.
  """
  if cu_seqlens.ndim == 2:
    # Batched: loop over B (values are concrete at trace time).
    B = cu_seqlens.shape[0]
    per_batch_4d = [[] for _ in tensors_4d]
    per_batch_3d = [[] for _ in tensors_3d]
    padded_cus = []
    t_aligned_sizes = []
    for b in range(B):
      t4 = [t[:, b:b+1, :, :] for t in tensors_4d]
      t3 = [t[:, b:b+1, :] for t in tensors_3d]
      aligned_4d, aligned_3d, padded_cu_b, _ = _align_seqs(
        t4, t3, cu_seqlens[b], align
      )
      for idx, a in enumerate(aligned_4d):
        per_batch_4d[idx].append(a)
      for idx, a in enumerate(aligned_3d):
        per_batch_3d[idx].append(a)
      padded_cus.append(padded_cu_b)
      t_aligned_sizes.append(aligned_4d[0].shape[2])

    T_max = max(t_aligned_sizes)
    # Pad each batch element to T_max and concatenate along B.
    def _pad_and_cat_4d(tensors_per_batch):
      padded = []
      for t in tensors_per_batch:
        pad_len = T_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    def _pad_and_cat_3d(tensors_per_batch):
      padded = []
      for t in tensors_per_batch:
        pad_len = T_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    out_4d = [_pad_and_cat_4d(per_batch_4d[i]) for i in range(len(tensors_4d))]
    out_3d = [_pad_and_cat_3d(per_batch_3d[i]) for i in range(len(tensors_3d))]
    stacked_cu = jnp.stack(padded_cus, axis=0)
    return out_4d, out_3d, stacked_cu, cu_seqlens

  # --- Single-batch path (original) ---
  N = cu_seqlens.shape[0] - 1
  T_old = tensors_4d[0].shape[2]

  seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
  padded_lens = ((seg_lens + align - 1) // align) * align
  padded_cu = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_lens)])
  T_new = ((T_old + N * (align - 1) + align - 1) // align) * align

  def _build_gather(i, gather_idx):
    old_start = cu_seqlens[i]
    new_start = padded_cu[i]
    sl = seg_lens[i]
    j = jnp.arange(T_new)
    in_seg = (j >= new_start) & (j < new_start + sl)
    src = old_start + (j - new_start)
    return jnp.where(in_seg, src, gather_idx)

  gather_idx = jnp.full(T_new, T_old, dtype=jnp.int32)
  gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)

  def repack_4d(t):
    # t: [H, B, T, K] — gather along axis 2 (T dimension)
    return jnp.pad(t, ((0, 0), (0, 0), (0, T_new - T_old), (0, 0)))[:, :, gather_idx]

  def repack_3d(t):
    # t: [H, B, T] — gather along axis 2 (T dimension)
    return jnp.pad(t, ((0, 0), (0, 0), (0, T_new - T_old)))[:, :, gather_idx]

  return (
    [repack_4d(t) for t in tensors_4d],
    [repack_3d(t) for t in tensors_3d],
    padded_cu,
    cu_seqlens,
  )


def _unalign_output(o, orig_cu_seqlens, aligned_cu_seqlens, T_out):
  """Reverse _align_seqs: scatter aligned output back to original positions.

  Supports batched cu_seqlens [B, N+1] — processes each batch element
  independently.
  """
  if orig_cu_seqlens.ndim == 2:
    B = orig_cu_seqlens.shape[0]
    per_batch = []
    for b in range(B):
      # Use slicing that works for both 3D [H,B,T] and 4D [H,B,T,X]
      ob_slice = jax.lax.dynamic_slice_in_dim(o, b, 1, axis=1)
      ob = _unalign_output(
        ob_slice,
        orig_cu_seqlens[b],
        aligned_cu_seqlens[b],
        T_out,
      )
      per_batch.append(ob)
    return jnp.concatenate(per_batch, axis=1)

  # --- Single-batch path (original) ---
  N = orig_cu_seqlens.shape[0] - 1
  orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]

  def _build_gather(i, gather_idx):
    orig_start = orig_cu_seqlens[i]
    aligned_start = aligned_cu_seqlens[i]
    sl = orig_seg_lens[i]
    j = jnp.arange(T_out)
    in_seg = (j >= orig_start) & (j < orig_start + sl)
    src = aligned_start + (j - orig_start)
    return jnp.where(in_seg, src, gather_idx)

  # Default to aligned_cu_seqlens[-1] — a known-zero padding position.
  # After the _align_seqs fix above, T_aligned > padded_cu[-1], so this
  # index is always valid and always reads padding (zero).
  safe_default = aligned_cu_seqlens[-1]
  gather_idx = jnp.full(T_out, safe_default, dtype=jnp.int32)
  gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)
  return o[:, :, gather_idx]


# =============================================================================
# Forward orchestrators
# =============================================================================

@jaxtyping.jaxtyped
def chunk_kda_fwd(
  q: Float[Array, "H B T K"],
  k: Float[Array, "H B T K"],
  v: Float[Array, "H B T V"],
  g: Float[Array, "H B T K"],
  beta: Float[Array, "H B T"],
  scale: float,
  initial_state: Float[Array, "B N H K V"] | None,
  output_final_state: bool,
  use_qk_l2norm_in_kernel: bool = False,
  cu_seqlens: Int[Array, "B N_MAX"] | None = None,
  chunk_indices: Int[Array, "B NT 2"] | None = None,
  chunk_size: int = 64,
  safe_gate: bool = True,
  lower_bound: float | None = None,
  use_gate_in_kernel: bool = False,
  A_log: Float[Array, "H"] | None = None,
  dt_bias: Float[Array, "H*K"] | None = None,
  disable_recompute: bool = False,
  cp_context: CPContext | None = None,
  _skip_align: bool = False,
  segment_ids: Int[Array, "B T"] | None = None,
):
  """KDA chunked forward pass using Neumann intra-chunk approximation.

  Signature aligned with FLA's ``chunk_kda_fwd`` from
  ``fla.ops.kda.chunk_fwd``. Four-stage pipeline:
    1. Gate activation + chunk-local cumsum (or cumsum-only if gates
       are pre-activated).
    2. Intra-chunk delta-rule solve via Neumann series.
    3. Inter-chunk hidden state propagation via delta-rule recurrence.
    4. Output computation (inter-chunk state + intra-chunk attention).

  Args:
      q:     [H, B, T, K]    -- query vectors (head-first layout).
      k:     [H, B, T, K]    -- key vectors.
      v:     [H, B, T, V]    -- value vectors.
      g:     [H, B, T, K]    -- per-element gate. Raw input when
                                use_gate_in_kernel=True, or pre-activated
                                (natural log space) when False.
      beta:  [H, B, T]       -- per-token scalar mixing coefficient.
      scale: float            -- attention scale factor (e.g. K ** -0.5).
      initial_state: [B, H, K, V], [N, H, K, V], or
                     [B, N, H, K, V] or None -- initial hidden state.
                     Non-varlen: 4D ``[B, H, K, V]``.  Varlen: 5D
                     ``[B, N, H, K, V]`` or 4D ``[N, H, K, V]``.
      output_final_state: bool -- whether to return the final hidden state.
      cu_seqlens: [N+1], [B, N+1] or None -- cumulative sequence lengths
                  (varlen).  Batched callers pass ``[B, N+1]``.
      chunk_indices: [NT, 2], [B, NT, 2] or None -- chunk index mapping
                     for varlen.  Batched callers pass ``[B, NT, 2]``.
      chunk_size: int         -- tile size BT (default 64).
      safe_gate: bool         -- reserved for midpoint stabilization.
      lower_bound: float or None -- if set, use sigmoid gate variant.
      use_gate_in_kernel: bool -- True: fuse gate activation + cumsum
                                 (requires A_log). False: cumsum only
                                 (gate pre-activated).
      A_log: [H] or None      -- log of decay parameter A. Required when
                                 use_gate_in_kernel=True.
      dt_bias: [H*K] or None  -- bias added to g before gate activation.
      disable_recompute: bool -- True: keep intermediates (w, u, kg, etc.).
                                 False: release them to save memory.
      cp_context: CPContext or None -- context parallelism metadata.

  Returns:
      12-tuple matching FLA's ``chunk_kda_fwd``:
        o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state
      All output tensors use head-first ``[H, B, T, ...]`` layout.
      When ``disable_recompute=False``, w/u/qg/kg/v_new are None.
      When ``disable_recompute=True``, h is saved for bwd reuse (v_new is
      derived from it in bwd Stage 0, not stored directly).
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size

  if use_qk_l2norm_in_kernel:
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

  # Context Parallel (CP) dispatch flag. The actual constraints
  # (initial_state / output_final_state / cu_seqlens) are enforced at the
  # chunk_kda entry; by the time we get here cp_context is either None or
  # fully populated by `_derive_cp_metadata_from_segment_ids`.
  _cp_active = cp_context is not None and cp_context.is_cp_enabled

  if segment_ids is not None and cu_seqlens is None:
    if segment_ids.ndim == 2 and segment_ids.shape[0] > 1:
      # B > 1: per-batch cu_seqlens
      cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=cdiv(T, BT))
    else:
      _seg1d = segment_ids[0] if segment_ids.ndim == 2 else segment_ids
      cu_seqlens = segment_ids_to_seqlens(_seg1d, max_segs=cdiv(T, BT))

  N = cu_seqlens.shape[-1] - 1 if cu_seqlens is not None else B
  # When segment_ids produces cu_seqlens padded to max_segs, N_padded may
  # exceed the actual number of segments in initial_state.  Pad with zeros
  # so the assertion passes (phantom segments use zero initial state).
  _is_varlen = cu_seqlens is not None
  if initial_state is not None:
    if _is_varlen:
      # Varlen: initial_state must be 5D (B, N, H, K, V)
      assert initial_state.ndim == 5, (
        f"Varlen initial_state must be 5D (B, N, H, K, V), got ndim={initial_state.ndim}"
      )
      if initial_state.shape[1] < N:
        pad_n = N - initial_state.shape[1]
        initial_state = jnp.pad(initial_state, ((0, 0), (0, pad_n), (0, 0), (0, 0), (0, 0)))
      assert_shape_or_none(initial_state, (B, N, H, K, V), "initial_state")
    else:
      # Non-varlen: initial_state is 4D (B, H, K, V) or 5D (B, 1, H, K, V)
      if initial_state.ndim == 5:
        initial_state = initial_state[:, 0]
      if initial_state.shape[0] < N:
        pad_n = N - initial_state.shape[0]
        initial_state = jnp.pad(initial_state, ((0, pad_n), (0, 0), (0, 0), (0, 0)))
      assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")

  _orig_cu_seqlens = cu_seqlens
  if cu_seqlens is not None and not _skip_align:
    # Varlen alignment
    T_input = T
    [q, k, v, g], [beta], cu_seqlens, _ = _align_seqs(
      [q, k, v, g],
      [beta],
      cu_seqlens,
      align=BT,
    )
    T = q.shape[2]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T)
    # Fix: _align_seqs pads g with 0, but softplus(0 + dt_bias) != 0 when
    # use_gate_in_kernel=True, producing non-zero gate activation at padding
    # positions.  This corrupts g_last (used for state propagation in Stage 3)
    # and kg (used for state update).  Set padding g to a large negative so
    # softplus(large_neg + dt_bias) ≈ 0, neutralising padding positions.
    if use_gate_in_kernel:
      orig_lens = jnp.diff(_orig_cu_seqlens, axis=-1)
      aligned_starts = cu_seqlens[..., :-1]
      pos = jnp.arange(T)
      if _orig_cu_seqlens.ndim == 1:
        in_range = (pos[None, :] >= aligned_starts[:, None]) & (
          pos[None, :] < (aligned_starts + orig_lens)[:, None]
        )
        valid_mask = in_range.any(axis=0)  # [T]
        g = jnp.where(valid_mask[None, None, :, None], g, -1e4)
      else:
        for b in range(_orig_cu_seqlens.shape[0]):
          in_range = (pos[None, :] >= aligned_starts[b, :, None]) & (
            pos[None, :] < (aligned_starts[b] + orig_lens[b])[:, None]
          )
          valid_mask = in_range.any(axis=0)
          g = g.at[:, b].set(
            jnp.where(valid_mask[None, :, None], g[:, b], -1e4)
          )
  elif cu_seqlens is not None and _skip_align:
    # Data already aligned by caller; just compute chunk_indices
    T_input = T
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT, max_T=T)

  assert T % BT == 0, f"Sequence length T={T} must be divisible by chunk_size={BT}"
  # ------------------------------------------------------------------
  # Step 1 + 2 (Fused): Gate cumsum + Intra-chunk solve
  # ------------------------------------------------------------------
  w, u, qg, kg, Aqk, Akk, g_cumsum = kda_fwd_intra_fused(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    scale=scale,
    cu_seqlens=cu_seqlens,
    chunk_size=BT,
    chunk_indices=chunk_indices,
    safe_gate=safe_gate,
    disable_recompute=disable_recompute,
    cumsum_scale=_RCP_LN2,
    A_log=A_log,
    dt_bias=dt_bias,
    use_gate_in_kernel=use_gate_in_kernel,
    lower_bound=lower_bound,
  )

  # ------------------------------------------------------------------
  # Stage CP (between Stage 1+2 and Stage 3): pre-process + all-gather +
  # merge to recover the rank-local initial_state from upstream ranks.
  #
  # Algorithm (design-doc §2.2 / §2.3):
  #   1. Pre-process: each rank assumes S_in = 0 and computes (S_ext, M)
  #      for its LAST segment only (segments fully within a rank start
  #      fresh and need no merge).
  #   2. All-gather both tensors across the cp axis (fp32; design-doc
  #      §2.5 red line).
  #   3. Locally merge upstream (S_ext, M) into S_in for THIS rank's
  #      first segment via M_j @ S_in + S_ext_j.
  #   4. Construct initial_state = [N_local, H, K, V] with [0] = S_in
  #      and the rest zero (intermediate segments start fresh).
  #
  # The recovered initial_state is fed into the fused Stage 3+4 path
  # (chunk_kda_fwd_h_o_varlen) — it accepts h0 via USE_INITIAL_STATE.
  # ------------------------------------------------------------------
  if _cp_active:
    # cu_seqlens here is the BT-aligned version produced by _align_seqs
    # above. pre-process is fully device-side and accepts traced
    # cu_seqlens — no host-side cu_seqlens_cpu needed. Reuse the
    # chunk_indices already computed above so pre-process doesn't redo
    # the prepare_chunk_indices work.
    S_ext_local, M_local = chunk_gated_delta_rule_fwd_h_pre_process(
      k=kg,
      w=w,
      u=u,
      gk=g_cumsum,
      cu_seqlens=cu_seqlens,
      chunk_indices=chunk_indices,
      chunk_size=BT,
      use_exp2=True,
    )
    S_ext_all, _ = all_gather_into_tensor(S_ext_local, cp_context.axis_name)
    M_all, _ = all_gather_into_tensor(M_local, cp_context.axis_name)
    rank = jax.lax.axis_index(cp_context.axis_name)

    pre_num = cp_context.pre_num_ranks
    is_first = cp_context.is_first_rank
    if B > 1 and hasattr(pre_num, 'ndim') and pre_num.ndim > 0:
      # Per-batch chain metadata: loop over B, call merge per batch element
      s_in_list = []
      for b in range(B):
        s_b = _merge_initial_state(
          S_ext_all[:, :, b:b+1],  # [cp, H, 1, K, V]
          M_all[:, :, b:b+1],      # [cp, H, 1, K, K]
          rank,
          pre_num[b],
          is_first[b],
        )
        s_in_list.append(s_b)  # [H, 1, K, V]
      S_in_first = jnp.concatenate(s_in_list, axis=1)  # [H, B, K, V]
    else:
      S_in_first = _merge_initial_state(
        S_ext_all, M_all, rank, pre_num, is_first,
      )
    # Build per-segment initial_state: only the FIRST segment inherits
    # state from upstream ranks; the rest start at zero.
    # S_in_first: [H, B, K, V] fp32 → transpose to [B, H, K, V]
    S_in_first_bhkv = jnp.transpose(S_in_first, (1, 0, 2, 3))  # [B, H, K, V]
    initial_state = (
      jnp.zeros((B, N, H, K, V), dtype=jnp.float32)
      .at[:, 0].set(S_in_first_bhkv)
    )

  # ------------------------------------------------------------------
  # Step 3 + Step 4: Inter-chunk state + Output (gather/scatter for varlen)
  #
  # The inter-chunk kernel and output kernel require BT-aligned
  # cu_seqlens (they index blocks via bos // BT).  For non-aligned
  # varlen sequences we reuse the existing _align_seqs / _unalign_output
  # utilities to gather inputs into a chunk-aligned layout, run Stages
  # 3 & 4, then scatter the output back.
  #
  # Padding semantics (relied on for correctness):
  #   - k/w/u/q/Aqk at padded tail positions = 0 (from _align_seqs's
  #     jnp.pad with default fill value 0). With zero k/w/u/q the state
  #     update reduces to h_pad = h_{t-1} * exp(g_pad) + 0, and the
  #     output reduces to q_pad * exp(...) @ h = 0 * ... = 0.
  #   - g_cumsum at padded tail positions = 0 ⇒ exp(g_pad) = 1, so the
  #     hidden state is carried through padded positions unchanged.
  # ------------------------------------------------------------------
  # Stage 1/2 already operate in BT-aligned layout; derived tensors inherit
  # that layout. Use the fused Stage 3+4 kernel for both varlen and fixed
  # inputs. A fixed batch is represented as one sequence [0, T] per batch.
  if _is_varlen:
    stage34_cu_seqlens = cu_seqlens
    stage34_chunk_indices = chunk_indices
    stage34_initial_state = initial_state
  else:
    stage34_cu_seqlens = jnp.broadcast_to(
      jnp.asarray([0, T], dtype=jnp.int32), (B, 2)
    )
    block_ids = jnp.arange(T // BT, dtype=jnp.int32)
    fixed_chunk_indices = jnp.stack(
      [jnp.zeros_like(block_ids), block_ids], axis=-1
    )
    stage34_chunk_indices = jnp.broadcast_to(
      fixed_chunk_indices[None, :, :], (B, T // BT, 2)
    )
    stage34_initial_state = (
      initial_state[:, None, ...] if initial_state is not None else None
    )

  o, final_state, h, v_new = chunk_kda_fwd_h_o_varlen(
    w=w,
    u=u,
    kg=kg,
    gk=g_cumsum,
    q=q,
    A=Aqk,
    cu_seqlens=stage34_cu_seqlens,
    chunk_indices=stage34_chunk_indices,
    initial_state=stage34_initial_state,
    output_final_state=output_final_state,
    scale=scale,
    chunk_size=BT,
    store_h=disable_recompute,
    store_v_new=False,
  )
  if not _is_varlen and final_state is not None:
    final_state = final_state[:, 0]

  # Unalign output (input was padded by _align_seqs; scatter wrote to
  # aligned positions, now map back to original cu_seqlens layout).
  if _orig_cu_seqlens is not None and not _skip_align:
    o = o.astype(q.dtype)
    o = _unalign_output(o, _orig_cu_seqlens, cu_seqlens, T_input)
  # ------------------------------------------------------------------
  # Memory optimization: release intermediates (disable_recompute=False)
  # ------------------------------------------------------------------
  if not disable_recompute:
    w, u, qg, kg, v_new = None, None, None, None, None
    h = None
    if use_gate_in_kernel:
      g_cumsum = None

  return o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state


def chunk_kda_fwd_custom(
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
    cu_seqlens, N_max = segment_ids_to_cu_seqlens(
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
    q_hat, rstd_q = l2norm_fwd(q_a)
    k_hat, rstd_k = l2norm_fwd(k_a)
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
  ) = chunk_kda_fwd(
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
      as_public_final_state(final_state, segment_ids=segment_ids),
  ), residuals
