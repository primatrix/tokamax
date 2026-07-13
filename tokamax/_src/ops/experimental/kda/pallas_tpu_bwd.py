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
"""Pallas TPU backward kernels for experimental KDA."""

from __future__ import annotations



# =============================================================================
# Mini-batch sizing helper
# =============================================================================

"""Unified VMEM-aware mini-batch estimation for Pallas TPU kernels."""



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
    from tokamax._src.ops.experimental.kda.utils import get_tpu_config
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
# Chunk-local cumsum helpers
# =============================================================================

"""Chunk-local cumulative sum with multiple backends.

Fixed-length inputs dispatch to a log-depth parallel scan or matmul;
variable-length inputs use a Pallas kernel with vectorised Hillis-Steele
prefix scan.
"""

import functools
import math

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
    export_public,
    get_interpret,
)

_TRIU_PRECISION = jax.lax.Precision.HIGHEST
_VMEM_HW_LIMIT_BYTES = 30 * 1024 * 1024  # 30 MB (TPUv6e VMEM ~32 MB)


# =============================================================================
# Fixed-length: Hillis-Steele scan (optimal for head_first=False)
# =============================================================================


def _hillis_steele_scan(
  x: jax.Array,
  axis: int,
) -> jax.Array:
  """Inclusive prefix scan with O(log n) parallel depth along ``axis``.

  Each iteration adds values from ``2 ** d`` positions to the left, so the
  dependency horizon doubles every round instead of advancing one element at a
  time as in a sequential cumsum.

  Args:
      x:    input tensor.
      axis: axis along which to compute the inclusive prefix scan.

  Returns:
      Tensor with the same shape as ``x``.
  """
  axis = axis % x.ndim
  length = x.shape[axis]
  assert length >= 1, f"scan axis must be non-empty, got length={length}"

  if length == 1:
    return x

  num_steps = int(math.log2(length))
  assert length == 1 << num_steps, (
    f"Hillis-Steele scan requires power-of-2 length, got {length}"
  )

  acc = x
  for i in range(num_steps):
    stride = 1 << i
    prefix = jax.lax.slice_in_dim(acc, 0, stride, axis=axis)
    suffix = jax.lax.slice_in_dim(acc, stride, length, axis=axis)
    shifted = jax.lax.slice_in_dim(acc, 0, length - stride, axis=axis)
    acc = jax.lax.concatenate([prefix, suffix + shifted], dimension=axis)

  return acc


def _chunk_local_cumsum_origin(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Chunk-local cumsum via Hillis-Steele scan (best for ``head_first=False``).

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
    cum_axis = 3
  else:
    g_work = jnp.pad(g, ((0, 0), (0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(B, NT, BT, H, S).astype(jnp.float32)
    cum_axis = 2

  if reverse:
    g_chunked = jnp.flip(g_chunked, axis=cum_axis)

  o_chunked = _hillis_steele_scan(g_chunked, axis=cum_axis)

  if reverse:
    o_chunked = jnp.flip(o_chunked, axis=cum_axis)

  if head_first:
    o = o_chunked.reshape(H, B, T_padded, S)[:, :, :T, :]
  else:
    o = o_chunked.reshape(B, T_padded, H, S)[:, :T, :, :]

  if scale is not None:
    o = o * scale

  return o.astype(out_dtype)


# =============================================================================
# Fixed-length: matmul with tril/triu mask (optimal for head_first=True)
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


# =============================================================================
# Full cumsum: recursive hierarchical triu matmul
# =============================================================================


def _triu_dot(a, b, contracting_a, contracting_b):
  """``dot_general`` with highest precision for triu matmul cumsum."""
  return jax.lax.dot_general(
    a, b,
    dimension_numbers=((contracting_a, contracting_b), ((), ())),
    precision=_TRIU_PRECISION,
  )


def _recursive_cumsum_2d(x_2d, chunk_size):
  """Recursive triu matmul cumsum on ``(B, L)`` along axis 1.

  Splits the sequence into ``chunk_size``-wide blocks, computes local
  cumsum via upper-triangular matmul, then recursively prefix-sums the
  chunk totals to obtain inter-chunk offsets.

  Recursion depth: ``ceil(log_{chunk_size}(L))``, typically <= 3.

  Args:
      x_2d:       [B, L] — flattened input.
      chunk_size:  block size for each recursion level.

  Returns:
      [B, L] — inclusive cumulative sum.
  """
  B, L = x_2d.shape

  # Base case: L fits in a single block
  if L <= chunk_size:
    triu = jnp.triu(jnp.ones((L, L), dtype=x_2d.dtype))
    return _triu_dot(x_2d, triu, (1,), (0,))

  # Pad to multiple of chunk_size
  seq_pad = (chunk_size - L % chunk_size) % chunk_size
  if seq_pad > 0:
    x_2d = jnp.pad(x_2d, [(0, 0), (0, seq_pad)])
  L_padded = L + seq_pad
  num_chunks = L_padded // chunk_size

  x_3d = x_2d.reshape(B, num_chunks, chunk_size)

  # Local cumsum within each chunk
  triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=x_3d.dtype))
  local_cs = _triu_dot(x_3d, triu, (2,), (0,))  # (B, num_chunks, chunk_size)

  # Chunk totals → recursive inclusive prefix sum
  chunk_totals = local_cs[:, :, -1]  # (B, num_chunks)
  totals_cumsum = _recursive_cumsum_2d(chunk_totals, chunk_size)

  # Exclusive offsets = inclusive_cumsum - own_total
  offsets = totals_cumsum - chunk_totals  # (B, num_chunks)
  result = local_cs + offsets[:, :, None]

  return result.reshape(B, L_padded)[:, :L]


def cumsum_triu_recursive(
  x: jax.Array,
  axis: int = -1,
  chunk_size: int = 128,
) -> jax.Array:
  """Full cumulative sum via recursive hierarchical triu matmul.

  Splits the target axis into ``chunk_size``-wide blocks and computes
  local cumsums via upper-triangular matrix multiplication, then
  recursively prefix-sums the chunk totals to propagate inter-chunk
  offsets.  All matmuls use ``Precision.HIGHEST`` for float32 accuracy.

  Equivalent to ``jnp.cumsum(x, axis=axis)`` but uses O(chunk_size^2)
  triu matrices instead of O(L) sequential scan, making it more
  TPU-friendly for long sequences.

  Args:
      x:          input tensor of any shape.
      axis:       axis along which to compute cumsum (default -1).
      chunk_size: block size per recursion level (default 128, aligned
                  to TPU MXU).

  Returns:
      Tensor of same shape and dtype as ``x`` — inclusive cumulative sum.
  """
  ndim = x.ndim
  assert ndim >= 1, f"x must have at least 1 dimension, got {ndim}"

  axis = axis % ndim
  L = x.shape[axis]

  # Flatten to (B, L)
  x_work = jnp.moveaxis(x, axis, -1)
  batch_shape = x_work.shape[:-1]
  B = math.prod(batch_shape) if batch_shape else 1
  x_2d = x_work.reshape(B, L)

  result = _recursive_cumsum_2d(x_2d, chunk_size)

  # Restore original shape
  result = result.reshape(*batch_shape, L) if batch_shape else result.reshape(L)
  return jnp.moveaxis(result, -1, axis)


# =============================================================================
# Pallas kernel: vectorised Hillis-Steele prefix scan (varlen)
# =============================================================================



def _chunk_cumsum_kernel_varlen(
    s_ref,
    o_ref,
    *,
    BT: int,
    NT: int,
    NT_PER_BLOCK: int,
    REVERSE: bool,
    HAS_SCALE: bool,
    scale: float,
):
    num_steps = int(math.log2(BT))
    i_t_block = pl.program_id(2)
    chunk_start = i_t_block * NT_PER_BLOCK

    def body(i_t_local, _):
        # Each chunk lives at local T offset (i_t_local * BT) within this
        # grid block. Assumes segment lengths are BT-aligned.
        local_start_t = i_t_local * BT
        local_start_t = pl.multiple_of(local_start_t, BT)

        s = s_ref[:, dslice(local_start_t, BT), :].astype(jnp.float32)

        if REVERSE:
            for d in range(num_steps):
                stride = 1 << d
                top = s[:, : BT - stride, :] + s[:, stride:, :]
                bot = s[:, BT - stride :, :]
                s = jnp.concatenate([top, bot], axis=1)
        else:
            for d in range(num_steps):
                stride = 1 << d
                top = s[:, :stride, :]
                bot = s[:, stride:, :] + s[:, :-stride, :]
                s = jnp.concatenate([top, bot], axis=1)

        if HAS_SCALE:
            s = s * scale

        o_ref[:, dslice(local_start_t, BT), :] = s.astype(o_ref.dtype)
        return 0

    # Tail grid block along T may have fewer chunks than NT_PER_BLOCK.
    iters = jnp.minimum(NT_PER_BLOCK, jnp.maximum(NT - chunk_start, 0))
    jax.lax.fori_loop(0, iters, body, 0)



# =============================================================================
# Pallas launcher for varlen mode
# =============================================================================


def _chunk_local_cumsum_pallas(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Pallas-based chunk-local cumsum for variable-length inputs.

  Uses BlockSpec to tile BH and S dimensions, with ``fori_loop`` +
  ``dslice`` to iterate over variable-length chunks along T inside the
  kernel.  BB is dynamically shrunk to fit tiles within hardware VMEM.

  Args:
      g:              [B, T, H, S] or [H, B, T, S] — input gates.
      chunk_size:     block size along T (must be power of 2).
      reverse:        if True, compute reverse (suffix) cumsum.
      scale:          optional multiplicative scale applied to the output.
      head_first:     if True, ``g`` is [H, B, T, S]; otherwise [B, T, H, S].
      output_dtype:   dtype of the output tensor (default float32).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum.
  """
  assert g.ndim == 4, f"g must be 4-D, got {g.ndim}-D"
  assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be power of 2"

  BT = chunk_size
  BS = 128
  BB = 8

  if head_first:
    H, B, T, S = g.shape
    g_flat = g.reshape(H * B, T, S)
  else:
    B, T, H, S = g.shape
    g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

  BH = B * H
  out_dtype = output_dtype or g.dtype
  HAS_SCALE = scale is not None
  scale_val = scale if scale is not None else 1.0

  interpret = get_interpret()

  # Pad S dimension to multiple of BS
  pad_S = (BS - (S % BS)) % BS
  if pad_S > 0:
    g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))
  S_padded = S + pad_S
  NS = S_padded // BS

  # Pad BH dimension to multiple of BB
  pad_BH = (BB - (BH % BB)) % BB
  if pad_BH > 0:
    g_flat = jnp.pad(g_flat, ((0, pad_BH), (0, 0), (0, 0)))
  BH_padded = BH + pad_BH

  NT = T // BT

  # Pad T so the last chunk of each sequence can read BT elements safely
  g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))
  T_alloc = T + BT

  # Dynamically shrink BB so that the compiler's double-buffered tiles
  # (4 × BB × T_alloc × BS × 4 bytes) fit within hardware VMEM.
  elem_bytes = 4  # float32
  while BB > 1 and 4 * BB * T_alloc * BS * elem_bytes > _VMEM_HW_LIMIT_BYTES:
    BB //= 2
  NBH = BH_padded // BB

  # If even with the shrunk BB the per-block T_alloc still exceeds VMEM,
  # tile T across a third grid axis. Each grid block processes NT_PER_BLOCK
  # consecutive chunks (T_BLOCK = NT_PER_BLOCK * BT). Requires segment
  # lengths to be BT-aligned.
  if 4 * BB * T_alloc * BS * elem_bytes > _VMEM_HW_LIMIT_BYTES:
    max_T_per_block = _VMEM_HW_LIMIT_BYTES // (4 * BB * BS * elem_bytes)
    NT_PER_BLOCK = max(1, max_T_per_block // BT)
    T_BLOCK = NT_PER_BLOCK * BT
  else:
    NT_PER_BLOCK = NT
    T_BLOCK = T_alloc

  NT_BLOCKS = (NT + NT_PER_BLOCK - 1) // NT_PER_BLOCK
  T_padded = NT_BLOCKS * T_BLOCK
  if T_padded > T_alloc:
    g_flat = jnp.pad(g_flat, ((0, 0), (0, T_padded - T_alloc), (0, 0)))

  grid = (NS, NBH, NT_BLOCKS)

  kernel = functools.partial(
    _chunk_cumsum_kernel_varlen,
    BT=BT, NT=NT,
    NT_PER_BLOCK=NT_PER_BLOCK,
    REVERSE=reverse, HAS_SCALE=HAS_SCALE, scale=scale_val,
  )

  def _index_map(i_s, i_bb, i_t_block, *_):
    return (i_bb, i_t_block, i_s)

  o_flat = pl.pallas_call(
    kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=grid,
      in_specs=[
        pl.BlockSpec(
          block_shape=(BB, T_BLOCK, BS),
          index_map=_index_map,
        ),
      ],
      out_specs=pl.BlockSpec(
        block_shape=(BB, T_BLOCK, BS),
        index_map=_index_map,
      ),
    ),
    out_shape=jax.ShapeDtypeStruct(g_flat.shape, out_dtype),
    interpret=interpret,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel"),
    ),
  )(g_flat)

  # Remove padding
  o_flat = o_flat[:BH, :T, :S]

  if head_first:
    return o_flat.reshape(H, B, T, S)
  else:
    return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# =============================================================================
# Public API
# =============================================================================

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
  """Chunk-local cumulative sum of gates with automatic backend dispatch.

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

  if head_first:
    return _chunk_local_cumsum_matmul(
      g,
      chunk_size,
      reverse,
      scale,
      head_first,
      output_dtype,
    )
  else:
    return _chunk_local_cumsum_origin(
      g,
      chunk_size,
      reverse,
      scale,
      head_first,
      output_dtype,
    )



# =============================================================================
# Chunk hidden-state kernels
# =============================================================================

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
    assert_shape,
    assert_shape_or_none,
    exp,
    export_public,
    get_interpret,
)


def _build_chunk_map(cu_seqlens, T_sum, BT):
    """Map each chunk to the sequence it belongs to.

    Chunks that fall beyond the last valid sequence (padding tail) are
    clamped to the last sequence so that downstream kernels never read
    ``cu_seqlens`` out of bounds.

    Args:
        cu_seqlens: [N+1] or [B, N+1] int array.
        T_sum: Total token count (per-batch when batched).
        BT: Chunk size.

    Returns:
        [NT] or [B, NT] int32 array mapping chunk index to sequence index.
    """
    if cu_seqlens.ndim == 2:
        B = cu_seqlens.shape[0]
        rows = [_build_chunk_map(cu_seqlens[b], T_sum, BT) for b in range(B)]
        return jnp.stack(rows, axis=0)
    NT = T_sum // BT
    chunk_ids = lax.iota(jnp.int32, NT)
    chunk_pos = chunk_ids * BT
    N = cu_seqlens.shape[-1] - 1
    seq_idx = jnp.searchsorted(cu_seqlens[1:], chunk_pos, side="right")
    seq_idx = jnp.clip(seq_idx, 0, N - 1)
    return seq_idx



def _chunk_fwd_h_kernel(
    k_ref,  # [1, 1, BT, BK]
    v_ref,  # [1, 1, BT, BV]
    h0_ref,  # [1, 1, BK, BV]
    gk_ref,  # [1, 1, BT, BK]
    g_ref,   # [1, 1, BT, 128]
    g_gamma_ref,  # [H]
    h_ref,  # [1, NS, 1, BK, BV] outputs
    ht_ref,  # [1, 1, BK , BV]
    scratch_ref, #[BK, BV]
    *,
    BT,
    BS,
    NT,
):

    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    NTS = BS // BT
    T = NT * BT
    i_b, i_h, i_k, i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2),pl.program_id(3),pl.program_id(4)

    if g_gamma_ref is not None:
        b_g = g_gamma_ref[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:,:] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:,:] = jnp.zeros((BK, BV), dtype=jnp.float32)

    @pl.when((i_t % NTS) == 0)
    def store_fn():
        i_s = i_t // NTS
        h_ref[0, i_s, 0] = scratch_ref[...].astype(h_ref.dtype)

    k_tile = k_ref[(0, 0, slice(None), slice(None))] # BT * BK
    v_tile = v_ref[(0, 0, slice(None), slice(None))] # BT * BV

    if g_ref is not None:
        b_g_scalar = g_ref[0, 0, slice(None), 0]  # [BT]
        b_g_scalar_last = b_g_scalar[BT - 1]       # scalar
        scratch_ref[...] *= exp(b_g_scalar_last)                 # uniform decay
        v_tile = (v_tile * exp(b_g_scalar_last - b_g_scalar)[:, None]).astype(v_tile.dtype)

    if g_gamma_ref is not None:
        # tpu not support scalar bf16 mul
        b_g_last = (g_gamma_ref[i_h].astype(jnp.float32) * jnp.minimum(BT, T - i_t * BT)).astype(g_gamma_ref.dtype)
        scratch_ref[...] *= exp(b_g_last)
        v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)


    if gk_ref is not None:
        gk_tile = gk_ref[(0, 0, slice(None), slice(None))] # BT * BK
        g_last = gk_tile[-1, :]
        decay = exp(g_last)
        scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
        k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
            k_tile.astype(jnp.float32).T,
            v_tile.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
    )

    @pl.when(i_t == NT - 1)
    def end():
        if ht_ref is not None:
            ht_ref[0, 0] = scratch_ref[...]


def check_chunk_fwd(x):
    assert x is None, "x should be None."


# note: The precision difference between this kernel on the TPU and FLA on the GPU is 5e-2.
@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
    ],
)
def chunk_fwd_h_kernel(
    k: jax.Array,
    v: jax.Array,  # [B,T,H,V]
    *,
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
):
    # todo: tune bk and bv for bast performance
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    BT = chunk_size
    BS = BT if split_size is None else split_size

    # =================== assert kernel requirements start ===================
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(gk, (B, T, H, K))
    # assert_shape_or_none(gv, (B, T, H, V))
    assert gv is None, "gv is currently not supported"
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu is currently not supported"
    assert cu_seqlens_dev is None, "cu_seqlens_dev is currently not supported"
    assert_shape_or_none(h0, (N, H, K, V))

    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."
    if cu_seqlens_cpu is not None:
        assert cu_seqlens_cpu[0] == 0, "cu_seqlens_cpu must start with 0."
        assert (cu_seqlens_cpu % chunk_size == 0).all(), "cu_seqlens_cpu must be multiples of chunk_size."

    assert BS % BT == 0, (
        f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    )
    # =================== assert kernel requirements done ===================

    # N: the actual number of sequences in the batch with either equal or variable lengths

    N, NS = (
        B,
        T // BS,
    )  # split_offsets[-1] # NS number of chunk_size
    NT = T // BT

    k = jnp.transpose(k, (0, 2, 1, 3))  # (B,H,T,K)
    v = jnp.transpose(v, (0, 2, 1, 3))  # (B,H,T,V)
    if gk is not None:
        gk = jnp.transpose(gk, (0, 2, 1, 3))  # (B,H,T,K)

    if g is not None:
        g = jnp.transpose(g, (0, 2, 1))  # (B, H, T)
        g = jnp.broadcast_to(g[:, :, :, None], (B, H, T, 128))  # (B, H, T, 128)

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    def k_index_map(batch_index, head_index, k_index, _, t_index):
        return batch_index, head_index, t_index, k_index

    def gk_index_map(batch_index, head_index,  k_index, _, t_index):
        return batch_index, head_index, t_index, k_index

    def g_index_map(batch_index, head_index,  k_index, _, t_index):
        return batch_index, head_index, t_index, 0

    def v_index_map(batch_index, head_index, _, v_index, t_index):
        return batch_index, head_index, t_index, v_index

    def h0_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, head_index, k_index, v_index

    def h_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, 0, head_index, k_index, v_index

    def ht_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, head_index, k_index, v_index


    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(N, NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((1, NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((1, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, 1, BT, BK), k_index_map),
        pl.BlockSpec((1, 1, BT, BV), v_index_map),
    ]
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((1, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, 1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)

    if g is not None:
        in_specs.append(pl.BlockSpec((1, 1, BT, 128), g_index_map))
    else:
        in_specs.append(None)

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    kernel = functools.partial(
        _chunk_fwd_h_kernel,
        BT=BT,
        BS=BS,
        NT=NT,
    )
    interpret = get_interpret()
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            # vmem_limit_bytes=32 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
    )(k, v, h0, gk, g, g_gamma)

    h = h.reshape(B, -1, H, K, V)
    ht = ht.reshape(N, H, K, V) if ht is not None else None

    if output_final_state:
        return h, ht
    return h, None


def _chunk_fwd_h_scan(k, v, g, g_gamma, gk, h0,
                      output_final_state, states_in_fp32,
                      C, B, T, H, K, V, NT):
    """lax.scan-based forward state propagation for fixed-length sequences.

    Replaces the Python for-loop in chunk_fwd_h_ref to avoid XLA
    trace-time loop unrolling (which creates a huge HLO graph).
    """
    h_dtype = jnp.float32 if states_in_fp32 else k.dtype
    has_g = g is not None
    has_gk = gk is not None

    # Reshape into chunks: [B, NT, C, H, D] then [NT, B, C, H, D] for scan
    k_scan = k.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    v_scan = v.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    scan_inputs = (k_scan, v_scan)
    if has_g:
        g_scan = g.reshape(B, NT, C, H).transpose(1, 0, 2, 3)
        scan_inputs += (g_scan,)
    if has_gk:
        gk_scan = gk.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
        scan_inputs += (gk_scan,)

    # Precompute g_gamma decay terms (constant across chunks)
    if g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        g_last_gamma = g_gamma_f32 * C  # [H]
        state_decay = jnp.exp(g_last_gamma)  # [H]
        b_g_gamma = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]  # [C, H]
        v_decay = jnp.exp(g_last_gamma[None, :] - b_g_gamma)  # [C, H]

    def scan_fn(h, chunk_data):
        # Unpack scan inputs (structure determined at trace time)
        idx = 0
        b_k = chunk_data[idx]; idx += 1
        b_v = chunk_data[idx]; idx += 1
        if has_g:
            b_g_scalar = chunk_data[idx]; idx += 1
        if has_gk:
            b_gk = chunk_data[idx]

        h_out = h  # state BEFORE update

        # Scalar gate g: [B, C, H]
        if has_g:
            b_g_last = b_g_scalar[:, -1, :]  # [B, H]
            h = h * jnp.exp(b_g_last.astype(jnp.float32))[:, :, None, None]
            b_v = (b_v * jnp.exp((b_g_last[:, None, :] - b_g_scalar).astype(jnp.float32))[:, :, :, None]).astype(b_v.dtype)

        # Per-head fixed decay g_gamma
        if g_gamma is not None:
            h = h * state_decay[None, :, None, None]
            b_v = (b_v * v_decay[None, :, :, None]).astype(b_v.dtype)

        # Per-K-dim gate gk: [B, C, H, K]
        if has_gk:
            b_gk_last = b_gk[:, -1, :, :]  # [B, H, K]
            h = h * jnp.exp(b_gk_last.astype(jnp.float32))[:, :, :, None]
            b_k = b_k * jnp.exp((b_gk_last[:, None, :, :] - b_gk).astype(jnp.float32))

        # State update: h += k^T @ v  (contract C, batch B and H)
        kv = lax.dot_general(
            b_k,
            b_v,
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        h = h + kv
        return h, h_out

    # Initial state: [B, H, K, V]  (N = B for fixed-length)
    h_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if h0 is not None:
        h_init = h0.reshape(B, H, K, V).astype(jnp.float32)

    h_final, h_all = lax.scan(scan_fn, h_init, scan_inputs)
    # h_all: [NT, B, H, K, V] -> [B, NT, H, K, V]
    h_all = h_all.transpose(1, 0, 2, 3, 4).astype(h_dtype)

    ht = None
    if output_final_state:
        ht = h_final.astype(jnp.float32)  # [B, H, K, V]

    return h_all, ht


def chunk_fwd_h_ref(
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    h0: jax.Array | None = None,
    output_final_state: bool = False,
    states_in_fp32: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Inter-chunk hidden state propagation.

    Computes the hidden state at the start of each chunk by
    sequentially propagating through chunks.

    Args:
        k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
        v:  [B, T, H, V] — values
        g:  [B, T, H] — chunk-local cumsum of scalar gate (optional)
        g_gamma: [H] — per-head fixed decay rate (optional)
        gk: [B, T, H, K] — chunk-local cumsum of K-dim gates (optional)
        gv: [B, T, H, V] — V-dim gate (optional, currently unused)
        h0: [N, H, K, V] — initial hidden state (optional)
        output_final_state: whether to return final state
        states_in_fp32: if True, store h_all in float32 instead of k.dtype
        cu_seqlens_dev: cumulative sequence lengths (optional)
        cu_seqlens_cpu: alias for cu_seqlens (backward compat)
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or None — final hidden state
    """

    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_dev is None else cu_seqlens_dev.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_fwd_h"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
    )

    # Fast path: fixed-length sequences use lax.scan (avoids XLA loop unrolling)
    if cu_seqlens_cpu is None and gv is None:
        return _chunk_fwd_h_scan(
            k, v, g, g_gamma, gk, h0,
            output_final_state, states_in_fp32,
            C, B, T, H, K, V, NT,
        )

    k = k.reshape(-1, H, K)
    v = v.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None
    g = g.reshape(-1, H) if g is not None else None
    h0 = h0.reshape(-1, H, K, V) if h0 is not None else None

    h_dtype = jnp.float32 if states_in_fp32 else k.dtype
    ht = jnp.zeros([N, H, K, V], dtype=jnp.float32)
    h_all = jnp.zeros([B, NT, H, K, V], dtype=h_dtype)
    for i_n in range(N):
        if cu_seqlens_cpu is None:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        h = jnp.zeros((H, K, V), dtype=jnp.float32)
        if h0 is not None:
            h = h + h0[i_n].astype(jnp.float32)

        if g_gamma is not None:
            g_gamma_f32 = g_gamma.astype(jnp.float32)
            b_g = g_gamma_f32[None, :] * (jnp.arange(0, C) + 1)[:, None]  # [C, H] float32

        NT_seq = (eos - bos) // C
        for i_t in range(NT_seq):
            if cu_seqlens_cpu is None:
                h_all = h_all.at[i_n, i_t].set(h.astype(h_all.dtype))
            else:
                h_all = h_all.at[0, bos // C + i_t].set(h.astype(h_all.dtype))
            b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]

            if g is not None:
                b_g_scalar = g[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H]
                b_g_last = b_g_scalar[-1]  # [H]
                h *= exp(b_g_last)[:, None, None]  # (H, K, V)
                b_v = (b_v * exp(b_g_last[None, :] - b_g_scalar)[:, :, None]).astype(
                    b_v.dtype
                )

            if g_gamma is not None:
                b_g_last = g_gamma_f32 * jnp.minimum(C, (eos - bos) - i_t * C)  # [H] float32
                h *= exp(b_g_last[:, None, None])  # (H, K, V)
                b_v = (b_v * exp(b_g_last[None, :] - b_g)[:, :, None]).astype(
                    b_v.dtype
                )

            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                h *= exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

                b_k = b_k * exp(
                    b_gk_last[None, :, :] - b_gk
                )  # b_gk_last -> [C, H, K]

            h = h + lax.dot_general(
                b_k,
                b_v,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        if output_final_state:
            ht = ht.at[i_n].set(h.astype(ht.dtype))
    if output_final_state:
        return h_all, ht
    else:
        return h_all, None


def _chunk_bwd_dh_scan(q, do, g, g_gamma, gk, dht,
                       scale, output_dh0, states_in_fp32,
                       C, B, T, H, K, V, NT):
    """lax.scan-based backward state gradient propagation for fixed-length sequences.

    Mirrors _chunk_fwd_h_scan: replaces the Python for-loop in chunk_bwd_dh_ref
    to avoid XLA trace-time loop unrolling (which creates a huge HLO graph).
    """
    has_g = g is not None
    has_gk = gk is not None

    # Reshape into chunks: [B, NT, C, H, D] then [NT, B, C, H, D] for scan
    q_scan = q.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    do_scan = do.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    scan_inputs = (q_scan, do_scan)
    if has_g:
        g_scan = g.reshape(B, NT, C, H).transpose(1, 0, 2, 3)
        scan_inputs += (g_scan,)
    if has_gk:
        gk_scan = gk.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
        scan_inputs += (gk_scan,)

    # Precompute g_gamma decay terms (constant across chunks)
    if g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        g_last_gamma = g_gamma_f32 * C  # [H]
        state_decay = jnp.exp(g_last_gamma)  # [H]
        b_g_ramp = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]  # [C, H]

    def scan_fn(dh, chunk_data):
        # Unpack scan inputs (structure determined at trace time)
        idx = 0
        b_q = chunk_data[idx]; idx += 1
        b_do = chunk_data[idx]; idx += 1
        if has_g:
            b_g_scalar = chunk_data[idx]; idx += 1
        if has_gk:
            b_gk = chunk_data[idx]

        dh_out = dh  # state BEFORE update (stored at this chunk boundary)

        # Per-K-dim gate gk: [B, C, H, K]
        if has_gk:
            b_gk_last = b_gk[:, -1, :, :]  # [B, H, K]
            dh = dh * jnp.exp(b_gk_last.astype(jnp.float32))[:, :, :, None]
            b_q_hat = b_q * jnp.exp(b_gk.astype(jnp.float32)) * scale
        elif has_g:
            b_g_last = b_g_scalar[:, -1, :]  # [B, H]
            dh = dh * jnp.exp(b_g_last.astype(jnp.float32))[:, :, None, None]
            b_q_hat = (b_q * jnp.exp(b_g_scalar.astype(jnp.float32))[:, :, :, None] * scale)
        elif g_gamma is not None:
            dh = dh * state_decay[None, :, None, None]
            b_q_hat = (b_q * jnp.exp(b_g_ramp)[None, :, :, None] * scale)
        else:
            b_q_hat = b_q * scale

        # Accumulate: dh += q_hat^T @ do  (contract C, batch B and H)
        dh = dh + lax.dot_general(
            b_q_hat,
            b_do,
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        return dh, dh_out

    # Initial state: [B, H, K, V]
    dh_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if dht is not None:
        dh_init = dht.reshape(B, H, K, V).astype(jnp.float32)

    dh_final, dh_all = lax.scan(scan_fn, dh_init, scan_inputs, reverse=True)
    # dh_all: [NT, B, H, K, V] -> [B, NT, H, K, V]
    dh_all = dh_all.transpose(1, 0, 2, 3, 4)

    dh0 = None
    if output_dh0:
        dh0 = dh_final.astype(jnp.float32)  # [B, H, K, V]

    return dh_all, dh0


def chunk_bwd_dh_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    g_gamma: jax.Array,
    gk: jax.Array,
    do: jax.Array,
    h0: jax.Array | None = None,
    dht: jax.Array | None = None,
    scale: float = 1.0,
    output_dh0: bool = False,
    states_in_fp32: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Backward hidden state gradient propagation.

    Propagates gradients backward through chunks to compute dh at each
    chunk boundary and dh0.

    Args:
        q:   [B, T, H, K] — queries
        k:   [B, T, H, K] — keys
        v:   [B, T, H, V] — values
        gk:  [B, T, H, K] — chunk-local cumsum of gates
        do:  [B, T, H, V] — output gradient
        h0:  [N, H, K, V] — initial hidden state (optional)
        dht: [N, H, K, V] — terminal state gradient (optional)
        scale: scaling factor
        cu_seqlens_cpu: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V] — gradient at start of each chunk
        dh0: [N, H, K, V] or None — initial state gradient
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_bwd_dh"

    # Fast path: fixed-length sequences use lax.scan (avoids XLA loop unrolling)
    if cu_seqlens_cpu is None:
        return _chunk_bwd_dh_scan(
            q, do, g, g_gamma, gk, dht,
            scale, output_dh0, states_in_fp32,
            C, B, T, H, K, V, NT,
        )

    is_varlen = cu_seqlens_cpu is not None

    q = q.reshape(-1, H, K)
    do = do.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None

    dh_all = jnp.zeros([B, NT, H, K, V], dtype=jnp.float32)
    dh0_all = (
        jnp.zeros([N, H, K, V], dtype=jnp.float32)
        if (h0 is not None or dht is not None)
        else None
    )

    for i_n in range(N):
        if not is_varlen:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        NT_seq = (eos - bos) // C
        dh = jnp.zeros((H, K, V), dtype=jnp.float32)
        if dht is not None:
            dh = dh + dht[i_n].astype(jnp.float32)

        for i_t in range(NT_seq - 1, -1, -1):
            bi = 0 if is_varlen else i_n
            ti = bos // C + i_t if is_varlen else i_t
            dh_all = dh_all.at[bi, ti].set(dh)

            b_q = q[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_do = do[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]

            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                b_q_hat = b_q * exp(b_gk) * scale  # [C, H, K]
                dh = dh * exp(b_gk_last[:, :, None])
            else:
                b_q_hat = b_q * scale

            # contract over C (dim 0) and H (dim 1): [C,H,K]^T @ [C,H,V] -> [H,K,V]
            dh = dh + lax.dot_general(
                b_q_hat,
                b_do,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )

        if dh0_all is not None:
            dh0_all = dh0_all.at[i_n].set(dh)

    return dh_all, dh0_all


def _chunk_bwd_dh_kernel(
    q_ref,          # [1, 1, BT, BK]
    do_ref,         # [1, 1, BT, BV]
    dht_ref,        # [N, 1, BK, BV]
    gk_ref,         # [1, 1, BT, BK]
    g_ref,          # [1, 1, BT]
    g_gamma,        # [H]
    cu_seqlens_ref, # [num_seq + 1]
    chunk_to_seq,   # [NT]
    dh_ref,         # [1, 1, BK, BV]
    dh0_ref,        # [N, 1, BK, BV] or None
    carry_ref,      # scratch VMEM (BK, BV)
    *,
    BT: int,
    NT: int,
    scale: float,
):
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]

    i_c = pl.program_id(3)  # chunk index (0 = last chunk in time)
    i_t = NT - 1 - i_c      # global chunk index (forward order)
    t0 = i_t * BT            # global time offset

    # Load carry from previous step, or init zeros for first step
    b_dh = lax.cond(
        i_c == 0,
        lambda _: jnp.zeros((BK, BV), dtype=jnp.float32),
        lambda _: carry_ref[...].astype(jnp.float32),
        operand=None,
    )

    if g_gamma is not None:
        head_index = pl.program_id(0)
        # tpu not support scalar bf16 mul
        b_g_ramp = (g_gamma[head_index].astype(jnp.float32) * (jnp.arange(0, BT) + 1)).astype(g_gamma.dtype)  # [BT]

    seq_idx = chunk_to_seq[i_t]
    eos = cu_seqlens_ref[seq_idx + 1]

    # reset dh at sequence boundary (last chunk of each sequence)
    is_last_chunk = (t0 + BT >= eos)

    def reset_state(_):
        if dht_ref is not None:
            return dht_ref[seq_idx, 0].astype(jnp.float32)
        return jnp.zeros((BK, BV), dtype=jnp.float32)

    b_dh = lax.cond(is_last_chunk, reset_state, lambda _: b_dh, operand=None)

    # store dh (after reset, before compute)
    dh_ref[0, 0] = b_dh.astype(dh_ref.dtype)

    b_q = q_ref[(0, 0, slice(None), slice(None))]    # [BT, BK]
    b_do = do_ref[(0, 0, slice(None), slice(None))]   # [BT, BV]
    b_q = (b_q * scale).astype(b_q.dtype)

    # scalar gate (g)
    if g_ref is not None:
        b_g_scalar = g_ref[0, 0, slice(None)]  # [BT]
        b_g_scalar_last = b_g_scalar[BT - 1]
        b_dh *= exp(b_g_scalar_last)
        b_q = (b_q * exp(b_g_scalar)[:, None]).astype(b_q.dtype)

    # per-head fixed decay (g_gamma)
    if g_gamma is not None:
        # tpu not support scalar bf16 mul
        b_g_last = (g_gamma[head_index].astype(jnp.float32) * jnp.minimum(BT, eos - t0)).astype(g_gamma.dtype)
        b_dh *= exp(b_g_last)
        b_q = (b_q * exp(b_g_ramp)[:, None]).astype(b_q.dtype)

    # per-K-dim gate (gk)
    if gk_ref is not None:
        b_gk = gk_ref[(0, 0, slice(None), slice(None))]  # [BT, BK]
        g_last = b_gk[BT - 1, :]
        b_dh = b_dh * exp(g_last)[:, None]  # [BK, BV] * [BK, 1]
        b_q = (b_q * exp(b_gk)).astype(b_q.dtype)

    b_dh = b_dh + jax.lax.dot(
        b_q.astype(jnp.float32).T, b_do.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # write dh0 at sequence start
    bos = cu_seqlens_ref[seq_idx]

    @pl.when(t0 == bos)
    def _():
        if dh0_ref is not None:
            dh0_ref[seq_idx, 0] = b_dh.astype(dh0_ref.dtype)

    # Save carry for next step
    carry_ref[...] = b_dh.astype(jnp.float32)


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "output_dh0",
        "chunk_size",
        "states_in_fp32",
    ],
)
def chunk_bwd_dh_kernel(
    q: jax.Array,                # [B, T, H, K]
    k: jax.Array,                # [B, T, H, K] (unused but kept for API compatibility)
    v: jax.Array,                # [B, T, H, V] (unused but kept for API compatibility)
    g: jax.Array | None = None,  # [B, T, H]
    g_gamma: jax.Array | None = None,  # [H]
    gk: jax.Array | None = None, # [B, T, H, K]
    do: jax.Array = None,        # [B, T, H, V]
    dht: jax.Array | None = None,# [N, H, K, V]
    scale: float = 1.0,
    output_dh0: bool = False,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 128,
    states_in_fp32: bool = False,
):
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = do.shape[-1]
    T_sum = B * T
    NT = T_sum // BT

    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    if cu_seqlens_dev is None:
        cu_seqlens_dev = jnp.arange(T_sum + 1, step=T)
    chunk_to_seq = _build_chunk_map(cu_seqlens=cu_seqlens_dev, T_sum=T_sum, BT=BT)
    N = len(cu_seqlens_dev) - 1

    # Reshape to (H, NT, BT, X) — one chunk per grid step
    q = jnp.reshape(q, (T_sum, H, K)).transpose(1, 0, 2).reshape(H, NT, BT, K)
    do = jnp.reshape(do, (T_sum, H, V)).transpose(1, 0, 2).reshape(H, NT, BT, V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K)).transpose(1, 0, 2).reshape(H, NT, BT, K)
    if g is not None:
        g = jnp.reshape(g, (T_sum, H)).transpose(1, 0).reshape(H, NT, BT)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # Reversed chunk order: c=0 → last chunk (processed first in backward)
    def idx_map_K(h, k, v, c): return h, NT - 1 - c, 0, k
    def idx_map_V(h, k, v, c): return h, NT - 1 - c, 0, v
    def idx_map_state(h, k, v, c): return 0, h, k, v

    dtype_out = q.dtype if not states_in_fp32 else jnp.float32

    out_shape = [
        jax.ShapeDtypeStruct(shape=(NT, H, K, V), dtype=dtype_out),
    ]
    out_specs = [
        pl.BlockSpec((1, 1, BK, BV), lambda h, k, v, c: (NT - 1 - c, h, k, v)),
    ]
    if output_dh0:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=dtype_out))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), idx_map_state))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, 1, BT, BK), idx_map_K),   # q
        pl.BlockSpec((1, 1, BT, BV), idx_map_V),    # do
        pl.BlockSpec((N, 1, BK, BV), idx_map_state) if dht is not None else None,  # dht
        pl.BlockSpec((1, 1, BT, BK), idx_map_K) if gk is not None else None,  # gk
        pl.BlockSpec((1, 1, BT), lambda h, k, v, c: (h, NT - 1 - c, 0)) if g is not None else None,  # g
        pl.BlockSpec(memory_space=pltpu.SMEM) if g_gamma is not None else None,  # g_gamma
        pl.BlockSpec(memory_space=pltpu.SMEM),      # cu_seqlens
        pl.BlockSpec(memory_space=pltpu.SMEM),      # chunk_to_seq
    ]

    kernel = functools.partial(_chunk_bwd_dh_kernel, BT=BT, NT=NT, scale=scale)

    interpret = get_interpret()
    dh_all, dh0 = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=32 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
    )(q, do, dht, gk, g, g_gamma, cu_seqlens_dev, chunk_to_seq)

    dh_all = dh_all.reshape(B, -1, H, K, V)
    dh0 = dh0.reshape(N, H, K, V) if dh0 is not None else None
    return dh_all, dh0


def _chunk_fwd_h_kernel_varlen(
    k_ref,  # [1, BT, BK]
    v_ref,  # [1, BT, BV]
    h0_ref,  # [N, 1, BK, BV]
    gk_ref,  # [1, BT, BK]
    g_gamma_ref, # [H,]
    cu_seqlens_ref,  # [num_seq+1]
    chunk_to_seq,  # [T_sum/BT]
    seq_real_lens_ref,  # [N] real (non-padded) length per sequence, or None
    h_ref,  # [NS, 1, BK, BV]
    ht_ref,  # [N, 1, BK , BV]
    scratch_ref, # [BK, BV]
    *,
    BT,
    BS,
):
    BT, BK = k_ref.shape[1], k_ref.shape[2]
    BV = v_ref.shape[2]

    NTS = BS // BT
    b_h_start = jnp.zeros((BK, BV), dtype=jnp.float32)

    i_h, i_k, i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)

    if g_gamma_ref is not None:
        b_g = g_gamma_ref[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    t0 = i_t * BT

    seq_idx = chunk_to_seq[i_t]

    bos = cu_seqlens_ref[seq_idx]
    eos = cu_seqlens_ref[seq_idx + 1]
    @pl.when(bos != eos)
    def _():
        # reset h state
        @pl.when(t0 == bos)
        def reset_state():
            if h0_ref is not None:
                scratch_ref[...] =  h0_ref[seq_idx, 0].astype(scratch_ref.dtype)
            else:
                scratch_ref[...] =  b_h_start

        # store intermediate state
        @pl.when(i_t % NTS == 0)
        def store_fn():
            s_i = i_t // NTS
            h_ref[s_i, 0] = scratch_ref[...].astype(h_ref.dtype)
            return None

        k_tile = k_ref[(0, slice(None), slice(None))]  # [BT,BK]
        v_tile = v_ref[(0, slice(None), slice(None))]  # [BT,BV]

        if g_gamma_ref is not None:
            # Use real sequence length for b_g_last when seq_real_lens is
            # available.  This is critical when sequences are padded to
            # chunk-aligned boundaries: the decay must only cover the
            # *real* tokens, not the padding.
            if seq_real_lens_ref is not None:
                real_eos = bos + seq_real_lens_ref[seq_idx]
                effective_remaining = jnp.maximum(real_eos - t0, 0)
            else:
                effective_remaining = eos - t0
            # tpu not support scalar bf16 mul
            L_chunk = jnp.minimum(BT, effective_remaining)
            b_g_last = (g_gamma_ref[i_h].astype(jnp.float32) * L_chunk).astype(g_gamma_ref.dtype)
            scratch_ref[...] *= exp(b_g_last)
            # Mask exponent to avoid NaN (0 * inf) in padding positions
            v_decay_exp = b_g_last - b_g
            v_decay_exp = jnp.where(jnp.arange(BT) < L_chunk, v_decay_exp, -1e9)
            v_tile = (v_tile * exp(v_decay_exp)[:, None]).astype(v_tile.dtype)

        if gk_ref is not None:
            gk_tile = gk_ref[(0, slice(None), slice(None))] # BT * BK
            g_last = gk_tile[-1, :]
            decay = exp(g_last)
            scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
            k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

        # state update
        scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
                k_tile.astype(jnp.float32).T,
                v_tile.astype(jnp.float32),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
        )

        @pl.when(t0 + BT >= eos)
        def write_final():
            if ht_ref is not None:
                ht_ref[seq_idx, 0] = scratch_ref[...].astype(jnp.float32)


def check_chunk_fwd(x):
    assert x is None, "x should be None."


# note: The precision difference between this kernel on the TPU and FLA on the GPU is 5e-2.
@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
    ],
)
def chunk_fwd_h_kernel_varlen(
    k: jax.Array,  # [B,T,H,K]
    v: jax.Array,  # [B,T,H,V]
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 128,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    seq_real_lens: jax.Array | None = None,  # [N] real (non-padded) seq lengths
):
    interpret = get_interpret()
    check_chunk_fwd(g)
    check_chunk_fwd(gv)
    # todo: tune bk and bv for bast performance
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, (
        f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths

    T_sum = B * T
    chunk_to_seq = _build_chunk_map(cu_seqlens=cu_seqlens_dev, T_sum=T_sum, BT=BT)

    N, NS = (
        len(cu_seqlens_dev) - 1,
        T_sum // BS,
    )  # split_offsets[-1] # NS number of chunk_size

    k = jnp.reshape(k, (T_sum, H, K))
    v = jnp.reshape(v, (T_sum, H, V))

    k = jnp.transpose(k, (1, 0, 2))  # (H,B*T,K)
    v = jnp.transpose(v, (1, 0, 2))  # (H,B*T,V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K))
        gk = jnp.transpose(gk, (1, 0, 2))  # (H,B*T,K)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV), T_sum//BT)

    def k_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def gk_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def v_index_map(head_index, _, v_index, t_index):
        return head_index, t_index, v_index

    def h0_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def ht_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def h_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, BT, BK), k_index_map),
        pl.BlockSpec((1, BT, BV), v_index_map),
    ]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((N, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)
    
    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    # seq_real_lens: [N] real (non-padded) sequence lengths for b_g_last
    if seq_real_lens is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    kernel = functools.partial(
        _chunk_fwd_h_kernel_varlen,
        BT=BT,
        BS=BS,
    )
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes = scratch_shapes,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
    )(k, v, h0, gk, g_gamma, cu_seqlens_dev, chunk_to_seq, seq_real_lens)
    if output_final_state:
        return h, ht
    return h, None



# =============================================================================
# Delta-rule hidden-state kernels
# =============================================================================

# tops/ops/common/chunk_delta_h.py
"""Delta-rule inter-chunk hidden state propagation (Pallas TPU kernel + JAX reference).

Shared by KDA and Gated Delta Rule. Implements the delta-rule recurrence
where the value is corrected by subtracting the state's prediction before
accumulating into the hidden state:

    v_new_t = v_t - w_t @ h_{t-1}    (delta correction)
    h_t = h_{t-1} * decay_t + k_t^T @ v_new_t  (state update)

This differs from standard GLA (common/chunk_h.py) which uses:
    h_t = h_{t-1} * decay_t + k_t^T @ v_t       (no delta correction)

Gate types supported:
  - g:  [B, T, H] scalar per-head gate (applied as exp2(g) to state, and
        exp2(g_last - g) to v_new for accumulation)
  - gk: [B, T, H, K] per-element gate (applied as exp2(gk) to state only;
        k is assumed to be already gated, so no additional k gating is done)

Both gates operate in log2 space: decay = exp2(g) or exp2(gk).
"""


import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
  align_up,
  assert_shape,
  assert_shape_or_none,
  cdiv,
  export_public,
  get_interpret,
  pad_to_multiple,
  prepare_chunk_indices,
)
from tokamax._src.ops.experimental.kda.utils import exp, exp2


def chunk_gated_delta_rule_fwd_h_ref(
  k: jax.Array,
  w: jax.Array,
  v: jax.Array,
  g: jax.Array | None = None,
  gk: jax.Array | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
  """Pure JAX reference for delta-rule inter-chunk state propagation.

  For each chunk c (with BT positions), stores the state BEFORE processing
  chunk c, then updates the state using all BT positions in chunk c.

  The algorithm for each chunk:
    1. Store h[c] = S (state before chunk)
    2. v_new = v - w @ S (delta correction using pre-decay state)
    3. Decay S:
       - If g (scalar): S *= exp2(g_last)
       - If gk (per-element): S *= exp2(gk_last[:, None])
    4. Gate v_new for accumulation (scalar g only):
       - If g: v_new *= exp2(g_last - g)[:, None]
       - If gk: no v_new gating (k is already position-gated)
    5. State update: S += k^T @ v_new

  IMPORTANT: k is expected to be already gated (e.g., kg from intra-chunk
  step). The gk gate is used ONLY for state decay, NOT for additionally
  gating k. This matches the Pallas kernel behavior.

  Args:
      k: [B, T, H, K] — Keys (gated, ready for outer product).
      w: [B, T, H, K] — Correction weights (w = beta * k * exp(g)).
      v: [B, T, H, V] — Values (after intra-chunk delta solve: u).
      g: [B, T, H] — Scalar per-head gate in log2 space. Optional.
      gk: [B, T, H, K] — Per-element gate in log2 space. Optional.
      initial_state: [B, H, K, V] — Initial hidden state. Optional.
      output_final_state: Whether to return the final hidden state.
      chunk_size: Chunk size. T must be divisible by chunk_size.

  Returns:
      h: [B, NT, H, K, V] — Hidden states (state before each chunk).
      v_new: [B, T, H, V] — Delta-corrected values (v - w @ h).
      final_state: [B, H, K, V] if output_final_state, else None.
  """
  B, T, H, K = k.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  assert_shape(k, (B, T, H, K), "k")
  assert_shape(w, (B, T, H, K), "w")
  assert_shape(v, (B, T, H, V), "v")
  assert_shape_or_none(g, (B, T, H), "g")
  assert_shape_or_none(gk, (B, T, H, K), "gk")
  assert_shape_or_none(initial_state, (B, H, K, V), "initial_state")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

  # Reshape to chunks: [B, T, H, D] → [B, H, NT, BT, D]
  k_c = jnp.transpose(k, (0, 2, 1, 3)).reshape(B, H, NT, BT, K).astype(jnp.float32)
  w_c = jnp.transpose(w, (0, 2, 1, 3)).reshape(B, H, NT, BT, K).astype(jnp.float32)
  v_c = jnp.transpose(v, (0, 2, 1, 3)).reshape(B, H, NT, BT, V).astype(jnp.float32)

  if g is not None:
    g_c = jnp.transpose(g, (0, 2, 1)).reshape(B, H, NT, BT).astype(jnp.float32)
  if gk is not None:
    gk_c = jnp.transpose(gk, (0, 2, 1, 3)).reshape(B, H, NT, BT, K).astype(jnp.float32)

  # State: [B, H, K, V]
  S = jnp.zeros((B, H, K, V), dtype=jnp.float32)
  if initial_state is not None:
    S = S + initial_state.astype(jnp.float32)

  h_all = jnp.zeros((B, H, NT, K, V), dtype=jnp.float32)
  v_new_all = jnp.zeros((B, H, NT, BT, V), dtype=jnp.float32)

  for i in range(NT):
    # 1. Store state BEFORE processing this chunk
    h_all = h_all.at[:, :, i].set(S)

    # 2. Delta correction: v_new = v - w @ S
    w_i = w_c[:, :, i]  # [B, H, BT, K]
    v_i = v_c[:, :, i]  # [B, H, BT, V]
    v_new_i = v_i - jnp.einsum("bhck,bhkv->bhcv", w_i, S)
    v_new_all = v_new_all.at[:, :, i].set(v_new_i)

    # 3. Decay state by gate
    if g is not None:
      g_last = g_c[:, :, i, -1]  # [B, H] — scalar gate at last position
      S = S * jnp.exp2(g_last)[:, :, None, None]

    if gk is not None:
      gk_last = gk_c[:, :, i, -1]  # [B, H, K] — per-element gate at last position
      S = S * jnp.exp2(gk_last)[:, :, :, None]

    # 4. Gate v_new for accumulation (scalar g only)
    # When g (scalar per-head): distribute gate into v_new for k^T @ v_new
    # When gk (per-element): k is already gated, no v_new gating needed
    if g is not None:
      g_chunk = g_c[:, :, i]  # [B, H, BT]
      g_last_val = g_c[:, :, i, -1:]  # [B, H, 1]
      v_new_i = v_new_i * jnp.exp2(g_last_val - g_chunk)[:, :, :, None]

    # 5. State update: S += k^T @ v_new
    # k is already gated (kg from intra-chunk), used directly
    k_i = k_c[:, :, i]  # [B, H, BT, K]
    S = S + jnp.einsum("bhck,bhcv->bhkv", k_i, v_new_i)

  final_state = S if output_final_state else None

  # Reshape: [B, H, NT, K, V] → [B, NT, H, K, V]
  h_all = jnp.transpose(h_all, (0, 2, 1, 3, 4))
  # Reshape v_new: [B, H, NT, BT, V] → [B, T, H, V]
  v_new_all = v_new_all.reshape(B, H, T, V)
  v_new_all = jnp.transpose(v_new_all, (0, 2, 1, 3))

  return h_all, v_new_all, final_state


# ---------------------------------------------------------------------------
# Pallas TPU kernels for delta-rule inter-chunk state propagation
# ---------------------------------------------------------------------------


def _prepare_chunk_offsets(seqlens: jax.Array, chunk_size: int) -> jax.Array:
  """Compute cumulative chunk-count offsets for variable-length sequences.

  Given cumulative sequence lengths *seqlens* ``[0, s1, s2, ...]`` and a
  fixed *chunk_size*, returns a prefix-sum array ``[0, NT_0, NT_0+NT_1, ...]``
  where ``NT_i = ceil(len_i / chunk_size)``.  Used by the varlen kernel to
  map ``(sequence_id, chunk_id)`` to a flat chunk index in the output tensor.
  """
  return jnp.pad(
    cdiv(jnp.diff(seqlens), chunk_size).astype(jnp.int32),
    (1, 0),
    constant_values=0,
  ).cumsum(-1)


# ── Varlen Pallas kernel ────────────────────────────────────────────────────


def _chunk_gated_delta_rule_fwd_varlen_kernel(
  seqlens_ref,
  chunk_to_seq_ref,
  k_ref,
  v_ref,
  w_ref,
  g_ref,
  gk_ref,
  h0_ref,
  h_ref,
  v_new_ref,
  ht_ref,
  scratch_ref,
  *,
  NT,
  USE_G,
  USE_GK,
  USE_INITIAL_STATE,
  STORE_FINAL_STATE,
  SAVE_NEW_VALUE,
  USE_EXP2,
  MINI_BATCH: int = 1,
):
  """Delta-rule inter-chunk state forward pass kernel for varlen.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, i_c):
    - Processes MB heads per grid point via batch-vectorized ops.
    - i_b is the batch index, i_c is the chunk index within that batch.
    - seq_idx = chunk_to_seq[i_b, i_c] identifies which sequence this chunk
      belongs to within batch i_b.
    - At t0 == bos: init scratch (h0 or zeros) for this sequence.
    - At t0 + BT >= eos: store final state for this sequence.

  Args:
      MINI_BATCH: int — number of heads processed per grid point (MB).
  """
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)
  seq_idx = chunk_to_seq_ref[i_b, i_c]

  bos = seqlens_ref[i_b, seq_idx]
  eos = seqlens_ref[i_b, seq_idx + 1]

  BT = k_ref.shape[2]
  t0 = i_c * BT
  K, V = k_ref.shape[-1], v_ref.shape[-1]

  @pl.when(t0 == bos)
  def _():
    if USE_INITIAL_STATE:
      scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]
    else:
      scratch_ref[:] = jnp.zeros_like(scratch_ref[:])

  # h output shape: [MB, 1, 1, K, V] — MB is the H dimension
  h_ref[:, 0, 0] = scratch_ref[:].astype(h_ref.dtype)  # [MB, K, V]

  b_k = k_ref[:,0,:].astype(jnp.float32)    # [MB, BT, K]
  b_w = w_ref[:,0,:].astype(jnp.float32)    # [MB, BT, K]
  b_u = v_ref[:,0,:].astype(jnp.float32)    # [MB, BT, V]

  # v_new = v - w @ state: [MB, BT, K] @ [MB, K, V] → [MB, BT, V]
  b_v = b_u - jnp.matmul(
    b_w,
    scratch_ref[:],
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  if SAVE_NEW_VALUE:
    v_new_ref[:,0,:] = b_v.astype(v_new_ref.dtype)

  if USE_G:
    b_g = g_ref[:, 0, :, 0].astype(jnp.float32)  # [MB, BT]
    b_g_last = b_g[:, BT - 1]                      # [MB]
    if USE_EXP2:
      b_v = b_v * exp2(b_g_last[:, None] - b_g)[:, :, None]
      scratch_ref[:] = scratch_ref[:] * exp2(b_g_last)[:, None, None]
    else:
      b_v = b_v * exp(b_g_last[:, None] - b_g)[:, :, None]
      scratch_ref[:] = scratch_ref[:] * exp(b_g_last)[:, None, None]

  if USE_GK:
    b_gk_last = gk_ref[:, 0, BT - 1].astype(jnp.float32)  # [MB, K]
    if USE_EXP2:
      scratch_ref[:] = scratch_ref[:] * exp2(b_gk_last)[:, :, None]
    else:
      scratch_ref[:] = scratch_ref[:] * exp(b_gk_last)[:, :, None]

  # state += k^T @ v: [MB, K, BT] @ [MB, BT, V] → [MB, K, V]
  scratch_ref[:] = scratch_ref[:] + jnp.matmul(
    b_k.transpose(0, 2, 1),
    b_v,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  @pl.when(t0 + BT >= eos)
  def _():
    if STORE_FINAL_STATE:
      ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "chunk_size",
    "BV",
    "save_new_value",
    "use_exp2",
    "mini_batch",
  ],
)
def _chunk_gated_delta_rule_fwd_varlen(
  k: jax.Array,
  w: jax.Array,
  v: jax.Array,
  seqlens: jax.Array,
  chunk_indices: jax.Array,
  g: jax.Array | None = None,
  gk: jax.Array | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
  BV: int = 64,
  save_new_value: bool = True,
  use_exp2: bool = False,
  mini_batch: int | None = None,
):
  """Varlen launcher for delta-rule inter-chunk state forward pass.

  Pads and reshapes inputs for the TPU kernel, dispatches
  ``_chunk_gated_delta_rule_fwd_varlen_kernel``, then un-pads outputs.

  Args:
      k: [H, B, T, K] -- Keys.
      w: [H, B, T, K] -- Correction weights.
      v: [H, B, T, V] -- Delta-corrected values from intra-chunk.
      seqlens: [B, N+1] -- Cumulative sequence lengths (per-batch).
      chunk_indices: [B, NT, 2] -- Precomputed chunk indices (per-batch).
      g: [B, T, H] -- Scalar per-head gate (optional).
      gk: [B, T, H, K] -- Per-element gate (optional).
      initial_state: [B, N, H, K, V] -- Initial hidden state (optional).
      output_final_state: Whether to return final hidden state.
      chunk_size: Chunk size.
      BV: V-dimension block size.
      save_new_value: Whether to compute and return v_new.
      use_exp2: Use exp2 for gate computation.

  Returns:
      h: [H, B, NT, K, V] -- Hidden states before each chunk.
      v_new: [H, B, T, V] or None -- Delta-corrected values.
      final_state: [B, N, H, K, V] or None.
  """
  H, B, T, K = k.shape
  V = v.shape[-1]
  BT = chunk_size

  assert seqlens is not None, "This varlen-only module requires seqlens"

  N = seqlens.shape[-1] - 1
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape_or_none(g, (H, B, T), "g")
  assert_shape_or_none(gk, (H, B, T, K), "gk")
  assert_shape_or_none(initial_state, (B, N, H, K, V), "initial_state")
  assert K <= 256, "current kernel does not support head dimension larger than 256."

  # --- Varlen launcher ---
  k = k.astype(jnp.float32)
  w = w.astype(jnp.float32)
  u_f32 = v.astype(jnp.float32)

  K_PADSIZE = int(align_up(K, 128))
  V_ALIGNED = int(align_up(V, 128))

  assert chunk_indices is not None
  NT = chunk_indices.shape[-2]
  chunk_to_seq = chunk_indices[:, :, 0].astype(jnp.int32)  # [B, NT]
  assert initial_state is None or initial_state.shape == (B, N, H, K, V)

  T_alloc = T

  k_pad = (
    jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    if K_PADSIZE > K
    else k
  )
  w_pad = (
    jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    if K_PADSIZE > K
    else w
  )
  k_t = k_pad
  w_t = w_pad

  v_pad = (
    jnp.pad(u_f32, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if V_ALIGNED > V
    else u_f32
  )
  v_t = v_pad

  if g is not None:
    g_fp32 = g.astype(jnp.float32).reshape(H, B, T, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
    g_t = g_fp32
  else:
    g_t = None

  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    if K_PADSIZE > K:
      gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    gk_t = gk_fp32
  else:
    gk_t = None

  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # --- Mini-batch auto-compute ---
  if mini_batch is None:
    per_head = K_PADSIZE * V_ALIGNED * 4  # scratch state bytes per head
    vmem_budget = 8 * 1024 * 1024  # 8 MB
    MB = max(1, vmem_budget // per_head)
    MB = min(MB, H, 16)
    while H % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  g_pad_size = g_t.shape[-1] if g_t is not None else 128
  h_spec = jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], k.dtype)
  v_new_spec = (
    jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
    if save_new_value
    else None
  )
  ht_spec = (
    jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state
    else None
  )

  def _t_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0)

  def _h_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
    return (h, b, c, 0, 0)

  k_blockspec = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  v_blockspec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
  w_blockspec = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
  g_blockspec = (
    pl.BlockSpec([MB, 1, BT, g_pad_size], index_map=_t_index_map)
    if g is not None
    else None
  )
  gk_blockspec = (
    pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
    if gk is not None
    else None
  )
  h0_blockspec = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (b, chunk_to_seq_ref[b, c], h, 0, 0)
    )
    if initial_state is not None
    else None
  )

  h_blockspec_out = pl.BlockSpec(
    [MB, 1, 1, K_PADSIZE, V_ALIGNED], index_map=_h_index_map
  )
  v_new_blockspec_out = (
    pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    if save_new_value
    else None
  )
  ht_blockspec_out = (
    pl.BlockSpec(
      [1, 1, MB, K_PADSIZE, V_ALIGNED],
      index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref: (b, chunk_to_seq_ref[b, c], h, 0, 0)
    )
    if output_final_state
    else None
  )

  scratch = pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32)
  grid = (H // MB, B, NT)
  interpret = get_interpret()

  h_out, v_new_out, ht_out = pl.pallas_call(
    functools.partial(
      _chunk_gated_delta_rule_fwd_varlen_kernel,
      NT=NT,
      USE_G=(g is not None),
      USE_GK=(gk is not None),
      USE_INITIAL_STATE=(initial_state is not None),
      STORE_FINAL_STATE=output_final_state,
      SAVE_NEW_VALUE=save_new_value,
      USE_EXP2=use_exp2,
      MINI_BATCH=MB,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=[
        k_blockspec,
        v_blockspec,
        w_blockspec,
        g_blockspec,
        gk_blockspec,
        h0_blockspec,
      ],
      out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
      scratch_shapes=[scratch],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary")
    ),
    out_shape=[h_spec, v_new_spec, ht_spec],
    interpret=interpret,
  )(seqlens, chunk_to_seq, k_t, v_t, w_t, g_t, gk_t, h0)

  h_out = h_out[:, :, :, :K, :V]
  v_new_out = (
    v_new_out[:, :, :T, :V] if save_new_value else None
  )

  if output_final_state and ht_out is not None:
    ht_out = ht_out[:, :, :, :K, :V]

    # Handle empty sequences: sequences with no chunks never execute kernel code,
    # so their final_state is uninitialized. Fill them with initial_state or zeros.
    seq_lens = jnp.diff(seqlens, axis=-1)  # [B, N]
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

  return h_out, v_new_out, ht_out


# ── Non-varlen Pallas kernel ────────────────────────────────────────────────


def _chunk_gated_delta_rule_fwd_kernel(
  k_ref,  # [1, 1, BT, K_PADSIZE]
  v_ref,  # [1, 1, BT, V_ALIGNED]
  w_ref,  # [1, 1, BT, K_PADSIZE]
  g_ref,  # [1, 1, BT, G_PAD]
  gk_ref,  # [1, 1, BT, K_PADSIZE]
  h0_ref,  # [1, 1, K_PADSIZE, V_ALIGNED]
  # outputs
  h_ref,  # [1, NT, 1, K_PADSIZE, V_ALIGNED]
  v_new_ref,  # [1, 1, BT, V_ALIGNED]
  ht_ref,  # [1, 1, K_PADSIZE, V_ALIGNED]
  scratch_ref,  # [K_PADSIZE, V_ALIGNED]
  *,
  NT,
  USE_EXP2,
):
  """Pallas kernel for one chunk of the gated delta-rule inter-chunk forward pass.

  This kernel is invoked on a grid of (B, H, NT). For each chunk ``idx_nt``
  it performs the following steps:

  1. **Initialise state** (chunk 0 only): set scratch (the running hidden
     state ``h``) to zeros or to the provided initial state ``h0``.
  2. **Snapshot**: write the *pre-update* state into ``h_ref`` so that the
     intra-chunk backward pass can use it.
  3. **Delta correction**: ``v_new = u - w @ h`` — correct the raw value
     by subtracting the projection of the current state through ``w``.
  4. **Gated state update**: apply scalar gate ``g`` and/or per-dim gate
     ``gk`` to decay the state, then accumulate the outer product
     ``k^T @ v_new`` into the running hidden state.
  5. **Final state** (last chunk only): write the state out to ``ht_ref``
     so it can be returned as the final hidden state.
  """

  idx_nt = pl.program_id(2)

  BT = k_ref.shape[2]
  K, V = k_ref.shape[-1], v_ref.shape[-1]
  b_k = k_ref[0, 0]

  @pl.when(idx_nt == 0)
  def _():
    scratch_ref[...] = jnp.zeros([K, V], dtype=jnp.float32)
    if h0_ref is not None:
      scratch_ref[...] = h0_ref[0, 0].astype(jnp.float32)

  h_ref[0, 0, idx_nt] = scratch_ref[...].astype(h_ref.dtype)

  b_w = w_ref[0, 0]
  b_v = jnp.dot(
    b_w.astype(jnp.float32),
    scratch_ref[...],
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  b_u = v_ref[0, 0]
  b_v = b_u.astype(b_v.dtype) - b_v
  if v_new_ref is not None:
    v_new_ref[0, 0] = b_v.astype(v_new_ref.dtype)

  if g_ref is not None:
    b_g = g_ref[0, 0, :, 0]
    b_g_last = g_ref[0, 0, BT - 1, 0].astype(jnp.float32)
    if USE_EXP2:
      b_v = b_v * exp2(b_g_last - b_g)[:, None]
      b_g_last = exp2(b_g_last)
    else:
      b_v = b_v * exp(b_g_last - b_g)[:, None]
      b_g_last = exp(b_g_last)
    scratch_ref[...] *= b_g_last
  if gk_ref is not None:
    b_gk_last = gk_ref[0, 0, BT - 1].astype(jnp.float32)
    if USE_EXP2:
      scratch_ref[...] *= exp2(b_gk_last)[:, None]
    else:
      scratch_ref[...] *= exp(b_gk_last)[:, None]

  scratch_ref[...] += jnp.dot(
    b_k.astype(jnp.float32).T,
    b_v.astype(jnp.float32),
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  @pl.when(idx_nt == NT - 1)
  def _():
    if ht_ref is not None:
      ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "output_final_state",
    "chunk_size",
    "save_new_value",
    "use_exp2",
  ],
)
def _chunk_gated_delta_rule_fwd(
  k: jax.Array,
  w: jax.Array,
  u: jax.Array,
  g: jax.Array | None = None,
  gk: jax.Array | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  chunk_size: int = 64,
  save_new_value: bool = True,
  use_exp2: bool = False,
):
  """Non-varlen launcher for the chunked gated delta rule forward pass.

  Requires T % chunk_size == 0 (enforced by ``chunk_gated_delta_rule_fwd_h``).
  Uses the dedicated ``_chunk_gated_delta_rule_fwd_kernel`` -- NOT the varlen
  path.  Follows _chunk_gla_fwd_o_gk's block-spec style: inputs are transposed
  to (H, B, T, dim) so each grid point loads one head's full T-length sequence.

  Grid:  (B, H, NT)

  Args:
      k: [H, B, T, K] -- Keys.
      w: [H, B, T, K] -- Correction weights.
      u: [H, B, T, V] -- Delta-corrected values from intra-chunk.
      g: [H, B, T] -- Scalar per-head gate (optional).
      gk: [H, B, T, K] -- Per-element gate (optional).
      initial_state: [B, H, K, V] -- Initial hidden state (optional).
      output_final_state: Whether to return final hidden state.
      chunk_size: Chunk size.
      save_new_value: Whether to compute and return v_new.
      use_exp2: Use exp2 for gate computation.

  Returns:
      h: [B, NT, H, K, V] -- Hidden states before each chunk.
      v_new: [B, T, H, V] or None -- Delta-corrected values.
      final_state: [B, H, K, V] or None.
  """
  H, B, T, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = T // BT  # exact -- T % BT == 0 enforced by caller

  BV = 128  # must be >=128 so pl.ds(idx_v*BV, BV) on last dim is provably 128-element-aligned on TPU
  K_PADSIZE = int(align_up(K, 128))  # pad K to 128-element blocks
  V_ALIGNED = int(align_up(V, BV))  # pad V to BV-element blocks (multiple of 128)
  NV = V_ALIGNED // BV

  # -- Pad and transpose inputs to (H, B, T, dim) layout ---
  # k, w: [B, T, H, K] -> pad K -> transpose -> (B, H, T, K_PADSIZE)
  k_pad = (
    jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K))) if K_PADSIZE > K else k
  )
  w_pad = (
    jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K))) if K_PADSIZE > K else w
  )
  k_t = k_pad  # (H, B, T, K_PADSIZE)
  w_t = w_pad  # (H, B, T, K_PADSIZE)

  # u (values): [B, T, H, V] -> pad V -> transpose -> (H, B, T, V_ALIGNED)
  u_pad = (
    jnp.pad(u, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V))) if V_ALIGNED > V else u
  )
  v_t = u_pad  # (H, B, T, V_ALIGNED)

  # g (scalar gate): [H, B, T] -> float32 -> [H, B, T, 1] -> pad last -> [H, B, T, G_PAD]
  if g is not None:
    g_fp32 = g.astype(jnp.float32).reshape(H, B, T, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)  # (H, B, T, 128)
    g_t = g_fp32  # (H, B, T, 128)
  else:
    g_t = None

  # gk (per-dim gate): [H, B, T, K] -> float32 -> pad K -> (H, B, T, K_PADSIZE)
  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    if K_PADSIZE > K:
      gk_fp32 = jnp.pad(gk_fp32, ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K)))
    gk_t = gk_fp32  # (H, B, T, K_PADSIZE)
  else:
    gk_t = None

  # h0 (initial state): [N=B, H, K, V] -> pad V, K -> transpose -> (B, H, K_PADSIZE, V_ALIGNED)
  if initial_state is not None:
    h0 = initial_state
    if V_ALIGNED > V:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
    if K_PADSIZE > K:
      h0 = jnp.pad(h0, ((0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
  else:
    h0 = None

  # -- Output shapes ---
  # h stored as [B, NT, H, K_PADSIZE, V_ALIGNED] with V before K (varlen convention).
  h_spec = jax.ShapeDtypeStruct([H, B, NT, K_PADSIZE, V_ALIGNED], k.dtype)
  v_new_spec = (
    jax.ShapeDtypeStruct([H, B, T, V_ALIGNED], jnp.float32) if save_new_value else None
  )
  ht_spec = (
    jax.ShapeDtypeStruct([B, H, K_PADSIZE, V_ALIGNED], jnp.float32)
    if output_final_state
    else None
  )

  # -- Block specs ---
  g_pad_size = (
    g_t.shape[-1] if g_t is not None else 128
  )  # always 128 after pad_to_multiple

  k_blockspec = pl.BlockSpec(
    [1, 1, BT, K_PADSIZE], index_map=lambda b, h, nt: (h, b, nt, 0)
  )
  v_blockspec = pl.BlockSpec(
    [1, 1, BT, V_ALIGNED], index_map=lambda b, h, nt: (h, b, nt, 0)
  )
  w_blockspec = pl.BlockSpec(
    [1, 1, BT, K_PADSIZE], index_map=lambda b, h, nt: (h, b, nt, 0)
  )
  g_blockspec = (
    pl.BlockSpec([1, 1, BT, g_pad_size], index_map=lambda b, h, nt: (h, b, nt, 0))
    if g is not None
    else None
  )
  gk_blockspec = (
    pl.BlockSpec([1, 1, BT, K_PADSIZE], index_map=lambda b, h, nt: (h, b, nt, 0))
    if gk is not None
    else None
  )
  h0_blockspec = (
    pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], index_map=lambda b, h, nt: (b, h, 0, 0))
    if initial_state is not None
    else None
  )

  h_blockspec_out = pl.BlockSpec(
    [1, 1, NT, K_PADSIZE, V_ALIGNED], lambda b, h, nt: (h, b, 0, 0, 0)
  )
  v_new_blockspec_out = (
    pl.BlockSpec([1, 1, BT, V_ALIGNED], lambda b, h, nt: (h, b, nt, 0))
    if save_new_value
    else None
  )
  ht_blockspec_out = (
    pl.BlockSpec([1, 1, K_PADSIZE, V_ALIGNED], lambda b, h, nt: (b, h, 0, 0))
    if output_final_state
    else None
  )

  scratch = pltpu.VMEM((K_PADSIZE, V_ALIGNED), jnp.float32)
  scratch_shapes = [scratch]

  grid = (B, H, NT)
  interpret = get_interpret()
  h_out, v_new_out, ht_out = pl.pallas_call(
    functools.partial(
      _chunk_gated_delta_rule_fwd_kernel,
      NT=NT,
      USE_EXP2=use_exp2,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=grid,
      in_specs=[
        k_blockspec,
        v_blockspec,
        w_blockspec,
        g_blockspec,
        gk_blockspec,
        h0_blockspec,
      ],
      out_specs=[h_blockspec_out, v_new_blockspec_out, ht_blockspec_out],
      scratch_shapes=scratch_shapes,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=(
        "parallel",
        "parallel",
        "arbitrary",
      ),
      # vmem_limit_bytes=32 * 1024 * 1024,
      disable_bounds_checks=True,
    ),
    out_shape=[h_spec, v_new_spec, ht_spec],
    interpret=interpret,
  )(k_t, v_t, w_t, g_t, gk_t, h0)

  # -- Post-process outputs ---
  # h: [H, B, NT, K_PADSIZE, V_ALIGNED] -> trim K and V padding -> [H, B, NT, K, V]
  h_out = h_out[:, :, :, :K, :V]

  # v_new: [B, H, T, V_ALIGNED] -> [B, T, H,  V]
  if save_new_value:
    if V_ALIGNED > V:
      v_new_out = v_new_out[:, :, :, :V]
  else:
    v_new_out = None

  # ht: [B, H, K_PADSIZE, V_ALIGNED] -> trim K and V padding -> [B, H, K, V]
  if output_final_state:
    ht_out = ht_out[:, :, :K, :V]
  else:
    ht_out = None

  return h_out, v_new_out, ht_out


# ── Public dispatch function ────────────────────────────────────────────────


def chunk_gated_delta_rule_fwd_h(
  k: jax.Array,  # [H, B, T, K]
  w: jax.Array,  # [H, B, T, K]
  u: jax.Array,  # [H, B, T, V]
  g: jax.Array | None = None,  # [H, B, T] scalar gate
  gk: jax.Array | None = None,  # [H, B, T, K] per-element gate
  initial_state: jax.Array | None = None,  # [B, H, K, V]
  output_final_state: bool = False,
  chunk_size: int = 64,
  save_new_value: bool = True,
  use_exp2: bool = True,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
  _cu_seqlens: jax.Array | None = None,
  _chunk_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
  """Dispatch delta-rule inter-chunk state forward to varlen or non-varlen kernel.

  Computes the chunked recurrence where the value is delta-corrected before
  accumulation:

      v_new_t = u_t - w_t @ h_{t-1}    (delta correction)
      h_t = h_{t-1} * decay_t + k_t^T @ v_new_t  (state update)

  Args:
      k: [H, B, T, K] -- Keys (gated, ready for outer product).
      w: [H, B, T, K] -- Correction weights.
      u: [H, B, T, V] -- Delta-corrected values from intra-chunk.
      g: [H, B, T] -- Scalar per-head gate in log2 space. Optional.
      gk: [H, B, T, K] -- Per-element gate in log2 space. Optional.
      initial_state: [N, H, K, V] -- Initial hidden state. N=B for non-varlen.
      output_final_state: Whether to return final hidden state.
      chunk_size: Chunk size. T must be divisible by chunk_size.
      save_new_value: Whether to compute and return v_new.
      use_exp2: Use exp2 for gate computation (True for log2 space).
      cu_seqlens: [N+1] -- Cumulative sequence lengths for varlen. Optional.
      chunk_indices: Precomputed chunk indices for varlen. Optional.

  Returns:
      h: [B, NT, H, K, V] -- Hidden states before each chunk.
      v_new: [B, T, H, V] -- Delta-corrected values, or None.
      final_state: [N, H, K, V] or None.
  """
  H, B, T, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)
  # Stage3+4 cherry-pick: accept private aliases (forwarded by callers that
  # precompute aligned cu_seqlens/chunk_indices for the gather/scatter path).
  if cu_seqlens is None and _cu_seqlens is not None:
    cu_seqlens = _cu_seqlens
  if chunk_indices is None and _chunk_indices is not None:
    chunk_indices = _chunk_indices
  N = B if cu_seqlens is None else cu_seqlens.shape[-1] - 1

  # -- Input validation ---
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(u, (H, B, T, V), "u")
  assert_shape_or_none(g, (H, B, T), "g")
  assert_shape_or_none(gk, (H, B, T, K), "gk")
  if cu_seqlens is not None:
    if initial_state is not None and initial_state.ndim == 4:
      initial_state = initial_state[None, :, :, :, :]
    assert_shape_or_none(initial_state, (B, N, H, K, V), "initial_state")
  else:
    if initial_state is not None and initial_state.ndim == 5:
      initial_state = initial_state[:, 0]
    assert_shape_or_none(initial_state, (N, H, K, V), "initial_state")
  assert K <= 256, "current kernel does not support head dimension larger than 256."

  if cu_seqlens is None:
    assert T % chunk_size == 0, (
      "For non-varlen input, T must be divisible by chunk_size"
    )
    return _chunk_gated_delta_rule_fwd(
      k,
      w,
      u,
      g=g,
      gk=gk,
      initial_state=initial_state,
      output_final_state=output_final_state,
      chunk_size=chunk_size,
      save_new_value=save_new_value,
      use_exp2=use_exp2,
    )
  else:
    # _chunk_gated_delta_rule_fwd_varlen expects B-batched inputs:
    #   seqlens: [B, N+1], chunk_indices: [B, NT, 2],
    #   initial_state: [B, N, H, K, V].
    # When B=1 the caller may pass 1D seqlens and 2D chunk_indices;
    # unsqueeze them here. initial_state must always be 5D.
    _varlen_cu = cu_seqlens
    _varlen_ci = chunk_indices
    _varlen_h0 = initial_state
    if _varlen_cu is not None and _varlen_cu.ndim == 1:
      _varlen_cu = _varlen_cu[None, :]
    if _varlen_ci is not None and _varlen_ci.ndim == 2:
      _varlen_ci = _varlen_ci[None, :, :]
    h, v_new, final_state = _chunk_gated_delta_rule_fwd_varlen(
      k,
      w,
      u,
      seqlens=_varlen_cu,
      chunk_indices=_varlen_ci,
      g=g,
      gk=gk,
      initial_state=_varlen_h0,
      output_final_state=output_final_state,
      chunk_size=chunk_size,
      save_new_value=save_new_value,
      use_exp2=use_exp2,
    )
    return h, v_new, final_state

def _chunk_gated_delta_rule_bwd_dhu_pre_process_kernel(
  chunk_active_ref,   # SMEM [NT] int32 -- per-chunk activeness mask
  q_ref,              # HBM [H*NT*BT, K]
  k_ref,              # HBM [H*NT*BT, K]
  w_ref,              # HBM [H*NT*BT, K]
  do_ref,             # HBM [H*NT*BT, V]
  dv_ref,             # HBM [H*NT*BT, V]
  gk_ref,             # HBM [H*NT*BT, K]
  # outputs
  dS_ext_ref,         # HBM [H, K, V]
  dM_ref,             # HBM [H, K, K]
  # scratch: 6 double-buffered inputs (with MB batch dim)
  q_scratch_ref,      # VMEM [2, MB, BT, K]
  k_scratch_ref,      # VMEM [2, MB, BT, K]
  w_scratch_ref,      # VMEM [2, MB, BT, K]
  do_scratch_ref,     # VMEM [2, MB, BT, V]
  dv_scratch_ref,     # VMEM [2, MB, BT, V]
  gk_scratch_ref,     # VMEM [2, MB, BT, K]
  # scratch: accumulators (with MB batch dim)
  dh_acc_ref,         # VMEM [MB, K, V] fp32
  M_acc_ref,          # VMEM [MB, K, K] fp32
  # semaphores
  sems,               # DMA [8, 2]
  *,
  H,
  NT,
  BT,
  K,
  MB,
  SCALE,
  USE_EXP2,
):
  """Reverse-time chunk-recurrence kernel with hand-pipelined DMA.

  Single-program kernel (grid=()) that loops over N_HG*NT iterations,
  processing head-groups of MB heads sequentially and chunks in reverse
  time order.  Double-buffered input DMA overlaps with MXU compute.
  MB heads are batched via ``dot_general`` batch dim (no Python unroll).
  """
  N_HG = H // MB
  TOTAL = N_HG * NT
  eye_k = jnp.eye(K, dtype=jnp.float32)
  SEM_OUT = 6 * MB

  def _async_copy(src, dst, sem, wait=False):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _iter_to_hg_i(it):
    hg = it // NT
    i_rev = it % NT
    return hg, NT - 1 - i_rev

  def start_input_dma(buf, hg, i):
    for h_local in range(MB):
      off = (hg * MB + h_local) * NT * BT + i * BT
      _async_copy(q_ref.at[pl.ds(off, BT), pl.ds(None)], q_scratch_ref.at[buf, h_local], sems.at[0 * MB + h_local, buf])
      _async_copy(k_ref.at[pl.ds(off, BT), pl.ds(None)], k_scratch_ref.at[buf, h_local], sems.at[1 * MB + h_local, buf])
      _async_copy(w_ref.at[pl.ds(off, BT), pl.ds(None)], w_scratch_ref.at[buf, h_local], sems.at[2 * MB + h_local, buf])
      _async_copy(do_ref.at[pl.ds(off, BT), pl.ds(None)], do_scratch_ref.at[buf, h_local], sems.at[3 * MB + h_local, buf])
      _async_copy(dv_ref.at[pl.ds(off, BT), pl.ds(None)], dv_scratch_ref.at[buf, h_local], sems.at[4 * MB + h_local, buf])
      _async_copy(gk_ref.at[pl.ds(off, BT), pl.ds(None)], gk_scratch_ref.at[buf, h_local], sems.at[5 * MB + h_local, buf])

  def wait_input_dma(buf, hg, i):
    for h_local in range(MB):
      off = (hg * MB + h_local) * NT * BT + i * BT
      _async_copy(q_ref.at[pl.ds(off, BT), pl.ds(None)], q_scratch_ref.at[buf, h_local], sems.at[0 * MB + h_local, buf], True)
      _async_copy(k_ref.at[pl.ds(off, BT), pl.ds(None)], k_scratch_ref.at[buf, h_local], sems.at[1 * MB + h_local, buf], True)
      _async_copy(w_ref.at[pl.ds(off, BT), pl.ds(None)], w_scratch_ref.at[buf, h_local], sems.at[2 * MB + h_local, buf], True)
      _async_copy(do_ref.at[pl.ds(off, BT), pl.ds(None)], do_scratch_ref.at[buf, h_local], sems.at[3 * MB + h_local, buf], True)
      _async_copy(dv_ref.at[pl.ds(off, BT), pl.ds(None)], dv_scratch_ref.at[buf, h_local], sems.at[4 * MB + h_local, buf], True)
      _async_copy(gk_ref.at[pl.ds(off, BT), pl.ds(None)], gk_scratch_ref.at[buf, h_local], sems.at[5 * MB + h_local, buf], True)

  # ── Pre-loop prologue: start DMA for first iteration ────────────────
  hg0, i0 = _iter_to_hg_i(0)
  start_input_dma(0, hg0, i0)

  # ── Main loop ────────────────────────────────────────────────────────
  @pl.loop(0, TOTAL, unroll=False)
  def body(it):
    buf = it % 2
    hg = it // NT
    i_rev = it % NT
    i = NT - 1 - i_rev

    # 1. Wait ping-in (current buffer DMA complete)
    wait_input_dma(buf, hg, i)

    # 2. Trigger pong-in (start DMA for next iteration into other buffer)
    @pl.when(it + 1 < TOTAL)
    def _():
      next_hg, next_i = _iter_to_hg_i(it + 1)
      start_input_dma(1 - buf, next_hg, next_i)

    # 3. Wait prev output DMA (ping out) — at head-group boundary
    @pl.when((i_rev == 0) & (hg > 0))
    def _():
      prev_off = (hg - 1) * MB
      _async_copy(dh_acc_ref, dS_ext_ref.at[pl.ds(prev_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT, 0], wait=True)
      _async_copy(M_acc_ref, dM_ref.at[pl.ds(prev_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT + 1, 0], wait=True)

    # 4. Reset accumulators at start of each head group
    @pl.when(i_rev == 0)
    def _():
      dh_acc_ref[...] = jnp.zeros_like(dh_acc_ref[...])
      M_acc_ref[...] = jnp.broadcast_to(eye_k, (MB, K, K))

    # 5. Compute on ping (only for active chunks)
    is_active = chunk_active_ref[i] != 0
    @pl.when(is_active)
    def _():
      bq = q_scratch_ref[buf].astype(jnp.float32)    # [MB, BT, K]
      bk = k_scratch_ref[buf].astype(jnp.float32)    # [MB, BT, K]
      bw = w_scratch_ref[buf].astype(jnp.float32)    # [MB, BT, K]
      bdo = do_scratch_ref[buf].astype(jnp.float32)   # [MB, BT, V]
      bdv = dv_scratch_ref[buf]                        # [MB, BT, V]
      bgk = gk_scratch_ref[buf].astype(jnp.float32)   # [MB, BT, K]

      dh = dh_acc_ref[...]   # [MB, K, V]
      M = M_acc_ref[...]     # [MB, K, K]

      # (1) dv_cur = k @ dh + dv → [MB, BT, V]
      dv_cur = jax.lax.dot_general(
        bk, dh,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
      ) + bdv

      # decay
      gk_last = jnp.maximum(bgk[:, BT - 1, :], jnp.asarray(-126.0, dtype=jnp.float32))
      decay = exp2(gk_last) if USE_EXP2 else exp(gk_last)   # [MB, K]

      # (2) dh *= decay → [MB, K, V]
      dh_new = dh * decay[:, :, None]

      # (3) dh += q.T @ do * scale - w.T @ dv_cur → [MB, K, V]
      dh_new = dh_new + (
        jax.lax.dot_general(
          bq, bdo,
          (((1,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
        ) * SCALE
        - jax.lax.dot_general(
          bw, dv_cur,
          (((1,), (1,)), ((0,), (0,))),
          preferred_element_type=jnp.float32,
        )
      )

      # (4) kM = k @ M → [MB, BT, K]
      kM = jax.lax.dot_general(
        bk, M,
        (((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
      )

      # (5) WkM = w.T @ kM → [MB, K, K]
      WkM = jax.lax.dot_general(
        bw, kM,
        (((1,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
      )
      M_new = M * decay[:, :, None] - WkM

      dh_acc_ref[...] = dh_new
      M_acc_ref[...] = M_new

    # 6. Trigger ping-out at end of each head group
    @pl.when(i_rev == NT - 1)
    def _():
      _async_copy(dh_acc_ref, dS_ext_ref.at[pl.ds(hg * MB, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT, 0])
      _async_copy(M_acc_ref, dM_ref.at[pl.ds(hg * MB, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT + 1, 0])

  # ── Post-loop epilogue: wait for final output DMA ────────────────────
  last_off = (N_HG - 1) * MB
  _async_copy(dh_acc_ref, dS_ext_ref.at[pl.ds(last_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT, 0], wait=True)
  _async_copy(M_acc_ref, dM_ref.at[pl.ds(last_off, MB), pl.ds(None), pl.ds(None)], sems.at[SEM_OUT + 1, 0], wait=True)


@functools.partial(
  jax.jit,
  static_argnames=[
    "scale",
    "chunk_size",
    "use_exp2",
  ],
)
def chunk_gated_delta_rule_bwd_dhu_pre_process(
  q: jax.Array,
  k: jax.Array,
  w: jax.Array,
  do: jax.Array,
  dv: jax.Array,
  gk: jax.Array,
  scale: float,
  segment_ids: jax.Array | None = None,
  chunk_size: int = 64,
  use_exp2: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Compute CP backward affine summary for the FIRST real segment on this rank.

  Symmetric with the forward pre-process: only the FIRST segment receives
  cross-rank initial state in the forward pass, so only its backward
  summary needs to be communicated to the upstream rank.  Segments that
  start fresh on this rank have dht=0 in Stage 2 and their backward
  state is consumed locally — they do not participate in the CP chain.

  Inputs are in head-first ``[H, B, T, X]`` layout (matching
  ``chunk_kda_bwd``'s native packing). ``B == 1`` is required.

  Args:
    segment_ids: [1, T] or [T] rank-local segment IDs (0 = padding).
  """
  H, B, T, K = q.shape
  V = do.shape[-1]
  BT = chunk_size
  NT = T // BT

  # Handle B > 1 by looping over batch elements with the B=1 kernel.
  if B > 1:
    ds_list, dm_list = [], []
    for b in range(B):
      seg_b = segment_ids[b] if segment_ids.ndim == 2 else segment_ids
      ds_b, dm_b = chunk_gated_delta_rule_bwd_dhu_pre_process(
        q=q[:, b:b+1], k=k[:, b:b+1], w=w[:, b:b+1],
        do=do[:, b:b+1], dv=dv[:, b:b+1], gk=gk[:, b:b+1],
        scale=scale, segment_ids=seg_b,
        chunk_size=chunk_size, use_exp2=use_exp2,
      )
      ds_list.append(ds_b)
      dm_list.append(dm_b)
    return jnp.concatenate(ds_list, axis=0), jnp.concatenate(dm_list, axis=0)

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert K % 128 == 0, f"K={K} must be a multiple of 128 (TPU lane alignment)"
  assert V % 128 == 0, f"V={V} must be a multiple of 128 (TPU lane alignment)"
  assert segment_ids is not None, "segment_ids is required"

  seg = segment_ids
  if seg.ndim == 2:
    assert seg.shape[0] == 1
    seg = seg[0]

  assert seg.shape[0] == T, (
    f"segment_ids length ({seg.shape[0]}) must equal T ({T}); caller must "
    f"align/pad segment_ids to match q/k/v's aligned T."
  )

  if NT == 0:
    return (
      jnp.zeros([B, H, K, V], dtype=jnp.float32),
      jnp.zeros([B, H, K, K], dtype=jnp.float32),
    )

  # ── Precompute per-chunk activeness mask (SMEM scalar prefetch) ──────
  first_real_idx = jnp.argmax((seg != 0).astype(jnp.int32))
  first_seg_id = seg[first_real_idx]
  has_real = first_seg_id != 0
  chunk_seg_ids = seg.reshape(NT, BT)[:, 0]
  chunk_active = ((chunk_seg_ids == first_seg_id) & has_real).astype(jnp.int32)

  # ── Auto MB: maximise VMEM utilisation, capped at min(H, 16) ──
  q_elem = 2 if q.dtype == jnp.bfloat16 else 4
  gk_elem = 2 if gk.dtype == jnp.bfloat16 else 4
  do_elem = 2 if do.dtype == jnp.bfloat16 else 4
  per_head = (
    2 * BT * (3 * K * q_elem + K * gk_elem + 2 * V * do_elem)
    + (K * V + K * K) * 4
  )
  vmem_budget = 8 * 1024 * 1024
  MB = max(1, vmem_budget // per_head)
  MB = min(MB, H, 16)
  while H % MB != 0 and MB > 1:
    MB -= 1

  # ── Reshape inputs: [H, 1, T, dim] → [H*NT*BT, dim] (no transpose) ──
  q_r = q.reshape(H * NT * BT, K)
  k_r = k.reshape(H * NT * BT, K)
  w_r = w.reshape(H * NT * BT, K)
  do_r = do.reshape(H * NT * BT, V)
  dv_r = dv.reshape(H * NT * BT, V)
  gk_r = gk.reshape(H * NT * BT, K)

  # ── Output shapes: [H, K, V] and [H, K, K] ──────────────────────────
  dS_ext_spec = jax.ShapeDtypeStruct([H, K, V], jnp.float32)
  dM_spec = jax.ShapeDtypeStruct([H, K, K], jnp.float32)

  # ── Scratch: double-buffered inputs + accumulators + semaphores ──────
  scratch_shapes = [
    pltpu.VMEM((2, MB, BT, K), q.dtype),   # q
    pltpu.VMEM((2, MB, BT, K), k.dtype),   # k
    pltpu.VMEM((2, MB, BT, K), w.dtype),   # w
    pltpu.VMEM((2, MB, BT, V), do.dtype),  # do
    pltpu.VMEM((2, MB, BT, V), dv.dtype),  # dv
    pltpu.VMEM((2, MB, BT, K), gk.dtype),  # gk
    pltpu.VMEM((MB, K, V), jnp.float32),   # dh accumulator
    pltpu.VMEM((MB, K, K), jnp.float32),   # M accumulator
    pltpu.SemaphoreType.DMA((6 * MB + 2, 2)),  # 6*MB input + 2 output channels
  ]

  in_specs = [pl.BlockSpec(memory_space=pl.ANY)] * 6
  out_specs = [pl.BlockSpec(memory_space=pl.ANY)] * 2

  interpret = get_interpret()

  dS_ext_raw, dM_raw = pl.pallas_call(
    functools.partial(
      _chunk_gated_delta_rule_bwd_dhu_pre_process_kernel,
      H=H, NT=NT, BT=BT, K=K, MB=MB,
      SCALE=float(scale),
      USE_EXP2=use_exp2,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=scratch_shapes,
    ),
    out_shape=[dS_ext_spec, dM_spec],
    interpret=interpret,
  )(chunk_active, q_r, k_r, w_r, do_r, dv_r, gk_r)

  # ── Reshape outputs: [H, K, V] → [1, H, K, V] ──────────────────────
  dS_ext = dS_ext_raw.reshape(B, H, K, V)
  dM = dM_raw.reshape(B, H, K, K)

  return dS_ext, dM


# ─── Context-Parallel Pre-Process ──────────────────────────────────────────


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
# KDA gate helpers
# =============================================================================

"""KDA gate: chunk-local cumulative sum in log2 space.

KDA uses per-element gates g: [H, B, T, K] in natural log space
(heads-first layout).
This module converts to log2 space and applies chunk-local cumsum.

The conversion is: g_log2 = g / ln(2), so that exp2(cumsum(g_log2)) == exp(cumsum(g)).
The kernel then uses exp2() for all gate computations — matching FLA convention.
"""

import functools
import math

import jax
import jax.numpy as jnp

from tokamax._src.ops.experimental.kda.utils import assert_shape

# Plain Python float; computed via `math.log` rather than `jnp.log` so that
# importing this module does not eagerly initialise the JAX/XLA backend.
# Doing so at module import time breaks downstream callers that need to invoke
# `jax.distributed.initialize()` first (e.g. multi-host training entry points).
_LN2 = math.log(2.0)


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


def kda_gate_bwd(
  g: jax.Array,
  A_log: jax.Array,
  dt_bias: jax.Array | None = None,
  dyg: jax.Array | None = None,
  lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Backward pass for the KDA gate function.

  Computes gradients w.r.t. the original gate input g, the log-parameter
  A_log, and the optional dt_bias.

  Forward semantics (for reference):
    - Without lower_bound:  yg = -exp(A_log) * softplus(g + dt_bias)
    - With lower_bound:     yg = lower_bound * sigmoid(exp(A_log) * g)

  Args:
      g:           [H, B, T, K] — original gate input (before activation),
                    heads-first layout.
      A_log:       [H]           — log of the A parameter.
      dt_bias:     [H * K] or None — optional bias added to g.
      dyg:         [H, B, T, K] — upstream gradient w.r.t. yg.
      lower_bound: float or None — if set, use sigmoid mode.

  Returns:
      dg:    [H, B, T, K] — gradient w.r.t. g (cast to g.dtype).
      dA:    [H]          — gradient w.r.t. A_log.
      dbias: [H * K] or None — gradient w.r.t. dt_bias.
  """
  H, K = g.shape[0], g.shape[-1]
  assert g.ndim == 4, f"g must be 4-D [H, B, T, K], got {g.ndim}-D"
  assert dyg is not None, "dyg must be provided"
  assert A_log.shape == (H,), f"A_log shape {A_log.shape} != ({H},)"

  g_f = g.astype(jnp.float32)
  dyg_f = dyg.astype(jnp.float32)

  # Apply bias if present
  if dt_bias is not None:
    g_f = g_f + dt_bias.reshape(H, 1, 1, K).astype(jnp.float32)

  if lower_bound is None:
    # Forward: yg = -exp(A_log) * softplus(g + bias)
    # softplus(x) = log(1 + exp(x)), d/dx softplus(x) = sigmoid(x)
    b_A = -jnp.exp(A_log.astype(jnp.float32))  # [H]
    b_yg = b_A.reshape(H, 1, 1, 1) * jax.nn.softplus(g_f)  # [H, B, T, K]
    dg_f = b_A.reshape(H, 1, 1, 1) * (dyg_f * jax.nn.sigmoid(g_f))  # [H, B, T, K]
    dA_per_elem = dyg_f * b_yg  # [H, B, T, K]
  else:
    # Forward: yg = lower_bound * sigmoid(exp(A_log) * g)
    b_A = jnp.exp(A_log.astype(jnp.float32))  # [H]
    b_inner = b_A.reshape(H, 1, 1, 1) * g_f  # [H, B, T, K]
    b_sig = jax.nn.sigmoid(b_inner)
    b_dsig = b_sig * (1.0 - b_sig)
    dg_f = dyg_f * (lower_bound * b_dsig) * b_A.reshape(H, 1, 1, 1)  # [H, B, T, K]
    dA_per_elem = dg_f * g_f  # [H, B, T, K]

  # dA: reduce over all dims except H (axis 0) → [H]
  reduce_axes = (1, 2, 3)
  dA = jnp.sum(dA_per_elem, axis=reduce_axes)
  dA = dA.astype(A_log.dtype)

  # Cast dg back to input dtype
  dg = dg_f.astype(g.dtype)

  # dbias: sum dg over B, T (axes 1, 2) → [H, K] → [H*K]
  if dt_bias is not None:
    dbias = jnp.sum(dg_f, axis=(1, 2)).reshape(-1)  # [H*K]
    dbias = dbias.astype(dt_bias.dtype)
  else:
    dbias = None

  return dg, dA, dbias



# =============================================================================
# WY recompute helpers
# =============================================================================

import functools
from functools import partial

import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
    align_up,
    assert_shape,
    get_interpret,
    get_tpu_config,
)

# =====================================================================
# recompute_w_u_fwd  —  Pallas kernel (replaces CPU einsum reference)
# =====================================================================


def _recompute_w_u_fwd_kernel(
  # Inputs (Ref)
  k_ref,
  v_ref,
  beta_ref,
  A_ref,
  q_ref,
  gk_ref,
  # Outputs (Ref)
  u_ref,
  w_ref,
  qg_ref,
  kg_ref,
  *,
  BT,
  K,
  V,
  MB,
):
  """Pallas kernel body for WY recomputation — MB chunk tiles.

  Processes MB chunks simultaneously via vectorized batched ops.

  Per chunk of size BT:
      u  = A @ (v * beta)            — delta-corrected values
      w  = A @ (k * beta * exp2(gk)) — correction weights
      qg = q * exp2(gk)             — gated query
      kg = k * exp2(gn - gk)        — gated key (gn = gk[BT-1])
  """
  dtype = q_ref.dtype

  q = q_ref[:]    # [MB, BT, K]
  k = k_ref[:]    # [MB, BT, K]
  v = v_ref[:]    # [MB, BT, V]
  beta = beta_ref[:]  # [MB, BT]
  gk = gk_ref[:]  # [MB, BT, K]
  A = A_ref[:]    # [MB, BT, BT]

  beta = beta[:, :, None]  # [MB, BT, 1]
  # u = A @ (v * beta)
  v_beta = v * beta  # [MB, BT, V]
  u = jnp.matmul(
    A.astype(jnp.float32),
    v_beta.astype(jnp.float32),
    preferred_element_type=jnp.float32,
  )  # [MB, BT, V]

  # w = A @ (k * beta * exp2(gk))
  k_beta_gated = k * beta * jnp.exp2(gk)  # [MB, BT, K]
  w = jnp.matmul(
    A.astype(jnp.float32),
    k_beta_gated.astype(jnp.float32),
    preferred_element_type=jnp.float32,
  )  # [MB, BT, K]

  # qg = q * exp2(gk)
  qg = (q * jnp.exp2(gk)).astype(dtype)

  # kg = k * exp2(gn - gk), where gn = gk at last position
  gn = gk[:, -1:, :]  # [MB, 1, K]
  kg = k * jnp.exp2(gn - gk)  # [MB, BT, K]

  u_ref[:] = u.astype(dtype)
  w_ref[:] = w.astype(dtype)
  qg_ref[:] = qg.astype(dtype)
  kg_ref[:] = kg.astype(dtype)


@functools.partial(jax.jit, static_argnames=["chunk_size", "mini_batch"])
def recompute_w_u_fwd_pallas(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  beta: jax.Array,
  A: jax.Array,
  gk: jax.Array,
  chunk_size: int = 64,
  mini_batch: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Pallas TPU implementation of recompute_w_u_fwd.

  Replaces the CPU reference (jnp.einsum) with an optimized Pallas kernel
  using explicit matmuls. Each chunk tile is processed independently in
  parallel across a flattened (H*B*NT // MB) grid.

  Within each chunk of size BT:
      u  = A @ (v * beta[:, None])            — WY-transformed value
      w  = A @ (k * beta[:, None] * exp2(gk)) — WY-transformed key (gated)
      qg = q * exp2(gk)                       — gated query
      kg = k * exp2(gn - gk)                  — gated key (gn = gk[-1])

  Args:
      q:    [H, B, T, K] — query vectors (heads-first layout).
      k:    [H, B, T, K] — key vectors.
      v:    [H, B, T, V] — value vectors (V may differ from K).
      beta: [H, B, T]    — per-token scalar mixing coefficient.
      A:    [H, B, T, BT] — Akk^{-1} matrix (lower triangular per chunk).
      gk:   [H, B, T, K] — chunk-local cumsummed gates (log2 space).
      chunk_size: int     — tile/chunk size (BT). T must be divisible by BT.
      mini_batch: int or None. Number of chunks per grid point for DMA
          granularity. None = auto-compute to maximise VMEM utilisation.

  Returns:
      w:  [H, B, T, K] — WY-transformed key (gated).
      u:  [H, B, T, V] — WY-transformed value.
      qg: [H, B, T, K] — gated query.
      kg: [H, B, T, K] — gated key.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(A, (H, B, T, BT), "A")
  assert_shape(gk, (H, B, T, K), "gk")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

  BH = B * H
  total = BH * NT

  q_r = q.reshape(-1, BT, K)
  k_r = k.reshape(-1, BT, K)
  v_r = v.reshape(-1, BT, V)
  gk_r = gk.reshape(-1, BT, K)
  beta_r = beta.reshape(-1, BT)
  A_r = A.reshape(-1, BT, BT)

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  align_minor = get_tpu_config().block_align_minor
  if mini_batch is None:
    elem_size = 2 if v.dtype == jnp.bfloat16 else 4
    # Input + output buffers (original dtype).
    in_bytes = (3 * BT * K + BT * V + BT * 1 + BT * BT) * elem_size
    out_bytes = (3 * BT * K + BT * V) * elem_size
    # float32 intermediates from kernel matmul casts:
    #   A_f32[BT,BT], v_beta_f32[BT,V], k_beta_gated_f32[BT,K],
    #   u_f32[BT,V], w_f32[BT,K] + ~1 matmul scratch per op
    f32_intermediate = (BT * BT + 3 * BT * V + 3 * BT * K) * 4
    # Pallas compiler adds alignment padding, instruction scratch, and
    # temporary VMEM allocations.  A 2× multiplier on our estimate has
    # been calibrated against observed OOMs on v6e (32 MB VMEM).
    per_chunk = (in_bytes + out_bytes + f32_intermediate) * 2
    vmem_budget = get_tpu_config().vmem_hw_limit_bytes
    # 1. hardware upper bound from VMEM, capped at total
    hw_mb = max(1, vmem_budget // per_chunk)
    MB = min(hw_mb, total)
    # 2. align upward to block_align_minor (TPU dim=-2 constraint)
    MB = min(align_up(MB, align_minor), total)
    # 3. try to make total divisible by MB; reduce MB first, pad as fallback
    need_pad = False
    if total % MB != 0:
      mb_try = MB
      while mb_try > 1:
        mb_try -= 1
        if total % mb_try == 0 and (mb_try == total or mb_try % align_minor == 0):
          MB = mb_try
          break
      else:
        need_pad = True
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"
    assert MB % align_minor == 0 or MB == total, (
      f"mini_batch={MB} must be divisible by {align_minor} or equal to total={total}"
      f" (TPU block shape constraint for 2D block spec)"
    )
    need_pad = False

  # ---- pad if needed: round total up to a multiple of MB ----
  total_padded = total
  if need_pad:
    total_padded = ((total + MB - 1) // MB) * MB
    pad_width = total_padded - total
    q_r = jnp.pad(q_r, [(0, pad_width), (0, 0), (0, 0)])
    k_r = jnp.pad(k_r, [(0, pad_width), (0, 0), (0, 0)])
    v_r = jnp.pad(v_r, [(0, pad_width), (0, 0), (0, 0)])
    gk_r = jnp.pad(gk_r, [(0, pad_width), (0, 0), (0, 0)])
    beta_r = jnp.pad(beta_r, [(0, pad_width), (0, 0)])
    A_r = jnp.pad(A_r, [(0, pad_width), (0, 0), (0, 0)])

  def _spec3(d1, d2):
    return pl.BlockSpec(block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0))

  in_specs = [
    _spec3(BT, K),   # k
    _spec3(BT, V),   # v
    pl.BlockSpec(block_shape=(MB, BT), index_map=lambda idx: (idx, 0)),
    _spec3(BT, BT),  # A
    _spec3(BT, K),   # q
    _spec3(BT, K),   # gk
  ]

  out_specs = [
    _spec3(BT, V),   # u
    _spec3(BT, K),   # w
    _spec3(BT, K),   # qg
    _spec3(BT, K),   # kg
  ]

  out_shape = [
    jax.ShapeDtypeStruct((total_padded, BT, V), k.dtype),   # u
    jax.ShapeDtypeStruct((total_padded, BT, K), k.dtype),   # w
    jax.ShapeDtypeStruct((total_padded, BT, K), k.dtype),   # qg
    jax.ShapeDtypeStruct((total_padded, BT, K), k.dtype),   # kg
  ]

  kernel_fn = partial(
    _recompute_w_u_fwd_kernel,
    BT=BT, K=K, V=V, MB=MB,
  )

  interpret = get_interpret()

  u_r, w_r, qg_r, kg_r = pl.pallas_call(
    kernel_fn,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(total_padded // MB,),
      in_specs=in_specs,
      out_specs=out_specs,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=interpret,
  )(k_r, v_r, beta_r, A_r, q_r, gk_r)

  # unpad padded chunks if any, then reshape [HB*NT, BT, X] -> [H, B, T, X]
  if need_pad:
    w_r = w_r[:total]
    u_r = u_r[:total]
    qg_r = qg_r[:total]
    kg_r = kg_r[:total]

  def _ir(x, d):
    return x.reshape(H, B, T, d)

  return (
    _ir(w_r, K),
    _ir(u_r, V),
    _ir(qg_r, K),
    _ir(kg_r, K),
  )


# =====================================================================
# compute_v_new_from_h  —  Pallas kernel
#
# Per-chunk: v_new[t] = u[t] - w[t] @ h[t]
# Used by KDA bwd Stage 0 fast-path when the pre-update inter-chunk
# state ``h`` is saved from the forward pass (tagged residual). Each
# chunk is independent — runs in parallel across a flat grid of
# (H*B*NT // MB,) tiles, each processing MB chunks simultaneously.
# MB head-batching matches the strategy used by chunk_gated_delta_rule_fwd_h
# to maximise MXU utilisation and minimise kernel launch overhead.
# =====================================================================


def _compute_v_new_from_h_kernel(
  # Inputs (Ref)
  u_ref,
  w_ref,
  h_ref,
  # Outputs (Ref)
  v_new_ref,
):
  """Pallas kernel body for per-chunk v_new computation with MB batching.

  Processes MB (head, batch, chunk) tiles simultaneously.

  Per chunk of size BT:
      v_new = u - w @ h

  Computation in float32 for accuracy (matches the original recurrence
  kernel's accumulator at ``chunk_delta_h.py``); output cast to ``u`` dtype.

  Args:
      u_ref:     Ref[MB, BT, V]  -- intra-chunk solved values.
      w_ref:     Ref[MB, BT, K]  -- correction weights.
      h_ref:     Ref[MB, K, V]   -- pre-update inter-chunk states.
      v_new_ref: Ref[MB, BT, V]  -- output v_new.
  """
  out_dtype = v_new_ref.dtype

  u = u_ref[:]  # [MB, BT, V]
  w = w_ref[:]  # [MB, BT, K]
  h = h_ref[:]  # [MB, K, V]

  # correction = w @ h: [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]  (fp32)
  correction = jnp.matmul(
    w.astype(jnp.float32),
    h.astype(jnp.float32),
    preferred_element_type=jnp.float32,
  )  # [MB, BT, V]

  v_new = u.astype(jnp.float32) - correction
  v_new_ref[:] = v_new.astype(out_dtype)


@functools.partial(jax.jit, static_argnames=["chunk_size", "mini_batch"])
def compute_v_new_from_h_pallas(
  u: jax.Array,
  w: jax.Array,
  h: jax.Array,
  chunk_size: int = 64,
  mini_batch: int | None = None,
) -> jax.Array:
  """Pallas TPU implementation of v_new = u - w @ h, per chunk.

  Uses MB-batched flat grid ``(H*B*NT // MB,)`` to maximise MXU utilisation
  and reduce kernel launch overhead — matching the strategy used by
  ``chunk_gated_delta_rule_fwd_h``.

  Used by KDA bwd Stage 0 when ``h`` is saved from forward (tagged residual)
  to avoid the sequential ``chunk_gated_delta_rule_fwd_h`` recurrence.

  Padded chunks (varlen): ``w``/``u`` are zero at padded positions so
  ``w @ h_pad = 0`` and ``v_new_pad = u_pad = 0`` regardless of saved
  ``h`` content at those slots.

  Args:
      u: [H, B, T, V]     -- intra-chunk solved values.
      w: [H, B, T, K]     -- correction weights.
      h: [H, B, NT, K, V] -- pre-update inter-chunk hidden states from fwd.
      chunk_size: int      -- tile/chunk size (BT). T must be divisible by BT.
      mini_batch: int or None. Number of (head, batch, chunk) tiles per grid
          point. None = auto-compute to maximise VMEM utilisation.

  Returns:
      v_new: [H, B, T, V] -- delta-corrected values, dtype = u.dtype.
  """
  H, B, T, K = w.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = T // BT
  total = H * B * NT  # flat tile count

  assert_shape(u, (H, B, T, V), "u")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(h, (H, B, NT, K, V), "h")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = 2 if u.dtype == jnp.bfloat16 else 4
    # per tile: u[BT,V] + w[BT,K] + h[K,V] + v_new[BT,V]
    per_tile = (2 * BT * V + BT * K + K * V) * elem_size
    vmem_budget = get_tpu_config().vmem_hw_limit_bytes
    MB = max(1, vmem_budget // per_tile)
    MB = min(MB, total, 32)
    while total % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"
  # Flatten: [H, B, NT, BT, X] -> [total, BT, X]
  u_r = u.reshape(total, BT, V)
  w_r = w.reshape(total, BT, K)
  # h: [H, B, NT, K, V] -> [total, K, V]
  h_r = h.reshape(total, K, V)

  def _spec(d1, d2):
    return pl.BlockSpec(block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0))

  v_new_r = pl.pallas_call(
    _compute_v_new_from_h_kernel,
    interpret=get_interpret(),
    out_shape=[
      jax.ShapeDtypeStruct(shape=(total, BT, V), dtype=u.dtype),
    ],
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(total // MB,),
      in_specs=[
        _spec(BT, V),  # u
        _spec(BT, K),  # w
        _spec(K, V),   # h
      ],
      out_specs=[
        _spec(BT, V),  # v_new
      ],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
  )(u_r, w_r, h_r)

  # pallas_call returns a list when out_shape is a list of length 1
  v_new_r = v_new_r[0] if isinstance(v_new_r, (list, tuple)) else v_new_r

  # Reshape back: [total, BT, V] -> [H, B, T, V]
  return v_new_r.reshape(H, B, T, V)



# =============================================================================
# KDA backward fusion kernels
# =============================================================================

"""Fused kernel implementations for KDA backward pass.

Each fusion eliminates HBM round-trips by merging adjacent kernels into a
single Pallas program.
"""


import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
    align_up,
    assert_shape,
    assert_shape_or_none,
    get_interpret,
    get_tpu_config,
)


def _flatten_batch_segment_ids(segment_ids_2d):
  """Flatten [B, T] segment_ids into global [B*T] with re-indexed IDs.

  Each batch element's segment IDs are offset so they don't collide
  across batches.  Padding (0) stays 0.

  Example:
      batch 0: [1,1,2,2,0,...] → [1,1,2,2,0,...]
      batch 1: [1,1,1,2,0,...] → [3,3,3,4,0,...]
  """
  max_per_batch = jnp.max(segment_ids_2d, axis=1)
  offsets = jnp.concatenate([
    jnp.zeros(1, dtype=jnp.int32),
    jnp.cumsum(max_per_batch)[:-1],
  ])
  valid = segment_ids_2d > 0
  return jnp.where(valid, segment_ids_2d + offsets[:, None], 0).reshape(-1)


def _dense_segment_ids(B, T):
  """Build dense segment IDs: one 1-indexed segment per batch row."""
  return jnp.repeat(jnp.arange(B, dtype=jnp.int32) + 1, T)


def _chunk_segment_metadata(chunk_seg_ids, batch_idx, chunk_id, NT):
  """Return segment metadata for one chunk from compressed per-chunk IDs.

  chunk_seg_ids: [B, NT] int32 — ``segment_ids.reshape(B, NT, BT)[:, :, 0]``.
  batch_idx: which batch element (0..B-1).
  chunk_id: chunk index within that batch (0..NT-1).
  NT: per-batch chunk count (T // BT).
  Returns: (seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk).
  """
  chunk_id = jnp.asarray(chunk_id, dtype=jnp.int32)
  batch_idx = jnp.asarray(batch_idx, dtype=jnp.int32)

  seg_cur = chunk_seg_ids[batch_idx, chunk_id]
  is_valid = seg_cur != 0

  prev_seg = jnp.where(chunk_id == 0, jnp.int32(0),
                        chunk_seg_ids[batch_idx, jnp.maximum(chunk_id - 1, 0)])
  next_seg = jnp.where(chunk_id + 1 >= NT, jnp.int32(0),
                        chunk_seg_ids[batch_idx, jnp.minimum(chunk_id + 1, NT - 1)])

  is_first_chunk = is_valid & (prev_seg != seg_cur)
  is_last_chunk = is_valid & (next_seg != seg_cur)
  seq_idx = jnp.maximum(seg_cur - 1, 0).astype(jnp.int32)
  return seg_cur, seq_idx, is_valid, is_first_chunk, is_last_chunk

# =====================================================================
# M1: recompute_w_u_fwd + compute_v_new_from_h
# =====================================================================

@partial(jax.jit, static_argnames=["dtype"])
def compute_m1_recompute(bq, bk, bv, bb, bA, bg, bh, dtype):
  """(u, w, v_new, qg, kg) from saved h.  Set truncate=False for full-recompute."""
  g_exp = jnp.exp2(bg)
  v_beta = bv * bb[:, :, None]
  u = jnp.matmul(bA.astype(jnp.float32), v_beta.astype(jnp.float32), preferred_element_type=jnp.float32)
  w = jnp.matmul(bA.astype(jnp.float32), (bk * bb[:, :, None] * g_exp).astype(jnp.float32),
                 preferred_element_type=jnp.float32)
  u_mat = u.astype(dtype)
  w_mat = w.astype(dtype)
  v_new = u_mat.astype(jnp.float32) - jnp.matmul(w_mat.astype(jnp.float32), bh.astype(jnp.float32), preferred_element_type=jnp.float32)
  qg = bq * g_exp
  kg = bk * jnp.exp2(bg[:, -1:, :] - bg)
  return u, w, v_new, qg, kg

def compute_dhu_recurrence(bkg, dh, bdv0, dh_tmp, g_exp_last, bqg, bw, bdo, scale):
  bdv = jnp.matmul(bkg, dh, preferred_element_type=jnp.float32) + bdv0
  dh_new = dh_tmp * g_exp_last[:, :, None] + jnp.matmul(
    jnp.concatenate([bqg * scale, -bw], axis=1).transpose(0, 2, 1),
    jnp.concatenate([bdo.astype(jnp.float32), bdv], axis=1),
    preferred_element_type=jnp.float32,
  )
  return bdv, dh_new


@partial(jax.jit, static_argnames=["scale", "precision"])
def compute_wy_backward(bdo, bdv, bvn, bv, bh, dh, bq, bk, bg, bb, bA, scale, precision):
  BT = bdo.shape[1]
  bh_t = bh.transpose(0, 2, 1)
  dq_acc = (jnp.matmul(bdo, bh_t, precision=precision, preferred_element_type=jnp.float32) * scale)
  b_dw = -jnp.matmul(bdv, bh_t, precision=precision, preferred_element_type=jnp.float32)
  dk_acc = jnp.matmul(bvn, dh.transpose(0, 2, 1), precision=precision, preferred_element_type=jnp.float32)
  bA_t = bA.transpose(0, 2, 1)
  b_dvb = jnp.matmul(bA_t, bdv, precision=precision, preferred_element_type=jnp.float32)
  db_acc = (b_dvb * bv).sum(axis=2)

  g_exp = jnp.exp2(bg)
  g_exp_last = g_exp[:, BT - 1, :]
  b_dgk = (bh * dh).sum(axis=2) * g_exp_last
  dq_acc = dq_acc * g_exp
  dk_acc = dk_acc * jnp.exp2(bg[:, BT - 1:BT, :] - bg)

  gb = g_exp * bb[:, :, None]
  kg_local = bk * g_exp
  dAkk_local = jnp.matmul(
    jnp.concatenate([bdv, b_dw], axis=2),
    jnp.concatenate([bv, kg_local], axis=2).transpose(0, 2, 1),
    precision=precision, preferred_element_type=jnp.float32)
  dkgb = jnp.matmul(bA_t, b_dw, precision=precision, preferred_element_type=jnp.float32)
  db_acc = db_acc + (dkgb * kg_local).sum(axis=2)

  kdk = bk * dk_acc
  b_dgk = b_dgk + kdk.sum(axis=1)
  idx = jnp.arange(BT, dtype=jnp.int32)
  m_last = (idx == BT - 1).astype(jnp.float32)
  dg_acc = (bq * dq_acc - kdk + m_last[None, :, None] * b_dgk[:, None, :] + kg_local * dkgb * bb[:, :, None])
  dk_acc = dk_acc + dkgb * gb

  m_lower = idx[:, None] > idx[None, :]
  dAkk_local = jnp.where(m_lower[None, :, :], dAkk_local * bb[:, None, :], 0.0)
  dAkk_local = jnp.matmul(dAkk_local, bA_t, precision=precision, preferred_element_type=jnp.float32)
  dAkk_local = jnp.matmul(bA_t, dAkk_local, precision=precision, preferred_element_type=jnp.float32)
  dAkk_local = jnp.where(m_lower[None, :, :], -dAkk_local, 0.0)
  return dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local


def _fused_recompute_w_u_vnew_from_h_kernel(
  k_ref,
  v_ref,
  beta_ref,
  A_ref,
  q_ref,
  g_ref,
  h_ref,
  w_ref,
  qg_ref,
  kg_ref,
  v_new_ref,
  *,
  BT,
  K,
  V,
  MB,
):
  """Compute w, qg, kg, and v_new for MB chunk tiles.

  Per chunk:
      u     = A @ (v * beta)
      w     = A @ (k * beta * exp2(g))
      qg    = q * exp2(g)
      kg    = k * exp2(g_last - g)
      v_new = u - w @ h
  """
  q = q_ref[:]  # [MB, BT, K]
  k = k_ref[:]  # [MB, BT, K]
  v = v_ref[:]  # [MB, BT, V]
  beta = beta_ref[:]  # [MB, BT]
  A = A_ref[:]  # [MB, BT, BT]
  g = g_ref[:]  # [MB, BT, K]
  h = h_ref[:]  # [MB, K, V]

  u, w, v_new, qg, kg = compute_m1_recompute(q, k, v, beta, A, g, h, k_ref.dtype)

  w_ref[:] = w.astype(w_ref.dtype)
  qg_ref[:] = qg.astype(qg_ref.dtype)
  kg_ref[:] = kg.astype(kg_ref.dtype)
  v_new_ref[:] = v_new.astype(v_new_ref.dtype)


@partial(jax.jit, static_argnames=["chunk_size", "mini_batch"])
def fused_recompute_w_u_vnew_from_h_pallas(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  beta: jax.Array,
  A: jax.Array,
  g: jax.Array,
  h: jax.Array,
  chunk_size: int = 64,
  mini_batch: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Fuse recompute w/qg/kg + compute v_new into one Pallas kernel.

  Eliminates the HBM round-trip for ``u`` by computing it internally
  and discarding it after deriving ``v_new``.

  Args:
      q:    [H, B, T, K] query vectors.
      k:    [H, B, T, K] key vectors.
      v:    [H, B, T, V] value vectors.
      beta: [H, B, T] per-token beta.
      A:    [H, B, T, BT] Akk inverse matrix.
      g:    [H, B, T, K] chunk-local cumsum gate in log2 space.
      h:    [H, B, NT, K, V] saved pre-update hidden states.
      chunk_size: Chunk size BT.
      mini_batch: Number of flattened chunk tiles per Pallas program.

  Returns:
      w:      [H, B, T, K]
      qg:     [H, B, T, K]
      kg:     [H, B, T, K]
      v_new:  [H, B, T, V]
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  total = H * B * NT
  total_orig = total  # saved for unpadding before _ir reshape

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(A, (H, B, T, BT), "A")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(h, (H, B, NT, K, V), "h")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

  def _r(x, d):
    return x.reshape(total, BT, d)

  q_r = _r(q, K)
  k_r = _r(k, K)
  v_r = _r(v, V)
  g_r = _r(g, K)
  beta_r = beta.reshape(total, BT)
  A_r = A.reshape(-1, BT, BT)
  h_r = h.reshape(total, K, V)

  align_minor = get_tpu_config().block_align_minor
  if mini_batch is None:
    elem_size = 2 if v.dtype == jnp.bfloat16 else 4
    in_bytes = (3 * BT * K + BT * V + BT + BT * BT + K * V) * elem_size  # beta: BT bytes (2D)
    out_bytes = (3 * BT * K + BT * V) * elem_size
    per_chunk = in_bytes + out_bytes
    vmem_budget = 8 * 1024 * 1024
    hw_mb = max(1, vmem_budget // per_chunk)
    MB = min(hw_mb, total)
    # align upward to block_align_minor (TPU dim=-2 constraint)
    MB = min(align_up(MB, align_minor), total)
    # try to make total divisible by MB; reduce MB first, pad as fallback
    need_pad = False
    if total % MB != 0:
      mb_try = MB
      while mb_try > 1:
        mb_try -= 1
        if total % mb_try == 0 and (mb_try == total or mb_try % align_minor == 0):
          MB = mb_try
          break
      else:
        need_pad = True
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"
    assert MB % align_minor == 0 or MB == total, (
      f"mini_batch={MB} must be divisible by {align_minor} or equal to total={total}"
    )
    need_pad = False

  # ---- pad if needed: round total up to a multiple of MB ----
  if need_pad:
    total_padded = ((total + MB - 1) // MB) * MB
    pad_width = total_padded - total
    q_r = jnp.pad(q_r, [(0, pad_width), (0, 0), (0, 0)])
    k_r = jnp.pad(k_r, [(0, pad_width), (0, 0), (0, 0)])
    v_r = jnp.pad(v_r, [(0, pad_width), (0, 0), (0, 0)])
    g_r = jnp.pad(g_r, [(0, pad_width), (0, 0), (0, 0)])
    beta_r = jnp.pad(beta_r, [(0, pad_width), (0, 0)])
    A_r = jnp.pad(A_r, [(0, pad_width), (0, 0), (0, 0)])
    h_r = jnp.pad(h_r, [(0, pad_width), (0, 0), (0, 0)])
    total = total_padded

  def _spec2(d1, d2=None):
    if d2 is None:
      return pl.BlockSpec(block_shape=(MB, d1), index_map=lambda idx: (idx, 0))
    return pl.BlockSpec(block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0))

  in_specs = [
    _spec2(BT, K),  # k
    _spec2(BT, V),  # v
    _spec2(BT),     # beta  ← 2D [MB, BT]
    _spec2(BT, BT),  # A
    _spec2(BT, K),  # q
    _spec2(BT, K),  # g
    _spec2(K, V),  # h
  ]
  out_specs = [
    _spec2(BT, K),  # w
    _spec2(BT, K),  # qg
    _spec2(BT, K),  # kg
    _spec2(BT, V),  # v_new
  ]
  out_shape = [
    jax.ShapeDtypeStruct((total, BT, K), q.dtype),
    jax.ShapeDtypeStruct((total, BT, K), q.dtype),
    jax.ShapeDtypeStruct((total, BT, K), q.dtype),
    jax.ShapeDtypeStruct((total, BT, V), v.dtype),
  ]

  kernel = partial(
    _fused_recompute_w_u_vnew_from_h_kernel,
    BT=BT,
    K=K,
    V=V,
    MB=MB,
  )

  w_r, qg_r, kg_r, v_new_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(total // MB,),
      in_specs=in_specs,
      out_specs=out_specs,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=get_interpret(),
  )(k_r, v_r, beta_r, A_r, q_r, g_r, h_r)

  def _ir(x, d):
    return x[:total_orig].reshape(H, B, T, d)

  return (
    _ir(w_r, K),
    _ir(qg_r, K),
    _ir(kg_r, K),
    _ir(v_new_r, V),
  )


# =====================================================================
# M2: intra backward + chunk-local reverse cumsum
# =====================================================================


def _fused_intra_bwd_cumsum_kernel(
  q_ref,
  k_ref,
  g_ref,
  beta_ref,
  dAqk_ref,
  dAkk_ref,
  dq_in_ref,
  dk_in_ref,
  db_in_ref,
  dg_in_ref,
  dq_ref,
  dk_ref,
  db_ref,
  dg_ref,
  *,
  chunk_size,
  head_dim,
  sub_chunk_size,
  mini_batch,
  scale,
):
  """Compute intra backward updates and reverse cumsum dg in one kernel body."""
  BC = sub_chunk_size
  NC = chunk_size // BC
  K = head_dim
  dtype = q_ref.dtype
  MB = mini_batch
  # fp32 inputs: restore HIGHEST precision to avoid MXU bf16 internal rounding
  # that causes catastrophic-cancellation errors in dg when gates→0.
  # bf16 inputs: default precision is fine (bwd_atol is already 5e-2).
  _prec = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else None

  # Load full chunk data — [MB, C, K] / [MB, C, 1]
  q = q_ref[:, 0, 0].astype(jnp.float32)  # [MB, C, K]
  k = k_ref[:, 0, 0].astype(jnp.float32)  # [MB, C, K]
  g = g_ref[:, 0, 0].astype(jnp.float32)  # [MB, C, K]
  beta = beta_ref[:, 0, 0].astype(jnp.float32)  # [MB, C, 1]

  dAqk_full = dAqk_ref[:, 0, 0]  # [MB, C, C]
  dAkk_full = dAkk_ref[:, 0, 0]  # [MB, C, C]

  idx = jnp.arange(chunk_size, dtype=jnp.int32)
  causal_mask = idx[:, None] >= idx[None, :]  # [C, C]
  strict_lower = idx[:, None] > idx[None, :]  # [C, C]
  cumsum_mask = (idx[:, None] <= idx[None, :]).astype(jnp.float32)  # [C, C]

  dAqk_full = jnp.where(causal_mask[None], dAqk_full, 0.0).astype(jnp.float32) * scale
  dAkk_full = jnp.where(strict_lower[None], dAkk_full, 0.0).astype(jnp.float32)

  # ── Block-structured fully-vectorised computation ─────────────────────────
  # Design rationale:
  #
  # BEFORE: for i_i in range(NC): for i_j in range(...): _dot_batch(...)
  #   O(NC²) serial Python-level HLO Dot dispatches (NC=4 → 40 Dot ops).
  #   XLA/MXU sees them as independent serial matmuls, no cross-dispatch fusion.
  #
  # AFTER (this implementation): reshape [MB,C,D]→[NC,MB,BC,D] and use
  #   (NC,MB)-batched dot_general for diagonal, scatter-add over pair stacks
  #   for off-diagonal.  Total HLO Dot ops = 8, constant regardless of NC.
  #   Dispatch count: NC=1→4, NC=2→8, NC=4→8  (was 4/12/40).
  #
  # The throughput gain is entirely from this vectorisation over the NC
  # dimension.  disable_recompute (saves fwd HBM re-reads) is a separate
  # orthogonal improvement applied at the chunk_kda_bwd orchestrator level.
  #
  # With auto sub_chunk_size=min(64,BT): BT=64→NC=1 (4 dispatches total).
  def _to_blocks(x):
    # [MB, C, D] → [NC, MB, BC, D]  via jnp.stack over NC slices
    return jnp.stack([x[:, i * BC : (i + 1) * BC] for i in range(NC)])

  q_b = _to_blocks(q)  # [NC, MB, BC, K]
  k_b = _to_blocks(k)
  g_b = _to_blocks(g)
  beta_b = _to_blocks(beta)  # [NC, MB, BC, 1]

  # ── Helper: flatten [NC, MB, BC, D] → [MB, C, D] ─────────────────────────
  def _from_blocks(x):
    if NC == 1:
      return x[0]
    return jnp.concatenate([x[i] for i in range(NC)], axis=1)

  # ── 1. ALL diagonal blocks via merged NM=NC*MB single-batch matmul ───────
  # Stack NC diagonal tiles of dAqk/dAkk: [NC, MB, BC, BC]
  dAqk_diag = jnp.stack(
    [dAqk_full[:, i * BC : (i + 1) * BC, i * BC : (i + 1) * BC] for i in range(NC)]
  )
  dAkk_diag = jnp.stack(
    [dAkk_full[:, i * BC : (i + 1) * BC, i * BC : (i + 1) * BC] for i in range(NC)]
  )

  g_max = jnp.max(g_b, axis=2, keepdims=True)  # [NC, MB, 1, K]
  row_d = jnp.exp2(g_b - g_max)  # [NC, MB, BC, K]  in (0,1], no overflow
  col_d = jnp.exp2(g_max - g_b)  # [NC, MB, BC, K]  >= 1; safe when BC*|lb| < 127

  k_til = k_b * col_d  # [NC, MB, BC, K]
  q_hat = q_b * row_d
  k_hat = k_b * beta_b * row_d

  # Mosaic (TPU Pallas) requires exactly 1 batch dim in dot_general.
  # Merge NC and MB into a single batch axis NM to avoid the 2-batch-dim
  # constraint: [NC, MB, M, K] → [NM, M, K].
  NM = NC * MB
  # _b1:  A @ B   i.e. contract last dim of A with second-to-last of B
  _b1 = (((2,), (1,)), ((0,), (0,)))
  # _b1t: A^T @ B i.e. contract second-to-last of A with second-to-last of B
  _b1t = (((1,), (1,)), ((0,), (0,)))

  def _f(x):
    return x.reshape(NM, x.shape[2], x.shape[3])

  dq_all = (
    _f(row_d)
    * jax.lax.dot_general(
      _f(dAqk_diag), _f(k_til), _b1, preferred_element_type=jnp.float32, precision=_prec
    )
  ).reshape(NC, MB, BC, K)  # [NC,MB,BC,K]

  dk_row_pre_all = (
    _f(row_d)
    * jax.lax.dot_general(
      _f(dAkk_diag), _f(k_til), _b1, preferred_element_type=jnp.float32, precision=_prec
    )
  ).reshape(NC, MB, BC, K)

  dk_col_all = (
    _f(col_d)
    * (
      jax.lax.dot_general(
        _f(dAqk_diag),
        _f(q_hat),
        _b1t,
        preferred_element_type=jnp.float32,
        precision=_prec,
      )
      + jax.lax.dot_general(
        _f(dAkk_diag),
        _f(k_hat),
        _b1t,
        preferred_element_type=jnp.float32,
        precision=_prec,
      )
    )
  ).reshape(NC, MB, BC, K)

  # ── 2. ALL off-diagonal row pairs (i_j < i_i) ────────────────────────────
  # Pairs: (i_i, i_j) for all i_i > i_j — NC*(NC-1)/2 total
  # Matmul: [N_row, MB, BC, BC] @ [N_row, MB, BC, K]
  #   batch=MB only (keep N_row for scatter-add), contract=BC
  #   einsum 'pmab,pmbk->pmak'  → [N_row, MB, BC, K]
  # Scatter-add results to i_i destination blocks.
  row_pairs = [(ii, ij) for ii in range(NC) for ij in range(ii)]
  if row_pairs:
    Aqk_rp = jnp.stack(
      [
        dAqk_full[:, ii * BC : (ii + 1) * BC, ij * BC : (ij + 1) * BC]
        for (ii, ij) in row_pairs
      ]
    )  # [N_row,MB,BC,BC]
    Akk_rp = jnp.stack(
      [
        dAkk_full[:, ii * BC : (ii + 1) * BC, ij * BC : (ij + 1) * BC]
        for (ii, ij) in row_pairs
      ]
    )
    k_rp = jnp.stack([k_b[ij] for (_, ij) in row_pairs])  # [N_row,MB,BC,K]
    g_rp = jnp.stack([g_b[ij] for (_, ij) in row_pairs])
    # reference = first row of i_i block
    g_ref_rp = jnp.stack(
      [g_b[ii, :, 0:1, :] for (ii, _) in row_pairs]
    )  # [N_row,MB,1,K]

    k_dec_rp = k_rp * jnp.exp2(g_ref_rp - g_rp)  # [N_row,MB,BC,K]

    # Merge N_row*MB for single-batch dot_general.
    # einsum 'pmab,pmbk->pmak': contract last dim of A with dim2 of B.
    N_row = len(row_pairs)
    NR = N_row * MB
    dq_rp = jax.lax.dot_general(
      Aqk_rp.reshape(NR, BC, BC),
      k_dec_rp.reshape(NR, BC, K),
      _b1,
      preferred_element_type=jnp.float32,
      precision=_prec,
    ).reshape(N_row, MB, BC, K)
    dkpre_rp = jax.lax.dot_general(
      Akk_rp.reshape(NR, BC, BC),
      k_dec_rp.reshape(NR, BC, K),
      _b1,
      preferred_element_type=jnp.float32,
      precision=_prec,
    ).reshape(N_row, MB, BC, K)

    # Accumulate per-block using Python list to avoid .at[ii].add() on the
    # NC-indexed leading dimension — Pallas on TPU may not handle scatter
    # on that dimension correctly.  Use input tensors (q_b/k_b) to derive
    # zero initializers, matching the main-branch pattern.
    dq_row_acc_list = [q_b[ii] * 0.0 for ii in range(NC)]  # NC×[MB,BC,K]
    dkpre_row_acc_list = [k_b[ii] * 0.0 for ii in range(NC)]
    for p_idx, (ii, _) in enumerate(row_pairs):
      dq_row_acc_list[ii] = dq_row_acc_list[ii] + dq_rp[p_idx]
      dkpre_row_acc_list[ii] = dkpre_row_acc_list[ii] + dkpre_rp[p_idx]

    # per-block row decay: exp2(g_b[i] - g_b[i, :, 0:1, :])
    row_decay = jnp.exp2(g_b - g_b[:, :, 0:1, :])  # [NC,MB,BC,K]
    dq_all = jnp.stack(
      [dq_all[ii] + row_decay[ii] * dq_row_acc_list[ii] for ii in range(NC)]
    )
    dk_row_pre_all = jnp.stack(
      [dk_row_pre_all[ii] + row_decay[ii] * dkpre_row_acc_list[ii] for ii in range(NC)]
    )

  # ── 3. ALL off-diagonal col pairs (i_j > i_i) ────────────────────────────
  # Pairs: (i_i, i_j) for all i_j > i_i
  # dA_ji^T matmul: [N_col, MB, BC_j, BC_i] contract BC_j with [N_col, MB, BC_j, K]
  #   einsum 'pmba,pmbk->pmak'  batch=MB only  → [N_col, MB, BC_i, K]
  col_pairs = [(ii, ij) for ii in range(NC) for ij in range(ii + 1, NC)]
  if col_pairs:
    # dA_ji slices: row=j, col=i
    Aqk_cp = jnp.stack(
      [
        dAqk_full[:, ij * BC : (ij + 1) * BC, ii * BC : (ii + 1) * BC]
        for (ii, ij) in col_pairs
      ]
    )  # [N_col,MB,BC_j,BC_i]
    Akk_cp = jnp.stack(
      [
        dAkk_full[:, ij * BC : (ij + 1) * BC, ii * BC : (ii + 1) * BC]
        for (ii, ij) in col_pairs
      ]
    )
    q_cp = jnp.stack([q_b[ij] for (_, ij) in col_pairs])  # [N_col,MB,BC,K]
    k_cp = jnp.stack([k_b[ij] for (_, ij) in col_pairs])
    g_cp = jnp.stack([g_b[ij] for (_, ij) in col_pairs])
    beta_cp = jnp.stack([beta_b[ij] for (_, ij) in col_pairs])  # [N_col,MB,BC,1]
    # reference = last row of i_i block
    g_ref_cp = jnp.stack(
      [g_b[ii, :, BC - 1 : BC, :] for (ii, _) in col_pairs]
    )  # [N_col,MB,1,K]

    row_d_cp = jnp.exp2(g_cp - g_ref_cp)  # [N_col,MB,BC,K]
    q_dec_cp = q_cp * row_d_cp
    kb_dec_cp = k_cp * beta_cp * row_d_cp

    # Merge N_col*MB for single-batch dot_general.
    # dA_ji^T @ q_dec: contract BC_j (dim1 after flatten) from both.
    N_col = len(col_pairs)
    NCP = N_col * MB
    dk_col_cp = (
      jax.lax.dot_general(
        Aqk_cp.reshape(NCP, BC, BC),
        q_dec_cp.reshape(NCP, BC, K),
        _b1t,
        preferred_element_type=jnp.float32,
        precision=_prec,
      )
      + jax.lax.dot_general(
        Akk_cp.reshape(NCP, BC, BC),
        kb_dec_cp.reshape(NCP, BC, K),
        _b1t,
        preferred_element_type=jnp.float32,
        precision=_prec,
      )
    ).reshape(N_col, MB, BC, K)  # [N_col,MB,BC_i,K]

    # Accumulate per-block using Python list (same reason as row_pairs).
    dk_col_acc_list = [k_b[ii] * 0.0 for ii in range(NC)]  # NC×[MB,BC,K]
    for p_idx, (ii, _) in enumerate(col_pairs):
      dk_col_acc_list[ii] = dk_col_acc_list[ii] + dk_col_cp[p_idx]

    col_decay = jnp.exp2(g_b[:, :, BC - 1 : BC, :] - g_b)  # [NC,MB,BC,K]
    dk_col_all = jnp.stack(
      [dk_col_all[ii] + col_decay[ii] * dk_col_acc_list[ii] for ii in range(NC)]
    )

  # ── Flatten [NC, MB, BC, K] → [MB, C, K] and compute final outputs ───────

  dq_out = _from_blocks(dq_all)  # [MB, C, K]
  dk_row_pre_out = _from_blocks(dk_row_pre_all)
  dk_col_out = _from_blocks(dk_col_all)

  dbeta_out = jnp.sum(
    k * dk_row_pre_out,
    axis=-1,
    keepdims=True,
  )  # [MB, C, 1]
  dk_row = beta * dk_row_pre_out  # [MB, C, K]
  dk_out = dk_row + dk_col_out  # [MB, C, K]
  dg_out = q * dq_out + k * (dk_row - dk_col_out)  # [MB, C, K]

  dq_total = dq_in_ref[:, 0, 0].astype(jnp.float32) + dq_out
  dk_total = dk_in_ref[:, 0, 0].astype(jnp.float32) + dk_out
  db_total = db_in_ref[:, 0, 0].astype(jnp.float32) + dbeta_out
  # Truncate dg to input dtype to match the original pipeline where
  # Stage 4 writes dg in k.dtype before Stage 5 cumsum reads it back.
  # Truncate the SUM of WY dg and intra dg to match original behavior where
  # both Stage 3 (WY) and Stage 4 (intra) produce k.dtype dg.
  dg_total = (dg_in_ref[:, 0, 0] + dg_out).astype(q_ref.dtype).astype(jnp.float32)

  dg_reverse_cumsum = jax.lax.dot_general(
    cumsum_mask,
    dg_total,
    (((1,), (1,)), ((), ())),
    precision=jax.lax.Precision.HIGHEST,
  ).transpose(1, 0, 2)

  dq_ref[:, 0, 0] = dq_total.astype(dq_ref.dtype)
  dk_ref[:, 0, 0] = dk_total.astype(dk_ref.dtype)
  db_ref[:, 0, 0] = db_total.astype(db_ref.dtype)
  dg_ref[:, 0, 0] = dg_reverse_cumsum.astype(dg_ref.dtype)


@partial(jax.jit, static_argnames=["chunk_size", "scale", "mini_batch"])
def fused_intra_bwd_cumsum_pallas(
  q: jax.Array,
  k: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  dAqk: jax.Array,
  dAkk: jax.Array,
  dq: jax.Array,
  dk: jax.Array,
  db: jax.Array,
  dg: jax.Array,
  chunk_size: int = 64,
  scale: float = 1.0,
  mini_batch: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Fuse KDA bwd intra updates with chunk-local reverse cumsum of ``dg``.

  Args:
      q:    [H, B, T, K] query vectors.
      k:    [H, B, T, K] key vectors.
      g:    [H, B, T, K] chunk-local cumsum gates in log2 space.
      beta: [H, B, T] per-token beta.
      dAqk: [H, B, T, BT] gradient of Aqk attention matrix.
      dAkk: [H, B, T, BT] gradient of Akk matrix.
      dq:   [H, B, T, K] incoming dq.
      dk:   [H, B, T, K] incoming dk.
      db:   [H, B, T] incoming db.
      dg:   [H, B, T, K] incoming dg.
      chunk_size: Chunk size BT.
      scale: Attention scale applied to dAqk (default 1.0, matching
          the orchestrator convention where dAqk already embeds scale).
      mini_batch: Number of heads per Pallas program.

  Returns:
      dq, dk, db, dg_raw in [H, B, T, *] layout, where dg_raw is the
      chunk-local reverse cumsum of incoming plus intra ``dg``.
  """
  H, B, T, K = q.shape
  BT = chunk_size
  NT = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(dAqk, (H, B, T, BT), "dAqk")
  assert_shape(dAkk, (H, B, T, BT), "dAkk")
  assert_shape(dq, (H, B, T, K), "dq")
  assert_shape(dk, (H, B, T, K), "dk")
  assert_shape(db, (H, B, T), "db")
  assert_shape(dg, (H, B, T, K), "dg")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  BC = min(16, BT)
  assert BT % BC == 0, f"chunk_size={BT} must be divisible by sub_chunk_size={BC}"

  if mini_batch is None:
    per_head_bytes = (6 * BT * K + 2 * BT * BT) * 4
    vmem_budget = 8 * 1024 * 1024
    MB = max(1, vmem_budget // per_head_bytes)
    MB = min(MB, H, 16)
    while MB > 1 and H % MB != 0:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  q_r = q.reshape(H, B, NT, BT, K)
  k_r = k.reshape(H, B, NT, BT, K)
  g_r = g.reshape(H, B, NT, BT, K)
  beta_r = beta.reshape(H, B, NT, BT, 1)
  dAqk_r = dAqk.reshape(H, B, NT, BT, BT)
  dAkk_r = dAkk.reshape(H, B, NT, BT, BT)
  dq_r = dq.reshape(H, B, NT, BT, K)
  dk_r = dk.reshape(H, B, NT, BT, K)
  db_r = db.reshape(H, B, NT, BT, 1)
  dg_r = dg.reshape(H, B, NT, BT, K)

  kernel = partial(
    _fused_intra_bwd_cumsum_kernel,
    chunk_size=BT,
    head_dim=K,
    sub_chunk_size=BC,
    mini_batch=MB,
    scale=scale,
  )
  grid = (H // MB, B, NT)

  out_shape = [
    jax.ShapeDtypeStruct((H, B, NT, BT, K), dq.dtype),
    jax.ShapeDtypeStruct((H, B, NT, BT, K), dk.dtype),
    jax.ShapeDtypeStruct((H, B, NT, BT, 1), db.dtype),
    jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
  ]

  qk_spec = pl.BlockSpec(
    index_map=lambda i, j, l: (i, j, l, 0, 0),
    block_shape=(MB, 1, 1, BT, K),
  )
  b_spec = pl.BlockSpec(
    index_map=lambda i, j, l: (i, j, l, 0, 0),
    block_shape=(MB, 1, 1, BT, 1),
  )
  A_spec = pl.BlockSpec(
    index_map=lambda i, j, l: (i, j, l, 0, 0),
    block_shape=(MB, 1, 1, BT, BT),
  )

  dq_o, dk_o, db_o, dg_o = pl.pallas_call(
    kernel,
    interpret=get_interpret(),
    out_shape=out_shape,
    in_specs=[
      qk_spec,
      qk_spec,
      qk_spec,
      b_spec,
      A_spec,
      A_spec,
      qk_spec,
      qk_spec,
      b_spec,
      qk_spec,
    ],
    out_specs=[qk_spec, qk_spec, b_spec, qk_spec],
    grid=grid,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "parallel"),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
  )(q_r, k_r, g_r, beta_r, dAqk_r, dAkk_r, dq_r, dk_r, db_r, dg_r)

  return (
    dq_o.reshape(H, B, T, K),
    dk_o.reshape(H, B, T, K),
    db_o.reshape(H, B, T),
    dg_o.reshape(H, B, T, K),
  )


# =====================================================================
# M3: dhu reverse recurrence + WY dq/dk/dv/db/dg/dAkk
# =====================================================================


def _fused_dhu_wy_kernel(
  chunk_seg_ids_ref,
  q_ref,
  k_ref,
  v_ref,
  v_new_ref,
  qg_ref,
  kg_ref,
  w_ref,
  g_ref,
  beta_ref,
  A_ref,
  h_ref,
  do_ref,
  dv0_ref,
  dht_ref,
  dq_ref,
  dk_ref,
  dv_ref,
  db_ref,
  dg_ref,
  dA_ref,
  dh0_ref,
  scratch_ref,
  *,
  BT,
  K,
  V,
  NT,
  scale,
  MB,
):
  """Fuse Dhu reverse recurrence and WY backward for one reverse chunk tile."""
  rev_c = pl.program_id(1)
  chunk_id = NT - 1 - rev_c
  _, seq_idx, _, is_first_chunk, is_last_chunk = _chunk_segment_metadata(
    chunk_seg_ids_ref, chunk_id, NT,
  )
  precision = None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  @pl.when(is_last_chunk)
  def _():
    scratch_ref[:] = dht_ref[0, :].astype(scratch_ref.dtype)

  dh = scratch_ref[:].astype(jnp.float32)
  bq = q_ref[:, 0].astype(jnp.float32)
  bk = k_ref[:, 0].astype(jnp.float32)
  bv = v_ref[:, 0].astype(jnp.float32)
  bvn = v_new_ref[:, 0].astype(jnp.float32)
  bqg = qg_ref[:, 0].astype(jnp.float32)
  bkg = kg_ref[:, 0].astype(jnp.float32)
  bw = w_ref[:, 0].astype(jnp.float32)
  bg = g_ref[:, 0].astype(jnp.float32)
  bb = beta_ref[:, 0, :, 0].astype(jnp.float32)
  bA = A_ref[:, 0].astype(jnp.float32)
  bh = h_ref[:, 0].astype(jnp.float32)
  bdo = do_ref[:, 0].astype(jnp.float32)
  bdv0 = dv0_ref[:, 0].astype(jnp.float32)

  # --- dhu reverse recurrence (NO precision= to match original dhu kernel) ---
  bdv = (
    jnp.matmul(
      bkg,
      dh,
      preferred_element_type=jnp.float32,
    )
    + bdv0
  )

  scratch_ref[:] = scratch_ref[:] * jnp.exp2(bg[:, BT - 1, :, None])
  scratch_ref[:] = scratch_ref[:] + (
    jnp.matmul(
      bqg.transpose(0, 2, 1),
      bdo,
      preferred_element_type=jnp.float32,
    )
    * scale
    - jnp.matmul(
      bw.transpose(0, 2, 1),
      bdv,
      preferred_element_type=jnp.float32,
    )
  )

  bdo_scaled = bdo * scale
  do_dv = jnp.concatenate([bdo_scaled, bdv], axis=1)
  dq_dw = jnp.matmul(
    do_dv,
    bh.transpose(0, 2, 1),
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  b_dq = dq_dw[:, :BT, :]
  b_dw = -dq_dw[:, BT:, :]

  b_dk = jnp.matmul(
    bvn,
    dh.transpose(0, 2, 1),
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  b_dA = jnp.matmul(
    bdv,
    bv.transpose(0, 2, 1),
    precision=precision,
    preferred_element_type=jnp.float32,
  )

  bA_t = bA.transpose(0, 2, 1)
  b_dvb = jnp.matmul(
    bA_t,
    bdv,
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  b_db = (b_dvb * bv).sum(axis=2)

  g_exp = jnp.exp2(bg)
  b_dgk = (bh * dh).sum(axis=2) * jnp.exp2(bg[:, BT - 1, :])
  b_dq = b_dq * g_exp
  b_dk = b_dk * jnp.exp2(bg[:, BT - 1 : BT, :] - bg)

  gb = g_exp * bb[:, :, None]
  kg_local = bk * g_exp
  b_dA = b_dA + jnp.matmul(
    b_dw,
    kg_local.transpose(0, 2, 1),
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  dkgb = jnp.matmul(
    bA_t,
    b_dw,
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  b_db = b_db + (dkgb * kg_local).sum(axis=2)

  kdk = bk * b_dk
  b_dgk = b_dgk + kdk.sum(axis=1)
  o_t = jnp.arange(BT)
  m_last = (o_t == BT - 1).astype(jnp.float32)
  b_dg = (
    bq * b_dq
    - kdk
    + m_last[None, :, None] * b_dgk[:, None, :]
    + kg_local * dkgb * bb[:, :, None]
  )
  b_dk = b_dk + dkgb * gb

  m_lower = o_t[:, None] > o_t[None, :]
  b_dA = jnp.where(m_lower[None, :, :], b_dA * bb[:, None, :], 0.0)
  b_dA = jnp.matmul(
    b_dA,
    bA_t,
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  b_dA = jnp.matmul(
    bA_t,
    b_dA,
    precision=precision,
    preferred_element_type=jnp.float32,
  )
  b_dA = jnp.where(m_lower[None, :, :], -b_dA, 0.0)

  dq_ref[:, 0] = b_dq
  dk_ref[:, 0] = b_dk
  dv_ref[:, 0] = b_dvb * bb[:, :, None]
  db_ref[:, 0, :, 0] = b_db
  dg_ref[:, 0] = b_dg
  dA_ref[:, 0] = b_dA

  @pl.when(is_first_chunk)
  def _():
    dh0_ref[0, :] = scratch_ref[:].astype(dh0_ref.dtype)


@partial(
  jax.jit,
  static_argnames=["chunk_size", "use_exp2", "scale", "mini_batch", "return_dh0"],
)
def fused_dhu_wy_pallas(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  v_new: jax.Array,
  qg: jax.Array,
  kg: jax.Array,
  w: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  A: jax.Array,
  h: jax.Array,
  do: jax.Array,
  dv0: jax.Array,
  dht: jax.Array | None,
  scale: float,
  *,
  segment_ids: jax.Array | None = None,
  chunk_size: int = 64,
  use_exp2: bool = True,
  mini_batch: int | None = None,
  return_dh0: bool = True,
) -> tuple[
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array | None,
]:
  """Fuse KDA bwd Dhu and WY backward into one Pallas call.

  Args:
      q:      [H, B, T, K] original query vectors.
      k:      [H, B, T, K] original key vectors.
      v:      [H, B, T, V] original value vectors.
      v_new:  [H, B, T, V] WY-transformed values.
      qg:     [H, B, T, K] gated query from stage 0.
      kg:     [H, B, T, K] gated key from stage 0.
      w:      [H, B, T, K] erase weights from stage 0.
      g:      [H, B, T, K] chunk-local cumsum gate in log2 space.
      beta:   [H, B, T] WY beta coefficients.
      A:      [H, B, T, BT] Akk inverse matrix.
      h:      [H, B, NT, K, V] saved forward hidden states.
      do:     [H, B, T, V] output gradient.
      dv0:    [H, B, T, V] value gradient entering Dhu.
      dht:    [N, H, K, V] final-state gradient, or None.
      scale:  Attention scale.
      segment_ids: None for uniform inputs, or [T] int32 segment IDs
        (1-indexed; 0=padding).  Converted to cu_seqlens internally.
      chunk_size: Chunk size BT.
      use_exp2: Must be True. M3 expects log2 gates and uses exp2 decay.
      mini_batch: Number of heads per program.
      return_dh0: Return ``dh0`` when True; return None to match the old Dhu
        API when the caller has no initial state.

  Returns:
      dq, dk, dv, db, dg, dAkk, dh0 in head-first layout. ``dh0`` has shape
      ``[N, H, K, V]`` (one slot per global segment) when ``return_dh0`` is
      True; otherwise None.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  T_sum = B * T
  NT = T_sum // BT
  NT_per_seq = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(v_new, (H, B, T, V), "v_new")
  assert_shape(qg, (H, B, T, K), "qg")
  assert_shape(kg, (H, B, T, K), "kg")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(A, (H, B, T, BT), "A")
  assert_shape(h, (H, B, NT_per_seq, K, V), "h")
  assert_shape(do, (H, B, T, V), "do")
  assert_shape(dv0, (H, B, T, V), "dv0")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert use_exp2 is True, (
    "fused_dhu_wy_pallas currently expects log2 gates and use_exp2=True"
  )

  if segment_ids is not None:
    if segment_ids.ndim == 2:
      seg_1d = _flatten_batch_segment_ids(segment_ids)
    elif B > 1:
      # 1D shared segment_ids with B>1: broadcast to [B, T] before flattening
      # so that global IDs span all B*T positions correctly.
      seg_1d = _flatten_batch_segment_ids(
        jnp.broadcast_to(segment_ids[None, :], (B, segment_ids.shape[0]))
      )
    else:
      seg_1d = segment_ids
  else:
    seg_1d = _dense_segment_ids(B, T)
  chunk_seg_ids = seg_1d.reshape(NT, BT)[:, 0]
  if dht is not None:
    N = dht.shape[0]
  elif segment_ids is not None:
    # Compute N so dht_arr covers all global segment slots.
    # NT (= B*T/BT) is a static safe upper bound for the max global segment
    # ID after _flatten_batch_segment_ids, avoiding int(jnp.max(tracer))
    # which raises ConcretizationTypeError inside shard_map / jit contexts.
    N = NT
  else:
    N = B
  assert_shape_or_none(dht, (N, H, K, V), "dht")
  dht_arr = dht if dht is not None else jnp.zeros((N, H, K, V), dtype=jnp.float32)

  if mini_batch is None:
    elem_size = 2 if q.dtype == jnp.bfloat16 else 4
    per_head = (7 * BT * K + 4 * BT * V + BT + BT * BT + 2 * K * V) * elem_size + (
      K * V + 3 * BT * K + BT * V + BT + BT * BT + K * V
    ) * 4
    hw = get_tpu_config()
    vmem_budget = hw.vmem_limit_bytes
    MB = max(1, vmem_budget // per_head)
    MB = min(MB, H, 16)
    while H % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  q_r = q.reshape(H, NT, BT, K)
  k_r = k.reshape(H, NT, BT, K)
  v_r = v.reshape(H, NT, BT, V)
  vn_r = v_new.reshape(H, NT, BT, V)
  qg_r = qg.reshape(H, NT, BT, K)
  kg_r = kg.reshape(H, NT, BT, K)
  w_r = w.reshape(H, NT, BT, K)
  g_r = g.reshape(H, NT, BT, K)
  beta_r = beta.reshape(H, NT, BT, 1)
  A_r = A.reshape(H, NT, BT, BT)
  h_r = h.reshape(H, NT, K, V)
  do_r = do.reshape(H, NT, BT, V)
  dv0_r = dv0.reshape(H, NT, BT, V)

  def idx_chunk(head, chunk, chunk_seg_ids_ref):
    return (head, NT - 1 - chunk, 0, 0)

  def idx_state(head, chunk, chunk_seg_ids_ref):
    chunk_id = NT - 1 - chunk
    _, seq_idx, _, _, _ = _chunk_segment_metadata(chunk_seg_ids_ref, chunk_id, NT)
    return (seq_idx, head, 0, 0)

  qk_spec = pl.BlockSpec((MB, 1, BT, K), index_map=idx_chunk)
  v_spec = pl.BlockSpec((MB, 1, BT, V), index_map=idx_chunk)
  b_spec = pl.BlockSpec((MB, 1, BT, 1), index_map=idx_chunk)
  A_spec = pl.BlockSpec((MB, 1, BT, BT), index_map=idx_chunk)
  h_spec = pl.BlockSpec((MB, 1, K, V), index_map=idx_chunk)
  state_spec = pl.BlockSpec((1, MB, K, V), index_map=idx_state)

  kernel = partial(
    _fused_dhu_wy_kernel,
    scale=scale,
    BT=BT,
    K=K,
    V=V,
    NT=NT,
    MB=MB,
  )
  scratch = pltpu.VMEM((MB, K, V), jnp.float32)
  out_shape = [
    jax.ShapeDtypeStruct((H, NT, BT, K), jnp.float32),
    jax.ShapeDtypeStruct((H, NT, BT, K), jnp.float32),
    jax.ShapeDtypeStruct((H, NT, BT, V), jnp.float32),
    jax.ShapeDtypeStruct((H, NT, BT, 1), jnp.float32),
    jax.ShapeDtypeStruct((H, NT, BT, K), jnp.float32),
    jax.ShapeDtypeStruct((H, NT, BT, BT), jnp.float32),
    jax.ShapeDtypeStruct((N, H, K, V), jnp.float32),
  ]

  dq_r, dk_r, dv_r, db_r, dg_r, dA_r, dh0_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(H // MB, NT),
      in_specs=[
        qk_spec,
        qk_spec,
        v_spec,
        v_spec,
        qk_spec,
        qk_spec,
        qk_spec,
        qk_spec,
        b_spec,
        A_spec,
        h_spec,
        v_spec,
        v_spec,
        state_spec,
      ],
      out_specs=[qk_spec, qk_spec, v_spec, b_spec, qk_spec, A_spec, state_spec],
      scratch_shapes=[scratch],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "arbitrary"),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=get_interpret(),
  )(
    chunk_seg_ids,
    q_r,
    k_r,
    v_r,
    vn_r,
    qg_r,
    kg_r,
    w_r,
    g_r,
    beta_r,
    A_r,
    h_r,
    do_r,
    dv0_r,
    dht_arr,
  )

  return (
    dq_r.reshape(H, B, T, K),
    dk_r.reshape(H, B, T, K),
    dv_r.reshape(H, B, T, V),
    db_r.reshape(H, B, T),
    dg_r.reshape(H, B, T, K),
    dA_r.reshape(H, B, T, BT),
    dh0_r if return_dh0 else None,
  )


# ══════════════════════════════════════════════════════════════════════════
# Shared L1 helpers extracted from M4 / M5 kernels
# ══════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=["ref_dtype", "precision"])
def compute_intra_backward(bq, bk, bg, bb, dAqk, dAkk,
                           dq_acc, dk_acc, db_acc, dg_acc,
                           precision, ref_dtype):
  BT = bq.shape[1]
  idx = jnp.arange(BT, dtype=jnp.int32)
  causal_mask = idx[:, None] >= idx[None, :]
  dAqk_full = jnp.where(causal_mask[None], dAqk, 0.0).astype(jnp.float32)
  dAkk_full = dAkk.astype(jnp.float32)

  BC = min(16, BT)
  NC = BT // BC

  def _to_blocks(x):
    return jnp.stack([x[:, i * BC:(i + 1) * BC] for i in range(NC)])
  def _from_blocks(x):
    if NC == 1:
      return x[0]
    return jnp.concatenate([x[i] for i in range(NC)], axis=1)

  q_b = _to_blocks(bq)
  k_b = _to_blocks(bk)
  g_b = _to_blocks(bg)
  beta_b = _to_blocks(bb[:, :, None])

  dAqk_diag = jnp.stack(
    [dAqk_full[:, i * BC:(i + 1) * BC, i * BC:(i + 1) * BC] for i in range(NC)])
  dAkk_diag = jnp.stack(
    [dAkk_full[:, i * BC:(i + 1) * BC, i * BC:(i + 1) * BC] for i in range(NC)])

  g_max = jnp.max(g_b, axis=2, keepdims=True)
  row_d = jnp.exp2(g_b - g_max)
  col_d = jnp.exp2(g_max - g_b)
  k_til = k_b * col_d
  q_hat = q_b * row_d
  k_hat = k_b * beta_b * row_d

  K = bq.shape[2]
  MB = bq.shape[0]
  NM = NC * MB
  _b1 = (((2,), (1,)), ((0,), (0,)))
  _b1t = (((1,), (1,)), ((0,), (0,)))
  def _f(x):
    return x.reshape(NM, x.shape[2], x.shape[3])

  dq_all = (_f(row_d) * jax.lax.dot_general(
    _f(dAqk_diag), _f(k_til), _b1,
    preferred_element_type=jnp.float32, precision=precision,
  )).reshape(NC, MB, BC, K)
  dk_row_pre_all = (_f(row_d) * jax.lax.dot_general(
    _f(dAkk_diag), _f(k_til), _b1,
    preferred_element_type=jnp.float32, precision=precision,
  )).reshape(NC, MB, BC, K)
  dk_col_all = (_f(col_d) * (
    jax.lax.dot_general(_f(dAqk_diag), _f(q_hat), _b1t,
                        preferred_element_type=jnp.float32, precision=precision)
    + jax.lax.dot_general(_f(dAkk_diag), _f(k_hat), _b1t,
                          preferred_element_type=jnp.float32, precision=precision)
  )).reshape(NC, MB, BC, K)

  row_pairs = [(ii, ij) for ii in range(NC) for ij in range(ii)]
  if row_pairs:
    Aqk_rp = jnp.stack(
      [dAqk_full[:, ii * BC:(ii + 1) * BC, ij * BC:(ij + 1) * BC]
       for (ii, ij) in row_pairs])
    Akk_rp = jnp.stack(
      [dAkk_full[:, ii * BC:(ii + 1) * BC, ij * BC:(ij + 1) * BC]
       for (ii, ij) in row_pairs])
    k_rp = jnp.stack([k_b[ij] for (_, ij) in row_pairs])
    g_rp = jnp.stack([g_b[ij] for (_, ij) in row_pairs])
    g_ref_rp = jnp.stack([g_b[ii, :, 0:1, :] for (ii, _) in row_pairs])
    k_dec_rp = k_rp * jnp.exp2(g_ref_rp - g_rp)

    NR = len(row_pairs) * MB
    dq_rp = jax.lax.dot_general(
      Aqk_rp.reshape(NR, BC, BC), k_dec_rp.reshape(NR, BC, K),
      _b1, preferred_element_type=jnp.float32, precision=precision,
    ).reshape(len(row_pairs), MB, BC, K)
    dkpre_rp = jax.lax.dot_general(
      Akk_rp.reshape(NR, BC, BC), k_dec_rp.reshape(NR, BC, K),
      _b1, preferred_element_type=jnp.float32, precision=precision,
    ).reshape(len(row_pairs), MB, BC, K)

    dq_row_acc = [q_b[ii] * 0.0 for ii in range(NC)]
    dkpre_row_acc = [k_b[ii] * 0.0 for ii in range(NC)]
    for p_idx, (ii, _) in enumerate(row_pairs):
      dq_row_acc[ii] = dq_row_acc[ii] + dq_rp[p_idx]
      dkpre_row_acc[ii] = dkpre_row_acc[ii] + dkpre_rp[p_idx]
    row_decay = jnp.exp2(g_b - g_b[:, :, 0:1, :])
    dq_all = jnp.stack(
      [dq_all[ii] + row_decay[ii] * dq_row_acc[ii] for ii in range(NC)])
    dk_row_pre_all = jnp.stack(
      [dk_row_pre_all[ii] + row_decay[ii] * dkpre_row_acc[ii] for ii in range(NC)])

  col_pairs = [(ii, ij) for ii in range(NC) for ij in range(ii + 1, NC)]
  if col_pairs:
    Aqk_cp = jnp.stack(
      [dAqk_full[:, ij * BC:(ij + 1) * BC, ii * BC:(ii + 1) * BC]
       for (ii, ij) in col_pairs])
    Akk_cp = jnp.stack(
      [dAkk_full[:, ij * BC:(ij + 1) * BC, ii * BC:(ii + 1) * BC]
       for (ii, ij) in col_pairs])
    q_cp = jnp.stack([q_b[ij] for (_, ij) in col_pairs])
    k_cp = jnp.stack([k_b[ij] for (_, ij) in col_pairs])
    g_cp = jnp.stack([g_b[ij] for (_, ij) in col_pairs])
    beta_cp = jnp.stack([beta_b[ij] for (_, ij) in col_pairs])
    g_ref_cp = jnp.stack([g_b[ii, :, BC - 1:BC, :] for (ii, _) in col_pairs])

    row_d_cp = jnp.exp2(g_cp - g_ref_cp)
    q_dec_cp = q_cp * row_d_cp
    kb_dec_cp = k_cp * beta_cp * row_d_cp

    NCP = len(col_pairs) * MB
    dk_col_cp = (
      jax.lax.dot_general(Aqk_cp.reshape(NCP, BC, BC),
                          q_dec_cp.reshape(NCP, BC, K), _b1t,
                          preferred_element_type=jnp.float32, precision=precision)
      + jax.lax.dot_general(Akk_cp.reshape(NCP, BC, BC),
                            kb_dec_cp.reshape(NCP, BC, K), _b1t,
                            preferred_element_type=jnp.float32, precision=precision)
    ).reshape(len(col_pairs), MB, BC, K)

    dk_col_acc = [k_b[ii] * 0.0 for ii in range(NC)]
    for p_idx, (ii, _) in enumerate(col_pairs):
      dk_col_acc[ii] = dk_col_acc[ii] + dk_col_cp[p_idx]
    col_decay = jnp.exp2(g_b[:, :, BC - 1:BC, :] - g_b)
    dk_col_all = jnp.stack(
      [dk_col_all[ii] + col_decay[ii] * dk_col_acc[ii] for ii in range(NC)])

  dq_intra = _from_blocks(dq_all)
  dk_row_pre = _from_blocks(dk_row_pre_all)
  dk_col = _from_blocks(dk_col_all)
  db_intra = jnp.sum(bk * dk_row_pre, axis=-1)
  dk_row = bb[:, :, None] * dk_row_pre
  dk_intra = dk_row + dk_col
  dg_intra = bq * dq_intra + bk * (dk_row - dk_col)

  dq_total = dq_acc + dq_intra
  dk_total = dk_acc + dk_intra
  db_total = db_acc + db_intra
  dg_total = (dg_acc + dg_intra).astype(ref_dtype).astype(jnp.float32)
  return dq_total, dk_total, db_total, dg_total


@jax.jit
def compute_reverse_cumsum_dg(dg_total):
  BT = dg_total.shape[1]
  idx = jnp.arange(BT, dtype=jnp.int32)
  cumsum_mask = (idx[:, None] <= idx[None, :]).astype(jnp.float32)
  return jax.lax.dot_general(
    cumsum_mask, dg_total, (((1,), (1,)), ((), ())),
    precision=jax.lax.Precision.HIGHEST,
  ).transpose(1, 0, 2)

# =====================================================================
# M4: dhu + WY + intra backward + chunk-local reverse cumsum
# =====================================================================


def _fused_dhu_wy_intra_cumsum_kernel(
  chunk_seg_ids_ref,
  q_ref,
  k_ref,
  v_ref,
  v_new_ref,
  qg_ref,
  kg_ref,
  w_ref,
  g_ref,
  beta_ref,
  A_ref,
  h_ref,
  do_ref,
  dv0_ref,
  dAqk_ref,
  dht_ref,
  dq_ref,
  dk_ref,
  dv_ref,
  db_ref,
  dg_ref,
  dh0_ref,
  dh_tmp_ref,
  *,
  BT,
  K,
  V,
  NT,
  scale,
  MB,
):
  """Fuse Dhu, WY, intra backward, and reverse cumsum for one chunk tile."""
  head_group = pl.program_id(0)
  batch_idx = pl.program_id(1)
  rev_c = pl.program_id(2)
  chunk_id = NT - 1 - rev_c
  _, seq_idx, _, is_first_chunk, is_last_chunk = _chunk_segment_metadata(
    chunk_seg_ids_ref, batch_idx, chunk_id, NT,
  )
  precision = None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  @pl.when(is_last_chunk)
  def _():
    dh_tmp_ref[:] = dht_ref[:, 0, 0, :].astype(dh_tmp_ref.dtype)

  dh = dh_tmp_ref[:].astype(jnp.float32)
  bq = q_ref[:, 0, 0].astype(jnp.float32)
  bk = k_ref[:, 0, 0].astype(jnp.float32)
  bv = v_ref[:, 0, 0].astype(jnp.float32)
  bvn = v_new_ref[:, 0, 0].astype(jnp.float32)
  bqg = qg_ref[:, 0, 0].astype(jnp.float32)
  bkg = kg_ref[:, 0, 0]
  bw = w_ref[:, 0, 0].astype(jnp.float32)
  bg = g_ref[:, 0, 0].astype(jnp.float32)
  g_exp_last = jnp.exp2(bg[:, BT - 1, :])
  bb = beta_ref[:, 0, 0, :, 0].astype(jnp.float32)
  bA = A_ref[:, 0, 0].astype(jnp.float32)
  bh = h_ref[:, 0, 0].astype(jnp.float32)
  bdo = do_ref[:, 0, 0]
  bdv0 = dv0_ref[:, 0, 0].astype(jnp.float32)
  bdAqk = dAqk_ref[:, 0, 0].astype(jnp.float32)

  # --- dhu reverse recurrence ---
  bdv, dh_new = compute_dhu_recurrence(bkg, dh, bdv0, dh_tmp_ref[:], g_exp_last, bqg, bw, bdo, scale)
  dh_tmp_ref[:] = dh_new

  # --- WY backward ---
  dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local = compute_wy_backward(
    bdo, bdv, bvn, bv, bh, dh, bq, bk, bg, bb, bA, scale, precision)

  # --- Intra backward + reverse cumsum ---
  dq_total, dk_total, db_total, dg_total = compute_intra_backward(
    bq, bk, bg, bb, bdAqk, dAkk_local, dq_acc, dk_acc, db_acc, dg_acc,
    precision=precision, ref_dtype=q_ref.dtype,
  )
  dg_reverse_cumsum = compute_reverse_cumsum_dg(dg_total)

  dq_ref[:, 0, 0] = dq_total.astype(dq_ref.dtype)
  dk_ref[:, 0, 0] = dk_total.astype(dk_ref.dtype)
  dv_ref[:, 0, 0] = (b_dvb * bb[:, :, None]).astype(dv_ref.dtype)
  db_ref[:, 0, 0, :, 0] = db_total.astype(db_ref.dtype)
  dg_ref[:, 0, 0] = dg_reverse_cumsum.astype(dg_ref.dtype)

  @pl.when(is_first_chunk)
  def _():
    dh0_ref[:, 0, 0, :] = dh_tmp_ref[:].astype(dh0_ref.dtype)


@partial(
  jax.jit,
  static_argnames=["chunk_size", "use_exp2", "scale", "mini_batch", "return_dh0", "N_MAX"],
)
def _fused_dhu_wy_intra_cumsum_pallas_jit(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  v_new: jax.Array,
  qg: jax.Array,
  kg: jax.Array,
  w: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  A: jax.Array,
  h: jax.Array,
  do: jax.Array,
  dv0: jax.Array,
  dAqk: jax.Array,
  dht: jax.Array | None,
  scale: float,
  *,
  segment_ids: jax.Array | None = None,
  chunk_size: int = 64,
  use_exp2: bool = True,
  mini_batch: int | None = None,
  return_dh0: bool = True,
  N_MAX: int | None = None
) -> tuple[
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array | None,
]:
  """Fuse KDA bwd Dhu, WY, intra backward, and reverse cumsum.

  Args:
      q:      [H, B, T, K] original query vectors.
      k:      [H, B, T, K] original key vectors.
      v:      [H, B, T, V] original value vectors.
      v_new:  [H, B, T, V] WY-transformed values.
      qg:     [H, B, T, K] gated query from stage 0.
      kg:     [H, B, T, K] gated key from stage 0.
      w:      [H, B, T, K] erase weights from stage 0.
      g:      [H, B, T, K] chunk-local cumsum gate in log2 space.
      beta:   [H, B, T] WY beta coefficients.
      A:      [H, B, T, BT] Akk inverse matrix.
      h:      [H, B, NT, K, V] saved forward hidden states.
      do:     [H, B, T, V] output gradient.
      dv0:    [H, B, T, V] value gradient entering Dhu.
      dAqk:   [H, B, T, BT] gradient of Aqk attention matrix.
      dht:    [B, N, H, K, V] already-merged final-state gradient, or None.
      scale:  Attention scale.
      segment_ids: None for uniform inputs, or [B, T] int32 segment IDs
        (1-indexed; 0=padding).  Per-batch IDs; no global flattening.
      chunk_size: Chunk size BT.
      use_exp2: Must be True. M4 expects log2 gates and uses exp2 decay.
      mini_batch: Number of heads per program.
      return_dh0: Return ``dh0`` when True; return None otherwise.

  Returns:
      dq, dk, dv, db, dg_raw, dh0 in head-first layout. ``dg_raw`` includes
      intra updates and the chunk-local reverse cumsum. ``dh0`` has shape
      [B, N, H, K, V] when requested; otherwise None.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  HB = H * B

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(v_new, (H, B, T, V), "v_new")
  assert_shape(qg, (H, B, T, K), "qg")
  assert_shape(kg, (H, B, T, K), "kg")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(A, (H, B, T, BT), "A")
  assert_shape(h, (H, B, NT, K, V), "h")
  assert_shape(do, (H, B, T, V), "do")
  assert_shape(dv0, (H, B, T, V), "dv0")
  assert_shape(dAqk, (H, B, T, BT), "dAqk")
  is_varlen = segment_ids is not None
  assert_shape_or_none(segment_ids, (B, T), "segment_ids")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert use_exp2 is True, (
    "fused_dhu_wy_intra_cumsum_pallas currently expects log2 gates and use_exp2=True"
  )

  segment_ids = jnp.ones((B, T), dtype=jnp.int32) if not is_varlen else segment_ids
  chunk_seg_ids = segment_ids.reshape(B, NT, BT)[:, :, 0]  # [B, NT]
  if dht is not None:
    N = dht.shape[1]  # dht is [B, N, H, K, V], N is dim 1
  elif is_varlen:
    N = N_MAX
  else:
    N = 1  # uniform: one sequence per batch element
  assert_shape_or_none(dht, (B, N, H, K, V), "dht")
  dht_arr = dht if dht is not None else jnp.zeros((B, N, H, K, V), dtype=jnp.float32)

  if mini_batch is None:
    elem_size = 2 if q.dtype == jnp.bfloat16 else 4
    io_per_head = (8 * BT * K + 4 * BT * V + BT + 3 * BT * BT + 2 * K * V) * elem_size + (
      K * V + 5 * BT * K + BT * V + BT + 2 * BT * BT + K * V
    ) * 4
    per_head = io_per_head + io_per_head * 3 // 2
    hw = get_tpu_config()
    vmem_budget = hw.vmem_limit_bytes
    MB = max(1, vmem_budget // per_head)
    MB = min(MB, H, 16)
    while H % MB != 0 and MB > 1:
      MB -= 1
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  # Keep [H, B, ...] layout; reshape T → (NT, BT) only. No transpose.
  # B is an independent dimension handled by a separate grid axis.
  q_r = q.reshape(H, B, NT, BT, K)
  k_r = k.reshape(H, B, NT, BT, K)
  v_r = v.reshape(H, B, NT, BT, V)
  vn_r = v_new.reshape(H, B, NT, BT, V)
  qg_r = qg.reshape(H, B, NT, BT, K)
  kg_r = kg.reshape(H, B, NT, BT, K)
  w_r = w.reshape(H, B, NT, BT, K)
  g_r = g.reshape(H, B, NT, BT, K)
  beta_r = beta.reshape(H, B, NT, BT, 1)
  A_r = A.reshape(H, B, NT, BT, BT)
  h_r = h  # already [H, B, NT, K, V]
  do_r = do.reshape(H, B, NT, BT, V)
  dv0_r = dv0.reshape(H, B, NT, BT, V)
  dAqk_r = dAqk.reshape(H, B, NT, BT, BT)

  def idx_chunk(head_group, batch, chunk, chunk_seg_ids_ref):
    return (head_group, batch, NT - 1 - chunk, 0, 0)

  def idx_state(head_group, batch, chunk, chunk_seg_ids_ref):
    chunk_id = NT - 1 - chunk
    _, seq_idx, _, _, _ = _chunk_segment_metadata(chunk_seg_ids_ref, batch, chunk_id, NT)
    return (head_group, batch, seq_idx, 0, 0)

  # dht_arr [B, N, H, K, V] → [H, B, N, K, V]
  dht_arr = dht_arr.transpose(2, 0, 1, 3, 4)

  qk_spec = pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk)
  v_spec = pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk)
  b_spec = pl.BlockSpec((MB, 1, 1, BT, 1), index_map=idx_chunk)
  A_spec = pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_chunk)
  h_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_chunk)
  state_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_state)

  kernel = partial(
    _fused_dhu_wy_intra_cumsum_kernel,
    scale=scale,
    BT=BT,
    K=K,
    V=V,
    NT=NT,
    MB=MB,
  )
  dh_tmp = pltpu.VMEM((MB, K, V), jnp.float32)
  out_shape = [
    jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
    jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
    jax.ShapeDtypeStruct((H, B, NT, BT, V), jnp.float32),
    jax.ShapeDtypeStruct((H, B, NT, BT, 1), jnp.float32),
    jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
    jax.ShapeDtypeStruct((H, B, N, K, V), jnp.float32),
  ]

  dq_r, dk_r, dv_r, db_r, dg_r, dh0_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(H // MB, B, NT),
      in_specs=[
        qk_spec,
        qk_spec,
        v_spec,
        v_spec,
        qk_spec,
        qk_spec,
        qk_spec,
        qk_spec,
        b_spec,
        A_spec,
        h_spec,
        v_spec,
        v_spec,
        A_spec,
        state_spec,
      ],
      out_specs=[qk_spec, qk_spec, v_spec, b_spec, qk_spec, state_spec],
      scratch_shapes=[dh_tmp],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=get_interpret(),
  )(
    chunk_seg_ids,
    q_r,
    k_r,
    v_r,
    vn_r,
    qg_r,
    kg_r,
    w_r,
    g_r,
    beta_r,
    A_r,
    h_r,
    do_r,
    dv0_r,
    dAqk_r,
    dht_arr,
  )

  dh0_out = dh0_r.transpose(1, 2, 0, 3, 4) if return_dh0 else None
  return (
    dq_r.reshape(H, B, T, K),
    dk_r.reshape(H, B, T, K),
    dv_r.reshape(H, B, T, V),
    db_r.reshape(H, B, T),
    dg_r.reshape(H, B, T, K),
    dh0_out,
  )


def fused_dhu_wy_intra_cumsum_pallas(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  v_new: jax.Array,
  qg: jax.Array,
  kg: jax.Array,
  w: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  A: jax.Array,
  h: jax.Array,
  do: jax.Array,
  dv0: jax.Array,
  dAqk: jax.Array,
  dht: jax.Array | None,
  scale: float,
  *,
  segment_ids: jax.Array | None = None,
  chunk_size: int = 64,
  use_exp2: bool = True,
  mini_batch: int | None = None,
  return_dh0: bool = True,
) -> tuple[
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array,
  jax.Array | None,
]:
  """Fuse KDA bwd Dhu, WY, intra backward, and reverse cumsum.

  Supports uniform inputs and M4B chunk-aligned varlen/post-CP inputs. CP
  ``all_gather`` and ``_merge_dht`` stay outside this wrapper; ``dht`` is the
  already-merged local final-state gradient.

  Args:
      q:      [H, B, T, K] original query vectors.
      k:      [H, B, T, K] original key vectors.
      v:      [H, B, T, V] original value vectors.
      v_new:  [H, B, T, V] WY-transformed values.
      qg:     [H, B, T, K] gated query from stage 0.
      kg:     [H, B, T, K] gated key from stage 0.
      w:      [H, B, T, K] erase weights from stage 0.
      g:      [H, B, T, K] chunk-local cumsum gate in log2 space.
      beta:   [H, B, T] WY beta coefficients.
      A:      [H, B, T, BT] Akk inverse matrix.
      h:      [H, B, NT, K, V] saved forward hidden states.
      do:     [H, B, T, V] output gradient.
      dv0:    [H, B, T, V] value gradient entering Dhu.
      dAqk:   [H, B, T, BT] gradient of Aqk attention matrix.
      dht:    [B, N, H, K, V] already-merged final-state gradient, or None.
      scale:  Attention scale.
      segment_ids: None for uniform inputs, or [T] / [B, T] int32
        segment IDs (1-indexed; 0=padding) for varlen.  Per-batch IDs;
        no global flattening.
      chunk_size: Chunk size BT.
      use_exp2: Must be True. M4 expects log2 gates and uses exp2 decay.
      mini_batch: Number of heads per program.
      return_dh0: Return ``dh0`` when True; return None otherwise.

  Returns:
      dq, dk, dv, db, dg_raw, dh0 in head-first layout. ``dg_raw`` includes
      intra updates and the chunk-local reverse cumsum. ``dh0`` has shape
      [B, N, H, K, V] when requested; otherwise None.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(v_new, (H, B, T, V), "v_new")
  assert_shape(qg, (H, B, T, K), "qg")
  assert_shape(kg, (H, B, T, K), "kg")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(A, (H, B, T, BT), "A")
  assert_shape(h, (H, B, NT, K, V), "h")
  assert_shape(do, (H, B, T, V), "do")
  assert_shape(dv0, (H, B, T, V), "dv0")
  assert_shape(dAqk, (H, B, T, BT), "dAqk")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  assert use_exp2 is True, (
    "fused_dhu_wy_intra_cumsum_pallas currently expects log2 gates and use_exp2=True"
  )

  # Compute N: uniform → 1, varlen → max segment_id per batch.
  # dht shape [B, N, H, K, V] — N is dim 1.
  if dht is not None:
    N = dht.shape[1]
  elif segment_ids is not None:
    assert N_MAX is not None
    N = N_MAX
  else:
    N = 1  # uniform: one sequence per batch element

  assert_shape_or_none(dht, (B, N, H, K, V), "dht")
  return _fused_dhu_wy_intra_cumsum_pallas_jit(
    q=q,
    k=k,
    v=v,
    v_new=v_new,
    qg=qg,
    kg=kg,
    w=w,
    g=g,
    beta=beta,
    A=A,
    h=h,
    do=do,
    dv0=dv0,
    dAqk=dAqk,
    dht=dht,
    scale=scale,
    segment_ids=segment_ids,
    chunk_size=chunk_size,
    use_exp2=use_exp2,
    mini_batch=mini_batch,
    return_dh0=return_dh0,
  )



# =============================================================================
# KDA chunked backward orchestrator
# =============================================================================

import math
import jax.numpy as jnp
import jax

from tokamax._src.ops.experimental.kda.utils import (
    assert_shape,
    exp,
    exp2,
    get_tpu_config,
)
from jax.experimental import pallas as pl
from functools import partial

from tokamax._src.ops.experimental.kda.utils import (
    align_up,
    assert_shape_or_none,
    cdiv,
    get_interpret,
    segment_ids_to_seqlens,
)
from tokamax._src.ops.experimental.kda.cp_utils import CPContext, _merge_dht, all_gather_into_tensor
from jax.experimental.pallas import tpu as pltpu
import functools

RCP_LN2 = 1.0 / math.log(2)


def _chunk_kda_bwd_wy_dqkg_fused_kernel(
  # --- 11 inputs ---
  q_ref, # [total, BT, K]
  k_ref, # [total, BT, K]
  v_ref, # [total, BT, V]
  vn_ref, # [total, BT, V]
  g_ref, # [total, BT, K]
  beta_ref, # [total, 128]
  A_ref, # [total, BT, 128]
  h_ref, # [total, K, V]
  do_ref, # [total, BT, V]
  dh_ref, # [total, K, V]
  dv_ref, # [total, BT, V]
  # --- 6 outputs ---
  dq_ref, # [total, BT, K]
  dk_ref, # [total, BT, K]
  dv2_ref, # [total, BT, V]
  dg_ref, # [total, BT, K]
  db_ref, # [total, 128]
  dA_ref, # [total, 128]
  # -- 11 scratch inputs ---
  q_scratch_ref, # [2, MB, BT, K]
  k_scratch_ref, # [2, MB, BT, K]
  v_scratch_ref, # [2, MB, BT, V]
  vn_scratch_ref, # [2, MB, BT, V]
  g_scratch_ref, # [2, MB, BT, K]
  beta_scratch_ref, # [2, MB, 128]
  A_scratch_ref, # [2, MB, BT, 128]
  h_scratch_ref, # [2, MB, K, V]
  do_scratch_ref, # [2, MB, BT, V]
  dh_scratch_ref, # [2, MB, K, V]
  dv_scratch_ref, # [2, MB, BT, V]
  # --- 6 scratch outputs ---
  dq_scratch_ref, # [2, MB, BT, K]
  dk_scratch_ref, # [2, MB, BT, K]
  dv2_scratch_ref, # [2, MB, BT, V]
  dg_scratch_ref, # [2, MB, BT, K]
  db_scratch_ref, # [2, MB, 128]
  dA_scratch_ref, # [2, MB, BT, 128]
  #
  sems, # [17, 2]
  *,
  scale,
  BT,
  K,
  V,
  MB,
  MB_pad,
  compute_dq,
  compute_dk,
  compute_dv2,
  compute_dg,
  compute_db,
  compute_dA,
  BT_PAD,
):
  precision = None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  o_t = jnp.arange(BT)
  m_last = (o_t == BT - 1).astype(jnp.float32)  # [BT]
  m_lower = o_t[:, None] > o_t[None, :]  # [BT, BT]

  def _async_copy(src, dst, sem, wait=False):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

  total = q_ref.shape[0]
  N_MB = total // MB
  def start_input_dma(buf, blk_slice, beta_slice):
    """Start all 11 input async copies for a block."""
    _async_copy(q_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], q_scratch_ref.at[buf], sems.at[0,buf])
    _async_copy(k_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], k_scratch_ref.at[buf], sems.at[1,buf])
    _async_copy(v_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], v_scratch_ref.at[buf], sems.at[2,buf])
    _async_copy(vn_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], vn_scratch_ref.at[buf], sems.at[3,buf])
    _async_copy(g_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], g_scratch_ref.at[buf], sems.at[4,buf])
    _async_copy(beta_ref.at[(beta_slice, pl.ds(None))], beta_scratch_ref.at[buf], sems.at[5,buf])
    _async_copy(A_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], A_scratch_ref.at[buf], sems.at[6,buf])
    _async_copy(h_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], h_scratch_ref.at[buf], sems.at[7,buf])
    _async_copy(do_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], do_scratch_ref.at[buf], sems.at[8,buf])
    _async_copy(dh_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], dh_scratch_ref.at[buf], sems.at[9,buf])
    _async_copy(dv_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], dv_scratch_ref.at[buf], sems.at[10,buf])

  def launch_output(b, sl, beta_sl, wait=False):
    """Start (or wait) all 6 output async copies."""
    _async_copy(dq_scratch_ref.at[b], dq_ref.at[(sl, pl.ds(None), pl.ds(None))], sems.at[11, b], wait=wait)
    _async_copy(dk_scratch_ref.at[b], dk_ref.at[(sl, pl.ds(None), pl.ds(None))], sems.at[12, b], wait=wait)
    _async_copy(dv2_scratch_ref.at[b], dv2_ref.at[(sl, pl.ds(None), pl.ds(None))], sems.at[13, b], wait=wait)
    _async_copy(dg_scratch_ref.at[b], dg_ref.at[(sl, pl.ds(None), pl.ds(None))], sems.at[14, b], wait=wait)
    _async_copy(db_scratch_ref.at[b], db_ref.at[(beta_sl, pl.ds(None))], sems.at[15, b], wait=wait)
    _async_copy(dA_scratch_ref.at[b], dA_ref.at[(sl, pl.ds(None))], sems.at[16, b], wait=wait)

  # === Start input DMA for block 0 before the loop ===
  start_input_dma(0, pl.ds(0, MB), pl.ds(0, MB_pad))
  @pl.when(N_MB > 1)
  def _():
    start_input_dma(1, pl.ds(MB, MB), pl.ds(MB_pad, MB_pad))

  @pl.loop(0, N_MB, unroll=False) # False for Compiler-friendly
  def body(blk_i):
    buf = blk_i % 2
    next_buf = buf ^ 1
    blk_slice = pl.ds(blk_i * MB, MB)
    beta_blk_slice = pl.ds(blk_i * MB_pad, MB_pad)
    prev_blk_slice = pl.ds((blk_i - 1) * MB, MB)
    prev_beta_blk_slice = pl.ds((blk_i - 1) * MB_pad, MB_pad)

    # --- Wait for inputs: vn, h, do, dh, dv ---
    _async_copy(vn_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], vn_scratch_ref.at[buf], sems.at[3,buf], True)
    _async_copy(h_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], h_scratch_ref.at[buf], sems.at[7,buf], True)
    _async_copy(do_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], do_scratch_ref.at[buf], sems.at[8,buf], True)
    _async_copy(dh_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], dh_scratch_ref.at[buf], sems.at[9,buf], True)
    _async_copy(dv_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], dv_scratch_ref.at[buf], sems.at[10,buf], True)
    _async_copy(v_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], v_scratch_ref.at[buf], sems.at[2,buf], True)
    _async_copy(A_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], A_scratch_ref.at[buf], sems.at[6,buf], True)
    _async_copy(beta_ref.at[(beta_blk_slice, pl.ds(None))], beta_scratch_ref.at[buf], sems.at[5,buf], True)
    _async_copy(g_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], g_scratch_ref.at[buf], sems.at[4,buf], True)
    _async_copy(k_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], k_scratch_ref.at[buf], sems.at[1,buf], True)
    _async_copy(q_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], q_scratch_ref.at[buf], sems.at[0,buf], True)
    bvn = vn_scratch_ref[buf]  # [MB, BT, V]
    bh = h_scratch_ref[buf]  # [MB, K, V]
    bdh = dh_scratch_ref[buf]  # [MB, K, V]
    bdo = do_scratch_ref[buf]  # [MB, BT, V]
    bdv = dv_scratch_ref[buf]  # [MB, BT, V]
    bv = v_scratch_ref[buf]
    bA = A_scratch_ref[buf]
    bb = beta_scratch_ref[buf][:MB]  # [MB, 128] (scratch is MB_pad, slice to MB)
    bk = k_scratch_ref[buf]  # [MB, BT, K]
    bq = q_scratch_ref[buf]  # [MB, BT, K]
    bg = g_scratch_ref[buf]  # [MB, BT, K]

    @pl.when(blk_i + 2 < N_MB)
    def _():
      start_input_dma(buf, pl.ds((blk_i + 2) * MB, MB), pl.ds((blk_i + 2) * MB_pad, MB_pad))

    # ===== Derived dependency flags (resolved at trace/compile time) =====
    need_dq_dw = compute_dq or compute_dk or compute_dg or compute_db or compute_dA
    need_dk_base = compute_dk or compute_dg
    need_dvb = compute_dv2 or compute_db
    need_dkgb = compute_dk or compute_db or compute_dg
    need_bA = need_dvb or need_dkgb or compute_dA
    need_kg = compute_dA or need_dkgb
    need_gate = need_dq_dw or need_dk_base or need_dvb or need_kg

    # --- M1: merged dq+dw = [do_scaled; dv] @ h^T ---
    if need_dq_dw:
      bdo_scaled = bdo.astype(jnp.float32) * scale  # [MB, BT, V]
      do_dv = jnp.concatenate([bdo_scaled, bdv.astype(jnp.float32)], axis=1)
      dq_dw = jnp.matmul(
        do_dv,
        bh.astype(jnp.float32).transpose(0, 2, 1),
        precision=precision,
        preferred_element_type=jnp.float32,
      )  # [MB, 2*BT, K]
      b_dq = dq_dw[:, :BT, :]
      b_dw = dq_dw[:, BT:, :]

    # --- M2: dk = vn @ dh^T ---
    if need_dk_base:
      b_dk = jnp.matmul(
        bvn.astype(jnp.float32),
        bdh.astype(jnp.float32).transpose(0, 2, 1),
        precision=precision,
        preferred_element_type=jnp.float32,
      )  # [MB, BT, K]

    # --- M3: dA_partial = dv @ v^T (only for dA) ---
    if compute_dA:
      b_dA_acc = jnp.matmul(
        bdv.astype(jnp.float32),
        bv.astype(jnp.float32).transpose(0, 2, 1),
        precision=precision,
        preferred_element_type=jnp.float32,
      )  # [MB, BT, BT]

    # --- Prepare bA_f32 ---
    if need_bA:
      bA_f32 = bA[:, :, :BT].astype(jnp.float32).transpose(0, 2, 1)  # [MB, BT, BT]

    # --- M4: b_dvb = A @ dv (for dv2, db) ---
    if need_dvb:
      b_dvb = jnp.matmul(
        bA_f32,
        bdv.astype(jnp.float32),
        precision=precision,
        preferred_element_type=jnp.float32,
      )  # [MB, BT, V]

    # --- Beta slice (used by dv2, dk, dg, dA) ---
    bb = bb[:, :BT]  # [MB, BT]

    # --- Output dv2 ---
    if compute_dv2:
      b_dv2 = b_dvb * bb[:, :BT][:, :, None]
      dv2_scratch_ref[buf] = b_dv2
    else:
      dv2_scratch_ref[buf] = jnp.zeros(dv2_scratch_ref.shape[1:])
    _async_copy(dv2_scratch_ref.at[buf], dv2_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], sems.at[13, buf])

    # --- db partial from dvb ---
    if compute_db:
      b_db_acc = (b_dvb * bv).sum(axis=2)  # [MB, BT]

    # --- Gate application ---
    if need_gate:
      bgn = bg[:, -1, :]  # [MB, K]
      gk_exp = exp2(bg)  # [MB, BT, K]

    if compute_dg:
      b_dgk = (bh * bdh).sum(axis=2)  # [MB, K]
      b_dgk = b_dgk * exp2(bgn)  # [MB, K]

    if compute_dq or compute_dg:
      b_dq = b_dq * gk_exp  # [MB, BT, K]

    if need_dk_base:
      b_dk = b_dk * exp2(bgn[:, None, :] - bg)  # [MB, BT, K]

    if compute_dk:
      gb = gk_exp * bb[:, :BT][:, :, None]  # [MB, BT, K]

    # --- kg = k * gk_exp, negate b_dw ---
    if need_kg:
      kg = bk * gk_exp  # [MB, BT, K]
    if need_dq_dw:
      b_dw = -b_dw

    # --- M5: dA += dw @ kg^T (only for dA) ---
    if compute_dA:
      b_dA_acc = b_dA_acc + jnp.matmul(
        b_dw,
        kg.astype(jnp.float32).transpose(0, 2, 1),
        precision=precision,
        preferred_element_type=jnp.float32,
      )
    # --- M6: dkgb = A @ dw (for dk, db, dg) ---
    if need_dkgb:
      dkgb = jnp.matmul(
        bA_f32,
        b_dw,
        precision=precision,
        preferred_element_type=jnp.float32,
      )
    if compute_db:
      b_db_acc = b_db_acc + (dkgb * kg).sum(axis=2)  # [MB, BT]

    # --- Output db ---
    if compute_db:
      db_scratch_ref[buf] = jnp.pad(b_db_acc, ((0, MB_pad - MB), (0, BT_PAD - BT)))
    else:
      db_scratch_ref[buf] = jnp.zeros(db_scratch_ref.shape[1:])
    _async_copy(db_scratch_ref.at[buf], db_ref.at[(beta_blk_slice, pl.ds(None))], sems.at[15, buf])

    # --- dg computation ---
    if compute_dg:
      kdk = bk * b_dk  # [MB, BT, K]
      b_dgk = b_dgk + kdk.sum(axis=1)  # [MB, K]
      b_dg = (
        bq * b_dq - kdk
        + m_last[None, :, None] * b_dgk[:, None, :]
        + kg * dkgb * bb[:, :BT][:, :, None]
      )

    # --- dk final ---
    if compute_dk:
      b_dk = b_dk + dkgb * gb

    # --- Output dq, dk, dg ---
    if compute_dq:
      dq_scratch_ref[buf] = b_dq
    else:
      dq_scratch_ref[buf] = jnp.zeros(dq_scratch_ref.shape[1:])

    if compute_dk:
      dk_scratch_ref[buf] = b_dk
    else:
      dk_scratch_ref[buf] = jnp.zeros(dk_scratch_ref.shape[1:])

    if compute_dg:
      dg_scratch_ref[buf] = b_dg
    else:
      dg_scratch_ref[buf] = jnp.zeros(dg_scratch_ref.shape[1:])

    _async_copy(dq_scratch_ref.at[buf], dq_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], sems.at[11, buf])
    _async_copy(dg_scratch_ref.at[buf], dg_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], sems.at[14, buf])
    _async_copy(dk_scratch_ref.at[buf], dk_ref.at[(blk_slice, pl.ds(None), pl.ds(None))], sems.at[12, buf])

    # --- M7, M8: dA post-processing and output ---
    if compute_dA:
      b_dA_acc = jnp.where(m_lower[None, :, :], b_dA_acc * bb[:, :BT][:, None, :], 0.0)
      b_dA_acc = jnp.matmul(
        b_dA_acc,
        bA_f32,
        precision=precision,
        preferred_element_type=jnp.float32,
      )
      b_dA_acc = jnp.matmul(
        bA_f32,
        b_dA_acc,
        precision=precision,
        preferred_element_type=jnp.float32,
      )
      b_dA_acc = jnp.where(m_lower[None, :, :], -b_dA_acc, 0.0)
      dA_scratch_ref[buf] = jnp.pad(b_dA_acc, ((0, 0), (0, 0), (0, BT_PAD - BT)))
    else:
      dA_scratch_ref[buf] = jnp.zeros(dA_scratch_ref.shape[1:])
    _async_copy(dA_scratch_ref.at[buf], dA_ref.at[(blk_slice, pl.ds(None))], sems.at[16, buf])

    @pl.when(blk_i > 0)
    def _():
      _async_copy(dA_scratch_ref.at[next_buf], dA_ref.at[(prev_blk_slice, pl.ds(None))], sems.at[16, next_buf], wait=True)
      _async_copy(dg_scratch_ref.at[next_buf], dg_ref.at[(prev_blk_slice, pl.ds(None), pl.ds(None))], sems.at[14, next_buf], wait=True)
      _async_copy(db_scratch_ref.at[next_buf], db_ref.at[(prev_beta_blk_slice, pl.ds(None))], sems.at[15, next_buf], wait=True)
      _async_copy(dv2_scratch_ref.at[next_buf], dv2_ref.at[(prev_blk_slice, pl.ds(None), pl.ds(None))], sems.at[13, next_buf], wait=True)
      _async_copy(dk_scratch_ref.at[next_buf], dk_ref.at[(prev_blk_slice, pl.ds(None), pl.ds(None))], sems.at[12, next_buf], wait=True)
      _async_copy(dq_scratch_ref.at[next_buf], dq_ref.at[(prev_blk_slice, pl.ds(None), pl.ds(None))], sems.at[11, next_buf], wait=True)

  # === Wait for last iteration's output DMAs ===
  last_buf = (N_MB - 1) % 2
  last_slice = pl.ds((N_MB - 1) * MB, MB)
  last_beta_slice = pl.ds((N_MB - 1) * MB_pad, MB_pad)
  launch_output(last_buf, last_slice, last_beta_slice, wait=True)


@functools.partial(
  jax.jit, static_argnames=["chunk_size", "scale", "mini_batch"]
)
def chunk_kda_bwd_wy_dqkg_fused_kernel(
  q,
  k,
  v,
  v_new,
  g,
  beta,
  A,
  h,
  do,
  dh,
  dv,
  scale,
  chunk_size=64,
  mini_batch=None,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
):
  """
  JAX Pallas implementation of chunk_kda_bwd_wy_dqkg_fused.

  Args:
      q:      [H, B, T, K]  query tensor.
      k:      [H, B, T, K]  key tensor.
      v:      [H, B, T, V]  original value tensor.
      v_new:  [H, B, T, V]  WY-transformed value tensor.
      g:      [H, B, T, K]  log-space cumsum gate (base-2 scaled).
      beta:   [H, B, T]     WY beta coefficients.
      A:      [H, B, T, BT] Akk inverse matrix (chunk_size = BT).
      h:      [H, B, NT, K, V]  per-chunk hidden states.
      do:     [H, B, T, V]  output gradient.
      dh:     [H, B, NT, K, V]  hidden state gradients.
      dv:     [H, B, T, V]  value gradient.
      scale:  float          softmax scaling factor.
      chunk_size: int        chunk size (BT). T must be divisible by chunk_size.
      mini_batch: int or None. Number of chunks per grid point for DMA
          granularity. None = auto-compute to maximise VMEM utilisation.

  Returns:
      dq:  [H, B, T, K]   query gradient (float32).
      dk:  [H, B, T, K]   key gradient (float32).
      dv2: [H, B, T, V]   value gradient (same dtype as v).
      db:  [H, B, T]      beta gradient (float32).
      dg:  [H, B, T, K]   gate gradient (float32).
      dA:  [H, B, T, BT]  Akk inverse gradient (float32).
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  # =================== input shape assertions ===================
  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(v_new, (H, B, T, V), "v_new")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(A, (H, B, T, BT), "A")
  assert_shape(h, (H, B, NT, K, V), "h")
  assert_shape(do, (H, B, T, V), "do")
  assert_shape(dh, (H, B, NT, K, V), "dh")
  assert_shape(dv, (H, B, T, V), "dv")

  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  hw = get_tpu_config()
  assert BT < hw.block_align_major and BT % hw.block_align_minor == 0, (
    f"BT must be less than {hw.block_align_major}, and be divisible by {hw.block_align_minor}"
  )
  # ==============================================================

  # ===== Profiling: env-var-controlled output gating =====
  compute_dq  = os.environ.get("KDA_BWD_COMPUTE_DQ",  "1") == "1"
  compute_dk  = os.environ.get("KDA_BWD_COMPUTE_DK",  "1") == "1"
  compute_dv2 = os.environ.get("KDA_BWD_COMPUTE_DV2", "1") == "1"
  compute_dg  = os.environ.get("KDA_BWD_COMPUTE_DG",  "1") == "1"
  compute_db  = os.environ.get("KDA_BWD_COMPUTE_DB",  "1") == "1"
  compute_dA  = os.environ.get("KDA_BWD_COMPUTE_DA",  "1") == "1"

  BH = B * H

  # [H, B, T, X] -> [HB*NT, BT, X]  (no transpose, just reshape)
  def _r(x, d):
    return x.reshape(BH * NT, BT, d)

  def _r3(x):
    return x.reshape(BH * NT, BT)

  # [H, B, NT, K, V] -> [HB*NT, K, V]  (no transpose, just reshape)
  def _r_h(x):
    return x.reshape(BH * NT, K, V)

  q_r = _r(q, K)
  k_r = _r(k, K)
  v_r = _r(v, V)
  vn_r = _r(v_new, V)
  g_r = _r(g, K)
  beta_r = _r3(beta)  # [BH*NT, BT, 1] for TPU alignment
  beta_r = jnp.pad(beta_r, ((0,0), (0, hw.block_align_major - BT)))
  A_r = _r(A, BT)
  A_r = jnp.pad(A_r, ((0, 0), (0, 0), (0, hw.block_align_major - BT)))
  h_r = _r_h(h)
  do_r = _r(do, V)
  dh_r = _r_h(dh)
  dv_r = _r(dv, V)

  # Pad K and V to multiples of block_align_major for TPU tiling
  K_pad = align_up(K, hw.block_align_major)
  V_pad = align_up(V, hw.block_align_major)
  if K_pad != K:
    q_r = jnp.pad(q_r, ((0, 0), (0, 0), (0, K_pad - K)))
    k_r = jnp.pad(k_r, ((0, 0), (0, 0), (0, K_pad - K)))
    g_r = jnp.pad(g_r, ((0, 0), (0, 0), (0, K_pad - K)))
  if V_pad != V:
    v_r = jnp.pad(v_r, ((0, 0), (0, 0), (0, V_pad - V)))
    vn_r = jnp.pad(vn_r, ((0, 0), (0, 0), (0, V_pad - V)))
    do_r = jnp.pad(do_r, ((0, 0), (0, 0), (0, V_pad - V)))
    dv_r = jnp.pad(dv_r, ((0, 0), (0, 0), (0, V_pad - V)))
  if K_pad != K or V_pad != V:
    h_r = jnp.pad(h_r, ((0, 0), (0, K_pad - K), (0, V_pad - V)))
    dh_r = jnp.pad(dh_r, ((0, 0), (0, K_pad - K), (0, V_pad - V)))

  total = BH * NT

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = 2 if q.dtype == jnp.bfloat16 else 4
    in_bytes = (5 * BT * K_pad + 4 * BT * V_pad + 2 * K_pad * V_pad + BT + BT * BT) * elem_size
    out_bytes = (3 * BT * K_pad + BT * V_pad + BT + BT * BT) * 4  # outputs always f32
    per_chunk = in_bytes + out_bytes
    MB = estimate_mini_batch(per_chunk, total, max_mb=16)
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"

  # Pad MB to multiple of block_align_minor for beta/db scratch to satisfy TPU tiling
  MB_pad = align_up(MB, hw.block_align_minor)
  N_MB = total // MB
  total_beta = N_MB * MB_pad  # padded total for beta/db HBM

  # Interleave-pad beta_r: (total, BT_PAD) -> (N_MB, MB, BT_PAD) -> (N_MB, MB_pad, BT_PAD) -> (N_MB*MB_pad, BT_PAD)
  BT_PAD = hw.block_align_major
  if MB_pad != MB:
    beta_r = beta_r.reshape(N_MB, MB, BT_PAD)
    beta_r = jnp.pad(beta_r, ((0, 0), (0, MB_pad - MB), (0, 0)))
    beta_r = beta_r.reshape(N_MB * MB_pad, BT_PAD)

  def _spec3(d1, d2):
    return pl.BlockSpec(memory_space=pl.ANY)

  in_specs = [
    _spec3(BT, K),  # q
    _spec3(BT, K),  # k
    _spec3(BT, V),  # v
    _spec3(BT, V),  # v_new
    _spec3(BT, K),  # g
    _spec3(BT, 1),  # beta [BT, 1]
    _spec3(BT, BT),  # A
    _spec3(K, V),  # h
    _spec3(BT, V),  # do
    _spec3(K, V),  # dh
    _spec3(BT, V),  # dv
  ]

  out_specs = [
    _spec3(BT, K),  # dq
    _spec3(BT, K),  # dk
    _spec3(BT, V),  # dv2
    _spec3(BT, K),  # dg
    _spec3(BT, 1),  # db [BT, 1]
    _spec3(BT, BT),  # dA
  ]

  out_shape = [
    jax.ShapeDtypeStruct((total, BT, K_pad), jnp.float32),
    jax.ShapeDtypeStruct((total, BT, K_pad), jnp.float32),
    jax.ShapeDtypeStruct((total, BT, V_pad), jnp.float32),
    jax.ShapeDtypeStruct((total, BT, K_pad), jnp.float32),
    jax.ShapeDtypeStruct((total_beta, BT_PAD), jnp.float32),  # db — padded to MB_pad-aligned
    jax.ShapeDtypeStruct((total, BT, BT_PAD), jnp.float32),
  ]

  kernel = partial(
    _chunk_kda_bwd_wy_dqkg_fused_kernel,
    scale=scale,
    BT=BT,
    K=K_pad,
    V=V_pad,
    MB=MB,
    MB_pad=MB_pad,
    compute_dq=compute_dq,
    compute_dk=compute_dk,
    compute_dv2=compute_dv2,
    compute_dg=compute_dg,
    compute_db=compute_db,
    compute_dA=compute_dA,
    BT_PAD=BT_PAD,
  )

  interpret = get_interpret()


  q_scratch = pltpu.VMEM((2, MB, BT, K_pad), q.dtype)
  k_scratch = pltpu.VMEM((2, MB, BT, K_pad), k.dtype)
  v_scratch = pltpu.VMEM((2, MB, BT, V_pad), v.dtype)
  v_new_scratch = pltpu.VMEM((2, MB, BT, V_pad), v_new.dtype)
  g_scratch = pltpu.VMEM((2, MB, BT, K_pad), g.dtype)
  beta_scratch = pltpu.VMEM((2, MB_pad, BT_PAD), beta.dtype)
  a_scratch = pltpu.VMEM((2, MB, BT, BT_PAD), A.dtype)
  h_scratch = pltpu.VMEM((2, MB, K_pad, V_pad), h.dtype)
  do_scratch = pltpu.VMEM((2, MB, BT, V_pad), do.dtype)
  dh_scratch = pltpu.VMEM((2, MB, K_pad, V_pad), dh.dtype)
  dv_scratch = pltpu.VMEM((2, MB, BT, V_pad), dv.dtype)

  dq_scratch = pltpu.VMEM((2, MB, BT, K_pad), jnp.float32)
  dk_scratch = pltpu.VMEM((2, MB, BT, K_pad), jnp.float32)
  dv2_scratch = pltpu.VMEM((2, MB, BT, V_pad), jnp.float32)
  dg_scratch = pltpu.VMEM((2, MB, BT, K_pad), jnp.float32)
  db_scratch = pltpu.VMEM((2, MB_pad, BT_PAD), jnp.float32)
  da_scratch = pltpu.VMEM((2, MB, BT, BT_PAD), jnp.float32)
  scratch_shapes = [q_scratch, k_scratch, v_scratch, v_new_scratch,
                     g_scratch, beta_scratch, a_scratch, h_scratch,
                     do_scratch, dh_scratch, dv_scratch, dq_scratch,
                     dk_scratch, dv2_scratch, dg_scratch, db_scratch,
                     da_scratch]
  scratch_shapes.append(pltpu.SemaphoreType.DMA((17, 2)))
  dq_r, dk_r, dv2_r, dg_r, db_r, dA_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=scratch_shapes,
    ),
    compiler_params=pltpu.CompilerParams(
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=interpret,
  )(q_r, k_r, v_r, vn_r, g_r, beta_r, A_r, h_r, do_r, dh_r, dv_r)

  # Unpad db from (N_MB*MB_pad, BT_PAD) -> (total, BT_PAD)
  if MB_pad != MB:
    db_r = db_r.reshape(N_MB, MB_pad, BT_PAD)[:, :MB, :].reshape(total, BT_PAD)

  # Unpad K and V dimensions
  if K_pad != K:
    dq_r = dq_r[:, :, :K]
    dk_r = dk_r[:, :, :K]
    dg_r = dg_r[:, :, :K]
  if V_pad != V:
    dv2_r = dv2_r[:, :, :V]

  # [HB*NT, BT, X] -> [H, B, T, X]
  def _ir(x, d):
    return x.reshape(H, B, T, d)

  def _ir3(x):
    return x.reshape(H, B, T)

  return (
    _ir(dq_r, K),
    _ir(dk_r, K),
    _ir(dv2_r, V),
    _ir3(db_r[:, :BT]),
    _ir(dg_r, K),
    _ir(dA_r, BT_PAD)[:, :, :, :BT],
  )


# =====================================================================
# chunk_kda_bwd_dAv  —  Pallas kernel
# =====================================================================


def _chunk_kda_bwd_dAv_kernel(
  v_ref,
  A_ref,
  do_ref,
  dA_ref,
  dv_ref,
  *,
  scale,
  BT,
  BV,
  NV,
  V,
  MB,
):
  bv = v_ref[:]  # [MB, BT, V]
  bA = A_ref[:]  # [MB, BT, BT]
  bdo = do_ref[:]  # [MB, BT, V]

  m_causal = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  bA_masked = jnp.where(m_causal[None, :, :], bA, 0.0)  # [MB, BT, BT]

  b_dA = jnp.zeros((MB, BT, BT), jnp.float32)
  dv_blocks = []

  for i_v in range(NV):
    vs = i_v * BV
    ve = vs + BV

    b_v_blk = bv[:, :, vs:ve]    # [MB, BT, BV]
    b_do_blk = bdo[:, :, vs:ve]  # [MB, BT, BV]

    # dA += do @ v^T — contract BV (dim 2), batch MB (dim 0)
    b_dA += jax.lax.dot_general(
      b_do_blk,
      b_v_blk,
      (((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
    )

    # dv = A^T @ do — contract BT_row (dim 1), batch MB (dim 0)
    b_dv_blk = jax.lax.dot_general(
      bA_masked,
      b_do_blk,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
    )
    dv_blocks.append(b_dv_blk)

  b_dv = jnp.concatenate(dv_blocks, axis=2) if NV > 1 else dv_blocks[0]

  # Apply causal mask and scale
  b_dA = jnp.where(m_causal[None, :, :], b_dA * scale, 0.0)

  dA_ref[:] = b_dA
  dv_ref[:] = b_dv.astype(do_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "block_V",
    "mini_batch",
  ],
)
def chunk_kda_bwd_dAv_kernel(
  q,
  k,
  v,
  do,
  A,
  scale,
  chunk_size=64,
  block_V=None,
  mini_batch=None,
):
  """JAX Pallas implementation of chunk_kda_bwd_dAv.

  Computes the attention gradient dA and value gradient dv for the KDA
  backward pass.

  Args:
      q:     [H, B, T, K]   query tensor.
      k:     [H, B, T, K]   key tensor.
      v:     [H, B, T, V]   value tensor (v_new in the full backward).
      do:    [H, B, T, V]   output gradient.
      A:     [H, B, T, BT]  attention matrix (Aqk).
      scale: float           softmax scaling factor.
      chunk_size: int        chunk size (BT). T must be divisible by chunk_size.
      block_V: V-dimension tile size (BV). Defaults to V. Must divide V.
      mini_batch: int or None. Number of chunks per grid point for DMA
          granularity. None = auto-compute to maximise VMEM utilisation.

  Returns:
      dA: [H, B, T, BT]  attention gradient (float32).
      dv: [H, B, T, V]   value gradient (same dtype as do).
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT

  # =================== input shape assertions ===================
  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(do, (H, B, T, V), "do")
  assert_shape(A, (H, B, T, BT), "A")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

  BV = block_V if block_V is not None else V
  assert V % BV == 0, f"V={V} must be divisible by block_V={BV}"
  # ==============================================================
  NV = V // BV
  BH = B * H

  v_r = v.reshape(BH * NT, BT, V)
  A_r = A.reshape(BH * NT, BT, BT)
  do_r = do.reshape(BH * NT, BT, V)

  total = BH * NT

  # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = 2 if v.dtype == jnp.bfloat16 else 4
    in_bytes = (2 * BT * V + BT * BT) * elem_size
    out_bytes = (BT * BT * 4 + BT * V * 2)
    per_chunk = in_bytes + out_bytes
    MB = estimate_mini_batch(per_chunk, total, max_mb=32)
  else:
    MB = mini_batch
    assert total % MB == 0, f"total={total} must be divisible by mini_batch={MB}"
  def _spec3(d1, d2):
    return pl.BlockSpec(block_shape=(MB, d1, d2), index_map=lambda idx: (idx, 0, 0))

  in_specs = [
    _spec3(BT, V),   # v
    _spec3(BT, BT),  # A
    _spec3(BT, V),   # do
  ]

  out_specs = [
    _spec3(BT, BT),  # dA
    _spec3(BT, V),   # dv
  ]

  out_shape = [
    jax.ShapeDtypeStruct((total, BT, BT), jnp.float32),
    jax.ShapeDtypeStruct((total, BT, V), do.dtype),
  ]

  kernel = partial(
    _chunk_kda_bwd_dAv_kernel,
    scale=scale,
    BT=BT,
    BV=BV,
    NV=NV,
    V=V,
    MB=MB,
  )

  interpret = get_interpret()

  dA_r, dv_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=(total // MB,),
      in_specs=in_specs,
      out_specs=out_specs,
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel",),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=interpret,
  )(v_r, A_r, do_r)

  # [HB*NT, BT, X] -> [H, B, T, X]
  def _ir(x, d):
    return x.reshape(H, B, T, d)

  return _ir(dA_r, BT), _ir(dv_r, V)


# =====================================================================
# chunk_gated_delta_rule_bwd_dhu  —  Unified dhu kernel
# =====================================================================


def _chunk_gated_delta_rule_bwd_dhu_kernel(
  cu_seqlens_ref,  # [B, N+1] SMEM
  chunk_to_seq_ref,  # [B, NT] SMEM
  q_ref,
  k_ref,
  w_ref,
  gk_ref,  # [MB, 1, 1, BT, K]
  do_ref,
  dv_ref,  # [MB, 1, 1, BT, V]
  dht_ref,  # [1, 1, MB, K, V] — sliced via dynamic index_map
  dh_ref,  # [MB, 1, 1, K, V]
  dh0_ref,  # [1, 1, MB, K, V] — sliced via dynamic index_map
  dv2_ref,  # [MB, 1, 1, BT, V]
  scratch_ref,  # [MB, K, V]
  *,
  BT,
  K,
  V,
  NT,
  USE_EXP2,
  scale,
  MB,
):
  """Pallas kernel body for dhu reverse recurrence.

  Grid: (H//MB, B, NT).  MB heads are processed per grid point via
  vectorized batched ops.  Chunks are iterated in reverse via the
  ``"arbitrary"`` grid dimension.  Sequence boundaries are detected
  using ``chunk_to_seq`` and ``cu_seqlens`` arrays in SMEM.
  """
  i_b = pl.program_id(1)
  i_c = pl.program_id(2)  # chunk counter (0 = last chunk in time)
  i_t = NT - 1 - i_c  # global chunk index (forward order)
  t0 = i_t * BT  # global time offset

  seq_idx = chunk_to_seq_ref[i_b, i_t]
  bos = cu_seqlens_ref[i_b, seq_idx]
  eos = cu_seqlens_ref[i_b, seq_idx + 1]

  _exp = exp2 if USE_EXP2 else exp

  # Vectorized over MB — all heads processed simultaneously
  # Reset scratch at sequence end (last chunk of each sequence, first processed in reverse)
  @pl.when(t0 + BT >= eos)
  def _():
    scratch_ref[:] = dht_ref[0, 0, :].astype(scratch_ref.dtype)

  # Snapshot current state gradient as this chunk's dh
  dh_ref[:, 0, 0] = scratch_ref[:].astype(dh_ref.dtype)

  bq = q_ref[:, 0, 0]  # [MB, BT, K]
  bk = k_ref[:, 0, 0]  # [MB, BT, K]
  bw = w_ref[:, 0, 0]  # [MB, BT, K]
  bgk = gk_ref[:, 0, 0]  # [MB, BT, K]
  bdo = do_ref[:, 0, 0]  # [MB, BT, V]
  bdv = dv_ref[:, 0, 0]  # [MB, BT, V]

  b_gk_last = bgk[:, BT - 1, :]  # [MB, K]

  # dv2 = k @ scratch + dv  — batched: [MB, BT, K] @ [MB, K, V] + [MB, BT, V]
  b_dv2 = (
    jnp.matmul(
      bk,
      scratch_ref[:],
      preferred_element_type=jnp.float32,
    )
    + bdv
  )  # [MB, BT, V]
  dv2_ref[:, 0, 0] = b_dv2.astype(dv2_ref.dtype)

  # Decay scratch by last-position gate
  scratch_ref[:] = scratch_ref[:] * _exp(b_gk_last[:, :, None])

  # Accumulate: dh += q^T @ do * scale - w^T @ dv2
  scratch_ref[:] = scratch_ref[:] + (
    jnp.matmul(
      bq.transpose(0, 2, 1),
      bdo,
      preferred_element_type=jnp.float32,
    )
    * scale
    - jnp.matmul(
      bw.transpose(0, 2, 1),
      b_dv2,
      preferred_element_type=jnp.float32,
    )
  )

  # Save dh0 at sequence start (first chunk of each sequence, last processed in reverse)
  @pl.when(t0 == bos)
  def _():
    dh0_ref[0, 0, :] = scratch_ref[:].astype(dh0_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "chunk_size",
    "scale",
    "use_exp2",
    "mini_batch",
  ],
)
def chunk_gated_delta_rule_bwd_dhu_kernel(
  q,
  k,
  w,
  gk,
  h0,
  dht,
  do,
  dv,
  scale,
  chunk_size=64,
  cu_seqlens: jax.Array | None = None,
  use_exp2: bool = True,
  mini_batch: int | None = None,
):
  """JAX Pallas implementation of chunk_gated_delta_rule_bwd_dhu.

  Handles both uniform-length and variable-length (varlen) sequences through
  a single unified code path.  For uniform inputs, synthetic ``cu_seqlens``
  are generated so that the same kernel body and index maps are used.

  Args:
      q:    [H, B, T, K]   gated query tensor.
      k:    [H, B, T, K]   gated key tensor.
      w:    [H, B, T, K]   delta-rule erase weight.
      gk:   [H, B, T, K]   per-key gate (cumsum'd, log2-space).
      h0:   [B, N, H, K, V] or None   initial hidden state.
      dht:  [B, N, H, K, V] or None   gradient of final state.
      do:   [H, B, T, V]   output gradient.
      dv:   [H, B, T, V]   value gradient from dAv stage.
      scale: float           softmax scaling factor.
      chunk_size: int        chunk size (BT).
      cu_seqlens: [B, N+1] or None  cumulative sequence lengths.
                  None for uniform-length inputs.
      use_exp2: bool         use exp2 for gate computation.
      mini_batch: int or None. Number of heads per grid point for DMA
          granularity. None = auto-compute to maximise VMEM utilisation.

  Returns:
      dh:  [H, B, NT, K, V]   per-chunk hidden state gradient.
      dh0: [B, N, H, K, V] or None   gradient of initial state.
      dv2: [H, B, T, V]       updated value gradient.
  """
  H, B, T, K = q.shape
  V = do.shape[-1]
  BT = chunk_size
  NT = T // BT

  # =================== input shape assertions ===================
  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(w, (H, B, T, K), "w")
  assert_shape(gk, (H, B, T, K), "gk")
  assert_shape(do, (H, B, T, V), "do")
  assert_shape(dv, (H, B, T, V), "dv")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  # ==============================================================

  # Unified: create synthetic cu_seqlens for uniform
  is_varlen = cu_seqlens is not None
  if cu_seqlens is None:
    cu_seqlens_dev = jnp.broadcast_to(
      jnp.array([[0, T]], dtype=jnp.int32), (B, 2)
    )
    N = 1
  else:
    cu_seqlens_dev = cu_seqlens
    # Ensure cu_seqlens_dev is 2D [B, N+1] for kernel block specs
    if cu_seqlens_dev.ndim == 1:
      cu_seqlens_dev = jnp.broadcast_to(cu_seqlens_dev[None, :], (B, cu_seqlens_dev.shape[0]))
    N = cu_seqlens_dev.shape[-1] - 1

  _squeeze_dh0 = False  # track if we need to squeeze output back to 4D
  if is_varlen:
    assert_shape_or_none(h0, (B, N, H, K, V), "h0")
    assert_shape_or_none(dht, (B, N, H, K, V), "dht")
  else:
    assert_shape_or_none(h0, (B, 1, H, K, V), "h0")
    assert_shape_or_none(dht, (B, 1, H, K, V), "dht")

  chunk_to_seq = _build_chunk_map(cu_seqlens_dev, T, BT)  # [B, NT]

  # Pack data: [H, B, T, X] → [H, B, NT, BT, X]
  q_r = q.reshape(H, B, NT, BT, K)
  k_r = k.reshape(H, B, NT, BT, K)
  w_r = w.reshape(H, B, NT, BT, K)
  gk_r = gk.reshape(H, B, NT, BT, K)
  do_r = do.reshape(H, B, NT, BT, V)
  dv_r = dv.reshape(H, B, NT, BT, V)

  # State arrays: [B, N, H, K, V]
  dht_arr = dht if dht is not None else jnp.zeros((B, N, H, K, V), dtype=jnp.float32)

  # ---- auto-compute mini-batch (MB) along H to maximise VMEM utilisation ----
  if mini_batch is None:
    elem_size = 2 if q.dtype == jnp.bfloat16 else 4
    # per head: 4 inputs [BT,K] + 2 inputs [BT,V] + 1 state [K,V] + scratch [K,V]
    in_bytes = (4 * BT * K + 2 * BT * V + K * V) * elem_size + K * V * 4  # scratch f32
    # per head: 1 out [K,V] + 1 state [K,V] + 1 out [BT,V]
    out_bytes = (2 * K * V + BT * V) * 4
    per_head = in_bytes + out_bytes
    MB = estimate_mini_batch(per_head, H, max_mb=32)
  else:
    MB = mini_batch
    assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

  kernel = partial(
    _chunk_gated_delta_rule_bwd_dhu_kernel,
    scale=scale,
    BT=BT,
    K=K,
    V=V,
    NT=NT,
    USE_EXP2=use_exp2,
    MB=MB,
  )
  interpret = get_interpret()
  scratch = pltpu.VMEM((MB, K, V), jnp.float32)

  # Index maps: reverse chunk order for the arbitrary dimension
  # With num_scalar_prefetch=2, index_maps receive (h, b, c, cu_seqlens_ref, chunk_to_seq_ref)
  # h is the head-batch index: actual heads are [h*MB, h*MB+MB)
  def idx_chunk_K(h, b, c, cu_seqlens_ref, chunk_to_seq_ref):
    return (h, b, NT - 1 - c, 0, 0)

  def idx_chunk_V(h, b, c, cu_seqlens_ref, chunk_to_seq_ref):
    return (h, b, NT - 1 - c, 0, 0)

  # dht/dh0 are [B, N, H, K, V] — batch MB heads along H dim
  def idx_state(h, b, c, cu_seqlens_ref, chunk_to_seq_ref):
    seq_idx = chunk_to_seq_ref[b, NT - 1 - c]
    return (b, seq_idx, h, 0, 0)

  in_specs = [
    pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk_K),  # q
    pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk_K),  # k
    pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk_K),  # w
    pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk_K),  # gk
    pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk_V),  # do
    pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk_V),  # dv
    pl.BlockSpec((1, 1, MB, K, V), index_map=idx_state),  # dht — MB heads per segment
  ]
  out_specs = [
    pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_chunk_K),  # dh
    pl.BlockSpec((1, 1, MB, K, V), index_map=idx_state),  # dh0 — MB heads per segment
    pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk_V),  # dv2
  ]
  out_shape = [
    jax.ShapeDtypeStruct((H, B, NT, K, V), jnp.float32),  # dh
    jax.ShapeDtypeStruct((B, N, H, K, V), jnp.float32),  # dh0
    jax.ShapeDtypeStruct((H, B, NT, BT, V), jnp.float32),  # dv2
  ]

  dh_r, dh0_r, dv2_r = pl.pallas_call(
    kernel,
    out_shape=out_shape,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=(H // MB, B, NT),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=[scratch],
    ),
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      disable_bounds_checks=True,
      vmem_limit_bytes=get_tpu_config().vmem_limit_bytes,
    ),
    interpret=interpret,
  )(cu_seqlens_dev, chunk_to_seq, q_r, k_r, w_r, gk_r, do_r, dv_r, dht_arr)

  # Output reshape
  dh = dh_r  # already [H, B, NT, K, V]
  dv2 = dv2_r.reshape(H, B, T, V)
  dh0 = dh0_r if h0 is not None else None

  return dh, dh0, dv2


# =====================================================================
# chunk_kda_bwd  —  6-stage backward orchestrator
# =====================================================================

@functools.partial(
  jax.jit,
  static_argnames=[
    "scale",
    "chunk_size",
    "safe_gate",
    "lower_bound",
    "use_gate_in_kernel",
    "transpose_state_layout",
    "disable_recompute",
    "cp_context",
    "N_max",
  ],
)
def chunk_kda_bwd(
  q: jax.Array,  # [H, B, T, K]
  k: jax.Array,  # [H, B, T, K]
  v: jax.Array,  # [H, B, T, V]
  beta: jax.Array,  # [H, B, T]
  Aqk: jax.Array,  # [H, B, T, BT]
  Akk: jax.Array,  # [H, B, T, BT]
  scale: float,
  initial_state: jax.Array,  # [N, H, K, V]
  do: jax.Array,  # [H, B, T, V]
  dht: jax.Array,  # [N, H, K, V]
  g: jax.Array | None = None,  # [H, B, T, K]
  g_org: jax.Array | None = None,  # [H, B, T, K] — original gate
  segment_ids: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
  chunk_size: int = 64,
  safe_gate: bool = False,
  lower_bound: float | None = None,
  use_gate_in_kernel: bool = False,
  A_log: jax.Array | None = None,
  dt_bias: jax.Array | None = None,
  disable_recompute: bool = False,
  cp_context: CPContext | None = None,
  transpose_state_layout: bool = False,
  N_max: int | None = None,
  **kwargs,
):
  """Chunk KDA backward — 6-stage pipeline.

  Aligned with FLA fla.ops.kda.chunk_bwd.chunk_kda_bwd.

  Stage 0: recompute forward intermediates (w, u, qg, kg, h, v_new)
  Stage 1: chunk_kda_bwd_dAv -> dAqk, dv
  Stage 2: chunk_gated_delta_rule_bwd_dhu -> dh, dh0, dv
  Stage 3: chunk_kda_bwd_wy_dqkg_fused -> dq, dk, dv, db, dg, dAkk
  Stage 4: chunk_kda_bwd_intra -> refine dq, dk, db, dg
  Stage 5: reverse cumsum on dg

  All tensor inputs and outputs use head-first ``[H, B, T, ...]`` layout.

  Args:
      q:     [H, B, T, K]   — query vectors.
      k:     [H, B, T, K]   — key vectors.
      v:     [H, B, T, V]   — value vectors.
      beta:  [H, B, T]      — per-token mixing coefficient.
      Aqk:   [H, B, T, BT]  — intra-chunk attention matrix (from fwd).
      Akk:   [H, B, T, BT]  — Akk inverse matrix (from fwd).
      scale: float           — attention scale factor.
      initial_state: [N, H, K, V], [B, N, H, K, V] or None — initial
                    hidden state.  Varlen callers pass ``[N, H, K, V]``
                    or ``[B, N, H, K, V]``; uniform callers pass
                    ``[B, H, K, V]``.
      do:    [H, B, T, V]   — output gradient (head-first).
      dht:   [N, H, K, V] or None — final state gradient (varlen) or
             ``[B, H, K, V]`` (uniform).  May also be ``[B, N, H, K, V]``.
      g:     [H, B, T, K]   — post cumsum gate in log2 space.
      chunk_size: int        — tile size.

  Returns:
      dq:    [H, B, T, K]
      dk:    [H, B, T, K]
      dv:    [H, B, T, V]
      db:    [H, B, T]
      dg:    [H, B, T, K]
      dh0:   [B, N, H, K, V] or None
      dA:    [H] or None — gate parameter gradient (only when use_gate_in_kernel=True).
      dbias: [H*K] or None — gate bias gradient (only when use_gate_in_kernel=True).
  """

  # ============= extract dimensions =============
  # All inputs are [H, B, T, X] (caller is responsible for transposing)
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  scale = K ** -0.5 if scale is None else scale
  cu_seqlens = kwargs.get("cu_seqlens", None)
  if (cu_seqlens is None) and (segment_ids is not None):
    if segment_ids.ndim == 1:
      segment_ids = segment_ids[None,]
    # per-batch cu_seqlens [B, N+1]
    caller_N_max = N_max
    if caller_N_max is not None:
      N_max = caller_N_max
    else:
      N_max = cdiv(T, BT)
      if initial_state is not None:
        # Varlen: (B, N, H, K, V)
        N_max = initial_state.shape[1]
    cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=N_max)

  # "N_max must be provided when segment_ids are used" — unless cu_seqlens
  # was already derived upstream (e.g. bwd receives it from fwd residuals).
  assert (segment_ids is None) or (cu_seqlens is not None) or (N_max is not None)
  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(beta, (H, B, T), "beta")
  assert_shape(Aqk, (H, B, T, BT), "Aqk")
  assert_shape(Akk, (H, B, T, BT), "Akk")
  assert_shape(do, (H, B, T, V), "do")
  # assert_shape_or_none(initial_state, (N_state, H, K, V), "initial_state")
  # assert_shape_or_none(dht, (N_state, H, K, V), "dht")

  # When use_gate_in_kernel=True and disable_recompute=False, g_cumsum is
  # None from forward (freed to save memory) and will be recomputed in
  # Stage 0 below from g_org.  Only assert non-None when we won't recompute.
  if not (use_gate_in_kernel and not disable_recompute):
    assert g is not None, "g (post-cumsum, log2 space) must be provided"
    assert_shape(g, (H, B, T, K), "g")
  assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
  _cp_active = cp_context is not None and cp_context.is_cp_enabled
  assert transpose_state_layout is False, "not support transpose_state_layout yet"

  # ============= assert input shapes and static properties =============
  # initial_state/dht: [B, H, K, V] (non-varlen) or [N, H, K, V] (varlen)

  # ---- Stage 0: Recompute forward intermediates ----
  # Two execution paths:
  #   Path A (disable_recompute=True): ``h`` was tagged with
  #     ``checkpoint_name("kda_residuals")`` in fwd, so nn.remat kept it.
  #     We rebuild w/u/qg/kg via the cheap ``recompute_w_u_fwd`` helper,
  #     then locally derive ``v_new = u - w @ h`` per-chunk in parallel —
  #     eliminating the sequential ``chunk_gated_delta_rule_fwd_h``
  #     recurrence (the only cross-chunk dependency in bwd).
  #   Path B (full recompute fallback): re-execute the entire
  #     ``chunk_gated_delta_rule_fwd_h`` recurrence to reconstruct both
  #     h and v_new. Used when disable_recompute=False.

  if disable_recompute:
    # Path A: save-h fast path.
    if use_gate_in_kernel:
      assert A_log is not None, "A_log must not be None when use_gate_in_kernel=True"
      g_cumsum = kda_gate_chunk_cumsum(
        g=g_org,
        A_log=A_log,
        chunk_size=BT,
        scale=RCP_LN2,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
      )
      g = g_cumsum  # already [H,B,T,K]

    # Saved h came in via _hf5 transpose at the custom_vjp boundary —
    # already [H, B, NT, K, V].
    h = kwargs["h"]
    assert h.shape == (H, B, NT, K, V), (
      f"saved h must be [H={H}, B={B}, NT={NT}, K={K}, V={V}], got {h.shape}"
    )

    # M1 fusion: recompute w/qg/kg + v_new in one kernel (no u HBM round-trip).
    w, qg, kg, v_new = fused_recompute_w_u_vnew_from_h_pallas(
      q=q,
      k=k,
      v=v,
      beta=beta,
      A=Akk,
      g=g,
      h=h,
      chunk_size=BT,
    )
  else:
    # Path B: full recompute fallback.
    if use_gate_in_kernel:
      assert A_log is not None, "A_log must not be None when use_gate_in_kernel=True"
      g_cumsum = kda_gate_chunk_cumsum(
        g=g_org,
        A_log=A_log,
        chunk_size=BT,
        scale=RCP_LN2,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
      )
      g = g_cumsum  # already [H,B,T,K]

    # recompute_w_u_fwd is natively [H,B,T,X]
    w, u, qg, kg = recompute_w_u_fwd(
      k=k,
      v=v,
      beta=beta,
      A=Akk,
      q=q,
      gk=g,
    )
    assert kg is not None

    # chunk_gated_delta_rule_fwd_h expects [B,T,H,X]
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
      k=kg,
      w=w,
      u=u,
      gk=g,
      initial_state=initial_state,
      output_final_state=False,
      chunk_size=chunk_size,
      cu_seqlens=cu_seqlens,
      chunk_indices=chunk_indices,
      use_exp2=True,
    )
    # Varlen: pad h from NT_total to NT chunks (padding chunks get zero state)
    if cu_seqlens is not None and h.shape[2] < NT:
      h = jnp.pad(h, ((0, 0), (0, 0), (0, NT - h.shape[2]), (0, 0), (0, 0)))

  # ---- Stage 1: dAqk and initial dv ----
  dAqk, dv = chunk_kda_bwd_dAv_kernel(
    q=q,
    k=k,
    v=v_new,
    do=do,
    A=Aqk,
    scale=scale,
    chunk_size=chunk_size,
  )

  if _cp_active:
    assert segment_ids is not None, "backward CP requires rank-local segment_ids"
    assert cp_context.post_num_ranks is not None, "backward CP requires post_num_ranks"
    assert cp_context.is_last_rank is not None, "backward CP requires is_last_rank"
    # pre_process requires segment_ids length == aligned T (q.shape[2]).
    # Caller's segment_ids is un-aligned (length T_orig); pad with 0
    # (= padding seg id, OOB chunks naturally inactive).
    if segment_ids.ndim == 1:
      segment_ids = segment_ids[None,]
    T_seg = segment_ids.shape[-1]
    if T_seg < T:
      pad_width = ((0, 0), (0, T - T_seg))
      segment_ids = jnp.pad(segment_ids, pad_width)
    elif T_seg > T:
      segment_ids = segment_ids[..., :T]
    # Inputs already in [H, B, T, X]; pre_process consumes this layout
    # directly (no transpose round-trip).
    dS_ext, dM = chunk_gated_delta_rule_bwd_dhu_pre_process(
      q=qg,
      k=kg,
      w=w,
      do=do,
      dv=dv,
      gk=g,
      scale=scale,
      segment_ids=segment_ids,
      chunk_size=BT,
      use_exp2=True,
    )
    # Pack dS_ext [B,H,K,V] and dM [B,H,K,K] into one tensor along the last
    # axis so a single all_gather covers both, halving the CP collective cost.
    packed = jnp.concatenate([dS_ext, dM], axis=-1)  # [B, H, K, V+K]
    packed_all, _ = all_gather_into_tensor(packed, cp_context.axis_name)
    dS_ext_all = packed_all[..., :V]                  # [cp, B, H, K, V]
    dM_all = packed_all[..., V:V + K]                 # [cp, B, H, K, K]
    rank = jax.lax.axis_index(cp_context.axis_name)
    post_num = cp_context.post_num_ranks
    is_last = cp_context.is_last_rank
    ds_list = []
    for b in range(B):
      ds_b = _merge_dht(
        dS_ext_all[:, b:b+1],  # [cp, 1, H, K, V]
        dM_all[:, b:b+1],      # [cp, 1, H, K, K]
        rank=rank,
        post_num_ranks=post_num[b],
        is_last_rank=is_last[b],
      )
      ds_list.append(ds_b)  # [1, H, K, V]
    dS_in = jnp.concatenate(ds_list, axis=0)  # [B, H, K, V]

    # dS_in: [B, H, K, V] — merged downstream gradient for each batch element.
    # M4 expects dht as [B, N, H, K, V] with per-batch segment indexing.
    # dS_in[b] goes to slot (last_seg_id_b - 1) in its own N dimension.
    N = N_max
    dht = jnp.zeros((B, N, H, K, V), dtype=jnp.float32)
    max_per_batch = jnp.max(segment_ids, axis=1)  # [B]
    for b in range(B):
      last_seg_id_b = max_per_batch[b]
      has_real_b = last_seg_id_b > 0
      dht_slot = jnp.maximum(last_seg_id_b - 1, 0)
      dht_slot = jnp.minimum(dht_slot, N - 1)
      dht_value_b = jnp.where(has_real_b, dS_in[b], jnp.zeros_like(dS_in[b]))
      dht = dht.at[b, dht_slot, :, :, :].set(dht_value_b)
    initial_state = None

  # M4 requires 2D segment_ids [B, T]
  if segment_ids is not None and segment_ids.ndim == 1:
    segment_ids = segment_ids[None,]

  # ---- Stage 2+3+4+5: M4 mega fusion (dhu + WY + intra + cumsum) ----
  # M4 expects dht as [B, N, H, K, V].
  # Normalize 4D dht to 5D (uniform: insert N=1 at dim1).
  if dht is not None and dht.ndim == 4 and segment_ids is None:
    dht = dht[:, None, :, :, :]
  dht_m4 = dht
  dq, dk, dv, db, dg, dh0 = _fused_dhu_wy_intra_cumsum_pallas_jit(
    q=q,
    k=k,
    v=v,
    v_new=v_new,
    qg=qg,
    kg=kg,
    w=w,
    g=g,
    beta=beta,
    A=Akk,
    h=h,
    do=do,
    dv0=dv,
    dAqk=dAqk,
    dht=dht_m4,
    scale=scale,
    segment_ids=segment_ids,
    chunk_size=chunk_size,
    use_exp2=True,
    return_dh0=initial_state is not None,
    N_MAX=N_max
  )

  dA, dbias = None, None
  if use_gate_in_kernel:
    dg, dA, dbias = kda_gate_bwd(
      g=g_org,
      A_log=A_log,
      dt_bias=dt_bias,
      dyg=dg,
      lower_bound=lower_bound,
    )

  # Non-varlen: squeeze dh0 5D [B,1,H,K,V] → 4D [B,H,K,V]
  # (varlen callers pass 5D, so dh0 stays 5D)
  _is_varlen = cu_seqlens is not None or (segment_ids is not None)
  if dh0 is not None and dh0.ndim == 5 and not _is_varlen:
    dh0 = dh0[:, 0]

  return dq, dk, dv, db, dg, dh0, dA, dbias
