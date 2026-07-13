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
# KDA intra-chunk forward kernels
# =============================================================================

"""KDA intra-chunk kernel: exact triangular solve for delta-rule dependencies.

Within each chunk, the delta rule creates lower-triangular dependencies
between positions. This module solves the resulting system (I + L)x = b
exactly via block forward substitution (block size 16), where L is the
strictly lower-triangular key-key interaction matrix scaled by beta.

Outputs (forward):
  w:       [H, B, T, K]           -- correction weights for inter-chunk state
  u:       [H, B, T, V]           -- delta-corrected values
  qg:      [H, B, T, K] or None   -- q * exp2(g), only if disable_recompute
  kg:      [H, B, T, K]           -- k * exp2(g_last - g)
  Aqk:     [H, B, T, BT]          -- query-key attention matrix (flattened)
  Akk:     [H, B, T, BT]          -- exact inverse of (I + L) (flattened)
"""


import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
    assert_shape,
    exp2,
    get_interpret,
)


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
  use_neumann: bool = True,
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
  elif use_neumann and q.dtype != jnp.float32:
    return kda_fwd_intra_varlen_v2(
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
  else:
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


def _kda_fwd_intra_varlen_kernel_v2(
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
  """Optimized varlen intra-chunk kernel: BC=16 sub-block Aqk/L + Neumann inversion.

  Replaces the BT×BT full-block computation with BC=16 sub-block iteration
  (16x VMEM reduction) and replaces sequential forward substitution with
  block-diagonal Neumann doubling + off-diagonal Neumann polynomial
  (better TPU MXU utilization via batched matmuls).

  All refs have leading singleton dims from BlockSpec: [1, 1, 1, BT, D].
  """
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

  causal_bc = jnp.tril(jnp.ones((BC, BC), dtype=jnp.float32))
  strict_bc = jnp.tril(jnp.ones((BC, BC), dtype=jnp.float32), k=-1)
  zeros_bc = jnp.zeros((BC, BC), dtype=jnp.float32)

  # --- BC=16 sub-block factored Aqk/L via dot_general (matmul on MXU) ---
  # Factorization: exp2(g_i - g_j) = exp2(g_i - ref) * exp2(ref - g_j)
  # with ref = g_i[0] (first element of i sub-block).
  # Since g is monotonically non-increasing:
  #   - exp2(g_i - ref): exponent <= 0, always in (0, 1]
  #   - exp2(ref - g_j): for j_sc < i_sc, ref <= g_j so exponent <= 0;
  #     for diagonal (j_sc == i_sc), max exponent = g[0] - g[BC-1]
  #     (bounded by BC * |lower_bound|, safe for float32)
  _dot_aqk = lambda a, b: jax.lax.dot_general(
    a, b, (((1,), (1,)), ((), ())),
    preferred_element_type=jnp.float32,
  )

  Aqk_rows = []
  L_rows = []
  for i_sc in range(NC):
    i_s = i_sc * BC
    q_i = q_f32[i_s : i_s + BC]
    k_i = k_f32[i_s : i_s + BC]
    g_i = g_f32[i_s : i_s + BC]
    beta_i = beta_f32[i_s : i_s + BC]

    gn = g_i[0:1, :]  # (1, K) reference point
    q_eg = q_i * jnp.exp2(g_i - gn)
    k_eg = k_i * jnp.exp2(g_i - gn)

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

        k_eng = k_j * jnp.exp2(gn - g_j)

        Aqk_blk = scale * _dot_aqk(q_eg, k_eng)
        Akk_blk = beta_i * _dot_aqk(k_eg, k_eng)

        if i_sc == j_sc:
          Aqk_blk = Aqk_blk * causal_bc
          Akk_blk = Akk_blk * strict_bc

        Aqk_blks.append(Aqk_blk)
        L_blks.append(Akk_blk)

    Aqk_rows.append(jnp.concatenate(Aqk_blks, axis=1))
    L_rows.append(jnp.concatenate(L_blks, axis=1))

  Aqk = jnp.concatenate(Aqk_rows, axis=0).astype(dtype)
  L = jnp.concatenate(L_rows, axis=0)

  # --- Neumann series inversion: (I + L)^{-1} ---
  BC_inv = 16
  NC_inv = BT // BC_inv
  inv_dtype = jnp.bfloat16 if dtype == jnp.bfloat16 else jnp.float32

  L = L.astype(inv_dtype)

  _idx = jnp.arange(BT, dtype=jnp.int32)
  _block_id = _idx // BC_inv
  _same_block = (_block_id[:, None] == _block_id[None, :]).astype(inv_dtype)
  block_mask = _same_block
  L_diag = L * block_mask
  F = L - L_diag

  _dot_inv = lambda a, b: jax.lax.dot_general(
    a, b, (((1,), (0,)), ((), ())),
    preferred_element_type=jnp.float32,
  ).astype(inv_dtype)

  # Level 1: Neumann doubling on block-diagonal
  I_bt = jnp.eye(BT, dtype=inv_dtype)
  neg_Ld = -L_diag
  S = I_bt + neg_Ld
  Mk = neg_Ld
  num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
  for _ in range(num_diag_steps):
    Mk = _dot_inv(Mk, Mk)
    S = _dot_inv(S, I_bt + Mk)
  P = S

  # Level 2: off-diagonal block Neumann
  v_beta = (v.astype(jnp.float32) * beta_f32).astype(inv_dtype)
  k_eg_beta = (k_f32 * jnp.exp2(g_f32) * beta_f32).astype(inv_dtype)
  rhs = jnp.concatenate([v_beta, k_eg_beta, I_bt], axis=-1)

  _dot_apply = lambda a, b: jax.lax.dot_general(
    a, b, (((1,), (0,)), ((), ())),
    preferred_element_type=jnp.float32,
  )

  if NC_inv == 1:
    result = _dot_apply(P, rhs)
  else:
    F_and_rhs = jnp.concatenate([F, rhs], axis=-1)
    P_merged = _dot_apply(P, F_and_rhs)
    G = P_merged[:, :BT]
    P_rhs = P_merged[:, BT:]

    inv_I_G = jnp.eye(BT, dtype=jnp.float32) - G
    Gk = G
    for k_step in range(2, NC_inv):
      Gk = _dot_apply(Gk, G)
      if k_step % 2 == 0:
        inv_I_G = inv_I_G + Gk
      else:
        inv_I_G = inv_I_G - Gk

    result = _dot_apply(inv_I_G, P_rhs)

  u = result[:, :value_dim]
  w = result[:, value_dim : value_dim + head_dim]
  A_inv = result[:, value_dim + head_dim :]

  # --- kg = k * exp2(g_last - g) ---
  g_last = g_f32[BT - 1 : BT, :]
  kg = k_f32 * exp2(g_last - g_f32)

  # --- qg = q * exp2(g) (optional) ---
  qg = q_f32 * exp2(g_f32) if disable_recompute else jnp.zeros_like(q_f32)

  # --- Store outputs ---
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
def kda_fwd_intra_varlen_v2(
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
  """Varlen intra-chunk forward: BC=16 sub-block + Neumann inversion.

  Upstream ``_align_seqs`` guarantees T is BT-aligned and no chunk
  straddles a sequence boundary, so this uses direct transpose+reshape
  (same layout path as ``pallas_kda_fwd_intra``).

  Args:
      q:     [H, B, T, K] -- query vectors (packed layout, B=1).
      k:     [H, B, T, K] -- key vectors.
      v:     [H, B, T, V] -- value vectors.
      gk:    [H, B, T, K] -- chunk-local cumsum of gates in log2 space.
      beta:  [H, B, T]    -- per-token scalar mixing coefficient.
      scale: float         -- attention scale factor.
      cu_seqlens: [N+1] int32 -- cumulative sequence lengths (BT-aligned).
      chunk_size: int      -- chunk size BT (default 64).
      chunk_indices: optional precomputed chunk index array (unused).
      safe_gate: bool      -- midpoint stabilization for sub-block exp2.
      disable_recompute: bool -- if True, also output qg.

  Returns:
      w:    [H, B, T, K]       -- correction weights.
      u:    [H, B, T, V]       -- delta-corrected values.
      qg:   [H, B, T, K] or None -- q * exp2(gk), only if disable_recompute.
      kg:   [H, B, T, K]       -- k * exp2(g_last - gk).
      Aqk:  [H, B, T, BT]      -- query-key attention matrix per chunk.
      Akk:  [H, B, T, BT]      -- (I + L)^{-1} matrix per chunk.
  """
  assert cu_seqlens is not None, "cu_seqlens must be provided for varlen"
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be BT-aligned (upstream _align_seqs guarantees this)"
  assert BT >= 16 and BT % 16 == 0
  NC = T // BT

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(gk, (H, B, T, K), "gk")
  assert_shape(beta, (H, B, T), "beta")

  # --- Direct transpose+reshape to [B, H, NC, BT, D] ---
  q_r = q.reshape(H, B, NC, BT, K)
  k_r = k.reshape(H, B, NC, BT, K)
  g_r = gk.reshape(H, B, NC, BT, K)
  beta_r = beta.reshape(H, B, NC, BT, 1)
  v_r = v.reshape(H, B, NC, BT, V)

  grid = (H, B, NC)

  def _make_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, n: (i, j, n, 0, 0), block_shape=(1, 1, 1, BT, last_dim)
    )

  (u_r, w_r, qg_r, kg_r, Aqk_r, Akk_inv_r) = pl.pallas_call(
    functools.partial(
      _kda_fwd_intra_varlen_kernel_v2,
      chunk_size=BT,
      head_dim=K,
      value_dim=V,
      scale=scale,
      disable_recompute=disable_recompute,
      safe_gate=safe_gate,
    ),
    interpret=get_interpret(),
    out_shape=[
      jax.ShapeDtypeStruct((H, B, NC, BT, V), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), q.dtype),
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

  # --- Reshape back to [H, B, T, D] (head-first) ---
  w_out = w_r.reshape(H, B, T, K)
  u_out = u_r.reshape(H, B, T, V)
  kg_out = kg_r.reshape(H, B, T, K)

  qg_out: jax.Array | None
  if disable_recompute:
    qg_out = qg_r.reshape(H, B, T, K)
  else:
    qg_out = None

  Aqk_flat = Aqk_r.reshape(H, B, NC * BT, BT)
  Akk_flat = Akk_inv_r.reshape(H, B, NC * BT, BT)

  return w_out, u_out, qg_out, kg_out, Aqk_flat, Akk_flat


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
# KDA fused gate + intra forward kernels
# =============================================================================

"""Fused Stage 1+2: gate activation + cumsum + intra-chunk solve in one kernel.

Eliminates the HBM round-trip of g_cumsum [H,B,T,K] between the separate
gate cumsum kernel and the intra-chunk solve kernel.  The cumsum is computed
via a tril matmul inside the Pallas kernel body, and the result stays in
VMEM for immediate use by the Aqk/L construction and Neumann inversion.

g_cumsum is still written to an output ref for downstream Stage 3/4.

All outputs use head-first [H, B, T, ...] layout.

Outputs (forward):
  w:         [H, B, T, K]           -- correction weights
  u:         [H, B, T, V]           -- delta-corrected values
  qg:        [H, B, T, K] or None   -- q * exp2(g_cumsum), if disable_recompute
  kg:        [H, B, T, K]           -- k * exp2(g_last - g_cumsum)
  Aqk:       [H, B, T, BT]          -- query-key attention matrix
  Akk:       [H, B, T, BT]          -- (I + L)^{-1} flattened
  g_cumsum:  [H, B, T, K]           -- chunk-local cumsum of activated gates
"""


import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import (
    assert_shape,
    exp2,
    get_interpret,
)

_RCP_LN2 = 1.0 / math.log(2)


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


# =========================================================================
# Non-varlen dispatch
# =========================================================================

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
  """Fused gate cumsum + intra-chunk solve (non-varlen) with mini-batch.

  Processes MB heads per grid point to amortize DMA overhead.

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


# =========================================================================
# Varlen dispatch
# =========================================================================

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
def kda_fwd_intra_fused_varlen(
  q,
  k,
  v,
  g,
  beta,
  scale,
  cu_seqlens,
  chunk_size=64,
  chunk_indices=None,
  safe_gate=True,
  disable_recompute=False,
  cumsum_scale=_RCP_LN2,
  A_log=None,
  dt_bias=None,
  use_gate_in_kernel=False,
  lower_bound=None,
  mini_batch=None,
):
  """Fused gate cumsum + intra-chunk solve (varlen) with mini-batch.

  Processes MB heads per grid point to amortize DMA overhead.

  Args:
      g:     [H, B, T, K] -- raw gate input (NOT cumsum'd), head-first layout.
      cu_seqlens: [N+1] int32 -- cumulative sequence lengths (BT-aligned).
      mini_batch: int or None -- heads per grid point (auto if None).

  Returns:
      7-tuple: (w, u, qg, kg, Aqk, Akk, g_cumsum) in head-first
      ``[H, B, T, ...]`` layout.  qg is None when ``disable_recompute=False``.
  """
  assert cu_seqlens is not None, "cu_seqlens must be provided for varlen"
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0, f"T={T} must be BT-aligned (upstream _align_seqs guarantees this)"
  assert BT >= 16 and BT % 16 == 0
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

  # --- [H, B, T, K] -> [H, B, NC, BT, K] ---
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
      index_map=lambda i, j, n: (i, j, n, 0, 0),
      block_shape=(MB, 1, 1, BT, last_dim),
    )

  def _make_per_head_spec(last_dim):
    return pl.BlockSpec(
      index_map=lambda i, j, n: (i, 0, 0, 0, 0),
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
      jax.ShapeDtypeStruct((H, B, NC, BT, V), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, K), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), q.dtype),
      jax.ShapeDtypeStruct((H, B, NC, BT, BT), q.dtype),
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
      dimension_semantics=("parallel", "parallel", "parallel")
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


# =========================================================================
# Router
# =========================================================================

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
  segment_ids: jax.Array | None = None,
):
  """Fused gate cumsum + intra-chunk solve router.

  Routes to varlen or non-varlen dispatch based on cu_seqlens.

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
      g_cumsum = pallas_kda_gate_cumsum(
        g=g,
        scale=_RCP_LN2,
        chunk_size=chunk_size,
      )

    # S2: intra-chunk solve (uses original kernel without BC=16)
    w, u, qg, kg, Aqk, Akk = kda_fwd_intra(
      q=q, k=k, v=v, gk=g_cumsum, beta=beta,
      scale=scale, cu_seqlens=cu_seqlens,
      chunk_size=chunk_size, chunk_indices=chunk_indices,
      safe_gate=safe_gate, disable_recompute=disable_recompute,
      use_neumann=False,  # fp32 uses exact solve
    )

    # Reshape Aqk/Akk from 5D [H,B,NC,BT,BT] to 4D [H,B,T,BT]
    # (fp32 fallback returns raw Pallas output; bf16 paths do this in their wrappers)
    Aqk = Aqk.reshape(q.shape[0], q.shape[1], -1, Aqk.shape[-1])
    Akk = Akk.reshape(q.shape[0], q.shape[1], -1, Akk.shape[-1])
    return w, u, qg, kg, Aqk, Akk, g_cumsum

  # bf16: use fused BC=16 kernel
  if cu_seqlens is None:
    return pallas_kda_fwd_intra_fused(
      q=q, k=k, v=v, g=g, beta=beta,
      scale=scale, chunk_size=chunk_size,
      safe_gate=safe_gate, disable_recompute=disable_recompute,
      cumsum_scale=cumsum_scale,
      A_log=A_log, dt_bias=dt_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
    )
  else:
    return kda_fwd_intra_fused_varlen(
      q=q, k=k, v=v, g=g, beta=beta,
      scale=scale, cu_seqlens=cu_seqlens,
      chunk_size=chunk_size, chunk_indices=chunk_indices,
      safe_gate=safe_gate, disable_recompute=disable_recompute,
      cumsum_scale=cumsum_scale,
      A_log=A_log, dt_bias=dt_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
    )



# =============================================================================
# GLA output kernels used by KDA forward
# =============================================================================

import functools

import jax
import jax.experimental.pallas as pl
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.kda.utils import (
    exp,
    exp2,
    get_interpret,
    pad_to_multiple,
    prepare_chunk_indices,
)


# =============================================================================
# Sub-function 1: chunk_local_cumsum
# =============================================================================


def chunk_local_cumsum_ref(
  g: jax.Array,
  chunk_size: int,
  scale: float | None = None,
  reverse: bool = False,
  cu_seqlens_cpu: jax.Array | None = None,
) -> jax.Array:
  """Chunk-local cumulative sum of gates.

  Args:
      g: [B, T, H, K] — log-space gates (T must be a multiple of chunk_size)
      chunk_size: block size
      cu_seqlens: unused, kept for interface compatibility

  Returns:
      g_cumsum: [B, T, H, K] — chunk-local cumsum
  """
  B, T, H, K = g.shape
  assert reverse == False, "Reverse mode not supported in chunk_local_cumsum"
  assert T % chunk_size == 0, (
    "T must be a multiple of chunk_size for chunk_local_cumsum"
  )
  assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % chunk_size == 0).all(), (
    "cu_seqlens must be multiples of chunk_size for chunk_local_cumsum"
  )
  g = g.reshape(-1, H, K)
  C = chunk_size
  NT = B * T // C
  g = g.reshape(NT, C, H, K)
  g_cumsum = jnp.cumsum(g, axis=1).reshape(B, T, H, K)
  if scale is not None:
    g_cumsum = g_cumsum * scale
  return g_cumsum


def chunk_cumsum_kernel(
  cu_seqlens_ref,
  chunk_indices_ref,
  s_ref,
  o_ref,
  *,
  BT: int,
  BS: int,
  REVERSE: bool,
  HAS_SCALE: bool,
  scale: float,
  IS_VARLEN: bool,
):
  i_s, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  if IS_VARLEN:
    i_n, local_i_t = chunk_indices_ref[i_t, 0], chunk_indices_ref[i_t, 1]
    bos, eos = cu_seqlens_ref[i_n], cu_seqlens_ref[i_n + 1]
    start_t = bos + local_i_t * BT
  else:
    start_t = i_t * BT

  start_s = i_s * BS

  # Each program handles one (BT, BS) tile.
  s = s_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)]

  if IS_VARLEN:
    T_seq = eos - bos
    valid_len = T_seq - local_i_t * BT
    valid_mask = (jnp.arange(BT) < valid_len).astype(jnp.float32)[:, None]
    s = s.astype(jnp.float32) * valid_mask
  else:
    s = s.astype(jnp.float32)
  T = s.shape[0]

  if REVERSE:
    rows = [s[T - 1]]
    for i in range(T - 2, -1, -1):
      rows.append(rows[-1] + s[i])
    rows.reverse()
    o = jnp.stack(rows, axis=0)

  else:
    rows = [s[0]]
    for i in range(1, T):
      rows.append(rows[-1] + s[i])
    o = jnp.stack(rows, axis=0)

  if HAS_SCALE:
    o = o * scale

  o_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)] = o.astype(o_ref.dtype)


def _gla_chunk_local_cumsum_vector(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  cu_seqlens: jax.Array | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
  chunk_indices: jax.Array | None = None,
) -> jax.Array:

  assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
    "chunk_size must be power of 2"
  )

  if head_first:
    B, H, T, S = g.shape
    # Normalize to (B*H, T, S) to greatly simplify pointer offsets in the kernel.
    g_flat = g.reshape(B * H, T, S)
  else:
    B, T, H, S = g.shape
    g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

  BT = chunk_size
  BS = 128
  out_dtype = output_dtype or g.dtype
  HAS_SCALE = scale is not None
  scale_val = scale if scale is not None else 1.0

  interpret = get_interpret()

  # Pad the S dimension to satisfy TPU shape constraints.
  pad_S = (BS - (S % BS)) % BS
  if pad_S > 0:
    g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))

  S_padded = S + pad_S
  NS = S_padded // BS

  # For fixed-length inputs, synthesize cu_seqlens/chunk_indices to simplify kernel control flow.
  is_varlen = cu_seqlens is not None
  if is_varlen:
    if chunk_indices is None:
      chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)
  else:
    NT = (T + BT - 1) // BT
    # Dummy arrays for scalar prefetch (not used in the kernel when IS_VARLEN=False).
    cu_seqlens = jnp.zeros(1, dtype=jnp.int32)
    chunk_indices = jnp.zeros((1, 2), dtype=jnp.int32)
  grid = (NS, NT, B * H)

  # In varlen mode, append BT padding at the end to prevent dslice overflow.
  if is_varlen:
    g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))

  kernel = functools.partial(
    chunk_cumsum_kernel,
    BT=BT,
    BS=BS,
    REVERSE=reverse,
    HAS_SCALE=HAS_SCALE,
    scale=scale_val,
    IS_VARLEN=is_varlen,
  )

  o_flat = pl.pallas_call(
    kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=pl.no_block_spec,
      out_specs=pl.no_block_spec,
    ),
    out_shape=jax.ShapeDtypeStruct(g_flat.shape, out_dtype),
    interpret=interpret,
    compiler_params=pltpu.CompilerParams(
      disable_bounds_checks=True,
    ),
  )(cu_seqlens, chunk_indices, g_flat)

  # Remove the padding added earlier.
  o_flat = o_flat[:, :T, :S]

  # Convert the normalized layout back to the user-facing layout.
  if head_first:
    return o_flat.reshape(B, H, T, S)
  else:
    return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# =============================================================================
# Sub-function 2: chunk_fwd_h
# =============================================================================


def chunk_fwd_h_ref(
  k: jax.Array,
  v: jax.Array,
  gk: jax.Array | None = None,
  h0: jax.Array | None = None,
  output_final_state: bool = False,
  cu_seqlens_cpu: jax.Array | None = None,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
  """Inter-chunk hidden state propagation.

  Computes the hidden state at the start of each chunk by
  sequentially propagating through chunks.

  Args:
      k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
      v:  [B, T, H, V] — values
      gk: [B, T, H, K] — chunk-local cumsum of gates
      h0: [N, H, K, V] — initial hidden state (optional)
      output_final_state: whether to return final state
      cu_seqlens_cpu: unused, kept for interface compatibility
      chunk_size: block size

  Returns:
      h:  [B, NT, H, K, V] — hidden state at the start of each chunk
      ht: [B, H, K, V] or None — final hidden state
  """
  B, T, H, K = k.shape
  V = v.shape[-1]
  C = chunk_size
  NT = T // C
  N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
  assert T % C == 0, "T must be a multiple of chunk_size for chunk_fwd_h"
  assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
    "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
  )
  # seqlens = jnp.diff(cu_seqlens_cpu) if cu_seqlens_cpu is not None else None

  k = k.reshape(-1, H, K)
  v = v.reshape(-1, H, V)
  gk = gk.reshape(-1, H, K) if gk is not None else None
  h0 = h0.reshape(-1, H, K, V) if h0 is not None else None

  ht = jnp.zeros([N, H, K, V], dtype=jnp.float32)
  h_all = jnp.zeros([B, NT, H, K, V], dtype=k.dtype)
  is_varlen = cu_seqlens_cpu is not None
  for i_n in range(N):
    if not is_varlen:
      bos = i_n * T
      eos = (i_n + 1) * T
    else:
      bos = int(cu_seqlens_cpu[i_n])
      eos = int(cu_seqlens_cpu[i_n + 1])

    h = jnp.zeros((H, K, V), dtype=jnp.float32)
    if h0 is not None:
      h = h + h0[i_n].astype(jnp.float32)

    NT = (eos - bos) // C
    for i_t in range(NT):
      # varlen (B=1): use absolute chunk index; non-varlen: (batch, local_chunk)
      bi = 0 if is_varlen else i_n
      ti = bos // C + i_t if is_varlen else i_t
      h_all = h_all.at[bi, ti].set(h.astype(h_all.dtype))
      b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
      b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]
      if gk is not None:
        b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
        b_gk_last = b_gk[-1]  # [H, K]
        h *= jnp.exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

        b_k = b_k * jnp.exp(b_gk_last[None, :, :] - b_gk)  # b_gk_last -> [C, H, K]

      # h += jnp.einsum("chk,chv->hkv")
      h = h + lax.dot_general(
        b_k,
        b_v,
        dimension_numbers=(((0,), (0,)), ((1,), (1,))),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
      )
    if output_final_state:
      ht = ht.at[i_n].set(h.astype(ht.dtype))

  return h_all, ht


# =============================================================================
# Sub-function 3: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk_ref(
  q: jax.Array,
  k: jax.Array,
  g: jax.Array,
  scale: float,
  cu_seqlens_dev: np.ndarray | None = None,
  chunk_size: int = 64,
) -> jax.Array:
  """Intra-chunk attention matrix with causal mask.

  Args:
      q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
      k: [B, T, H, K] — keys
      g: [B, T, H, K] — chunk-local cumsum of gates
      scale: scaling factor
      cu_seqlens: unused, kept for interface compatibility
      chunk_size: block size

  Returns:
      A: [B, NT, C, H, C] — intra-chunk causal attention matrix
  """
  B, T, H, K = q.shape
  C = chunk_size
  NT = T // C

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  g_c = g.reshape(B, NT, C, H, K)

  # Midpoint stabilization: halves exponent range to prevent exp overflow
  # at large chunk sizes.  Identity: exp(g[i])*exp(-g[j]) = exp(g[i]-m)*exp(m-g[j])
  g_n = (g_c[:, :, 0:1, :, :] + g_c[:, :, -1:, :, :]) * 0.5  # [B, NT, 1, H, K]
  q_gated = q_c * jnp.exp(g_c - g_n)
  k_gated = k_c * jnp.exp(g_n - g_c)

  # [B, NT, H, C, K] @ [B, NT, H, K, C] -> [B, NT, H, C, C] -> [B, NT, C, H, C]
  A = (
    jnp.einsum(
      "bnihk,bnjhk->bnihj",
      q_gated,
      k_gated,
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )
  A = A.reshape(B, T, H, C)

  return A


# =============================================================================
# Pallas kernel: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk_pl(
  q_ref,
  k_ref,
  g_ref,  # in
  A_ref,  # out
  *,
  BT,
  scale,
):
  """GLA forward intra-chunk attention matrix Pallas kernel.

  Grid: (H, total_NT) where total_NT = B * NT.
  Refs (after block spec indexing):
    q_ref/k_ref/g_ref: (1, 1, BT, K)
    A_ref: (1, 1, BT, BT)
  """
  b_q = q_ref[0, 0]  # (BT, K)
  b_k = k_ref[0, 0]  # (BT, K)
  b_g = g_ref[0, 0].astype(jnp.float32)  # (BT, K)

  # Midpoint stabilization: halves exponent range to prevent exp overflow
  # at large chunk sizes. Identity: exp(g[i])*exp(-g[j]) = exp(g[i]-m)*exp(m-g[j])
  g_mid = (b_g[0:1, :] + b_g[-1:, :]) * 0.5
  b_qg = (b_q * jnp.exp(b_g - g_mid)).astype(b_q.dtype)
  b_kg = (b_k * jnp.exp(g_mid - b_g)).astype(b_k.dtype)

  b_A = (
    jnp.dot(
      b_qg,
      b_kg.T,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )

  A_ref[0, 0] = b_A.astype(A_ref.dtype)


def chunk_gla_fwd_intra_gk(
  q: jax.Array,  # [B, T, H, K]
  k: jax.Array,  # [B, T, H, K]
  g: jax.Array,  # [B, T, H, K]
  scale: float,
  chunk_size: int,
) -> jax.Array:
  """Launcher for chunk_gla_fwd_intra_gk Pallas kernel.

  Pre-reshapes inputs to (H, total_NT, ...) so the kernel is
  agnostic to batch/varlen structure.

  Returns:
      A: [B, T, H, BT] — intra-chunk attention matrix (float32)
  """
  B, T, H, K = q.shape
  BT = chunk_size
  NT = T // BT
  total_NT = B * NT

  # Reshape: [B, T, H, K] -> [B, NT, BT, H, K] -> [H, B*NT, BT, K]
  _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
  _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
  _g = g.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)

  # Block specs — grid = (H, total_NT)
  spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
  A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
  A_shape = jax.ShapeDtypeStruct([H, total_NT, BT, BT], jnp.float32)

  A = pl.pallas_call(
    functools.partial(chunk_gla_fwd_intra_gk_pl, BT=BT, scale=scale),
    grid=(H, total_NT),
    out_shape=A_shape,
    in_specs=[spec, spec, spec],
    out_specs=A_spec,
    compiler_params=pltpu.CompilerParams(
      # vmem_limit_bytes=32 * 1024 * 1024,
      disable_bounds_checks=True,
    ),
    interpret=get_interpret(),
  )(_q, _k, _g)

  # Post-reshape: [H, total_NT, BT, BT] -> [B, T, H, BT]
  A = A.reshape(H, B, NT, BT, BT)
  A = A.transpose(1, 0, 2, 3, 4)  # (B, H, NT, BT, BT)
  A = A.reshape(B, H, NT * BT, BT)  # (B, H, T, BT)
  A = A.transpose(0, 2, 1, 3)  # (B, T, H, BT)
  return A


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================


def chunk_gla_fwd_o_gk_ref(
  q: jax.Array,
  v: jax.Array,
  gk: jax.Array,
  A: jax.Array,
  h: jax.Array,
  scale: float,
  cu_seqlens_cpu: jax.Array | None = None,
  chunk_size: int = 64,
) -> jax.Array:
  """Combine inter-chunk and intra-chunk contributions to produce output.

  Args:
      q: [B, T, H, K] — queries (T must be a multiple of chunk_size)
      v: [B, T, H, V] — values
      gk: [B, T, H, K] — chunk-local cumsum of gates
      A: [B, T, H, BT] — intra-chunk attention matrix
      h: [B, NT, H, K, V] — hidden state at start of each chunk
      scale: scaling factor
      cu_seqlens: unused, kept for interface compatibility
      chunk_size: block size

  Returns:
      o: [B, T, H, V]
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  NT = B * T // C
  assert T % C == 0, "T must be a multiple of chunk_size for chunk_gla_fwd_o_gk_ref"
  assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
    "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
  )

  q = q.reshape(-1, C, H, K)
  v = v.reshape(-1, C, H, V)
  gk = gk.reshape(-1, C, H, K)
  h = h.reshape(-1, H, K, V)
  A = A.reshape(-1, C, H, C)

  qg = q * jnp.exp(gk)

  # Inter-chunk: o_inter = scale * (q_gated @ h)
  o_inter = scale * jnp.einsum("nchk,nhkv->nchv", qg, h)  # [C, K] @ [K, V] -> [C, V]

  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))[
    :, None, :
  ]  # (C, 1, C) → broadcasts to (NT, C, H, C)
  n_A = jnp.where(causal_mask, A, 0.0)

  # [C, C] @ [C, V] -> [C, V]
  # Intra-chunk: o_intra = A @ v, contract over j (key position within chunk)
  o_intra = jnp.einsum("nihj,njhv->nihv", n_A, v)

  o = (o_inter + o_intra).reshape(B, T, H, V)
  return o


# =============================================================================
# Backward sub-function 1: chunk_gla_bwd_dA
# =============================================================================


def chunk_gla_bwd_dA_ref(
  v: jax.Array,
  do: jax.Array,
  scale: float,
  cu_seqlens_cpu: jax.Array | None = None,
  chunk_size: int = 64,
) -> jax.Array:
  """Gradient of the intra-chunk attention matrix.

  Args:
      v:  [B, T, H, V] — values
      do: [B, T, H, V] — output gradient
      scale: scaling factor
      chunk_size: block size

  Returns:
      dA: [B, T, H, C] — lower-triangular masked gradient
  """
  B, T, H, V = v.shape
  C = chunk_size
  NT = T // C

  v_c = v.reshape(B, NT, C, H, V)
  do_c = do.reshape(B, NT, C, H, V)

  # dA[i,j] = scale * do[i] . v[j]  for j <= i
  dA = (
    jnp.einsum("bnihv,bnjhv->bnihj", do_c, v_c, precision=lax.Precision.HIGHEST) * scale
  )

  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
  dA = jnp.where(causal_mask[None, None, :, None, :], dA, 0.0)

  dA = dA.reshape(B, T, H, C)
  return dA


# =============================================================================
# Backward sub-function 2: chunk_gla_bwd_dv
# =============================================================================


def chunk_gla_bwd_dv_ref(
  k: jax.Array,
  g_cumsum: jax.Array,
  A: jax.Array,
  do: jax.Array,
  dh: jax.Array,
  cu_seqlens_cpu: jax.Array | None = None,
  chunk_size: int = 64,
) -> jax.Array:
  """Gradient of v: intra-chunk (A^T @ do) + inter-chunk (k_decay @ dh).

  Args:
      k:        [B, T, H, K]
      g_cumsum: [B, T, H, K]
      A:        [B, T, H, C] — intra-chunk attention matrix
      do:       [B, T, H, V]
      dh:       [B, NT, H, K, V]
      chunk_size: block size

  Returns:
      dv: [B, T, H, V]
  """
  B, T, H, K = k.shape
  V = do.shape[-1]
  C = chunk_size
  NT = T // C

  k_c = k.reshape(B, NT, C, H, K)
  gc_c = g_cumsum.reshape(B, NT, C, H, K)
  do_c = do.reshape(B, NT, C, H, V)
  A_c = A.reshape(B, NT, C, H, C)

  # Intra-chunk: dv[j] = sum_{i>=j} A[i,j] * do[i]
  # A is lower-triangular (nonzero when i >= j), keep those entries
  causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
  A_masked = jnp.where(causal_mask[None, None, :, None, :], A_c, 0.0)
  dv_intra = jnp.einsum(
    "bnihj,bnihv->bnjhv", A_masked, do_c, precision=lax.Precision.HIGHEST
  )

  # Inter-chunk: k_decay @ dh
  gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K] — gate cumsum at chunk end
  k_decay = k_c * jnp.exp(gn[:, :, None, :, :] - gc_c)  # [B, NT, C, H, K]
  dv_inter = jnp.einsum(
    "bnchk,bnhkv->bnchv", k_decay, dh, precision=lax.Precision.HIGHEST
  )

  dv = (dv_intra + dv_inter).reshape(B, T, H, V)
  return dv


# =============================================================================
# Backward sub-function 3: chunk_gla_bwd_dqk_intra
# =============================================================================


def chunk_gla_bwd_dqk_intra_ref(
  q: jax.Array,
  k: jax.Array,
  g_cumsum: jax.Array,
  dA: jax.Array,
  cu_seqlens_cpu: jax.Array | None = None,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
  """Intra-chunk dq, dk from the attention matrix gradient dA.

  Args:
      q:        [B, T, H, K]
      k:        [B, T, H, K]
      g_cumsum: [B, T, H, K]
      dA:       [B, T, H, C]
      chunk_size: block size

  Returns:
      dq: [B, T, H, K]
      dk: [B, T, H, K]
  """
  B, T, H, K = q.shape
  C = chunk_size
  NT = T // C

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  gc_c = g_cumsum.reshape(B, NT, C, H, K)
  dA_c = dA.reshape(B, NT, C, H, C)

  # Midpoint stabilization: halves exponent range to prevent exp overflow
  # at large chunk sizes.  Identity: exp(g[i])*exp(-g[j]) = exp(g[i]-m)*exp(m-g[j])
  g_mid = (gc_c[:, :, 0:1, :, :] + gc_c[:, :, -1:, :, :]) * 0.5

  # dq[i] = exp(gc[i]) * sum_{j<=i} dA[i,j] * k[j] * exp(-gc[j])
  # dA is already lower-triangular masked, so causal constraint is embedded
  k_neg = k_c * jnp.exp(g_mid - gc_c)
  dq = jnp.exp(gc_c - g_mid) * jnp.einsum(
    "bnihj,bnjhk->bnihk", dA_c, k_neg, precision=lax.Precision.HIGHEST
  )

  # dk[j] = exp(-gc[j]) * sum_{i>=j} dA[i,j] * q[i] * exp(gc[i])
  q_pos = q_c * jnp.exp(gc_c - g_mid)
  dk = jnp.exp(g_mid - gc_c) * jnp.einsum(
    "bnihj,bnihk->bnjhk", dA_c, q_pos, precision=lax.Precision.HIGHEST
  )

  dq = dq.reshape(B, T, H, K)
  dk = dk.reshape(B, T, H, K)
  return dq, dk


# =============================================================================
# Backward sub-function 4: chunk_gla_bwd_dqkg
# =============================================================================


def chunk_gla_bwd_dqkg_ref(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  h: jax.Array,
  g_cumsum: jax.Array,
  do: jax.Array,
  dh: jax.Array,
  dq: jax.Array,
  dk: jax.Array,
  scale: float,
  cu_seqlens_cpu: jax.Array | None = None,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Inter-chunk dq, dk contributions + gate gradient dg.

  Args:
      q:        [B, T, H, K]
      k:        [B, T, H, K]
      v:        [B, T, H, V]
      h:        [B, NT, H, K, V] — hidden states at chunk starts
      g_cumsum: [B, T, H, K]
      do:       [B, T, H, V]
      dh:       [B, NT, H, K, V]
      dq:       [B, T, H, K] — intra-chunk dq
      dk:       [B, T, H, K] — intra-chunk dk
      scale: scaling factor
      chunk_size: block size

  Returns:
      dq: [B, T, H, K] — intra + inter combined
      dk: [B, T, H, K] — intra + inter combined
      dg: [B, T, H, K] — gate gradient
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size
  NT = T // C

  q_c = q.reshape(B, NT, C, H, K)
  k_c = k.reshape(B, NT, C, H, K)
  v_c = v.reshape(B, NT, C, H, V)
  gc_c = g_cumsum.reshape(B, NT, C, H, K)
  do_c = do.reshape(B, NT, C, H, V)
  dq_c = dq.reshape(B, NT, C, H, K)
  dk_c = dk.reshape(B, NT, C, H, K)

  gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K]

  # Midpoint stabilization for inter-chunk dq to prevent exp overflow
  g_mid = (gc_c[:, :, 0:1, :, :] + gc_c[:, :, -1:, :, :]) * 0.5

  # Inter-chunk dq: scale * exp(gc) * (do @ h^T over V)
  # Split exp(gc) = exp(gc - g_mid) * exp(g_mid), absorb exp(g_mid) into h
  h_scaled = h * jnp.exp(g_mid[:, :, 0, :, :])[:, :, :, :, None]
  dq_inter = (
    scale
    * jnp.exp(gc_c - g_mid)
    * jnp.einsum("bnchv,bnhkv->bnchk", do_c, h_scaled, precision=lax.Precision.HIGHEST)
  )

  # Inter-chunk dk: exp(gn - gc) * (v @ dh^T over V)
  dk_inter = jnp.exp(gn[:, :, None, :, :] - gc_c) * jnp.einsum(
    "bnchv,bnhkv->bnchk",
    v_c,
    dh,
    precision=lax.Precision.HIGHEST,
  )

  # Combine intra + inter
  dq_total = dq_c + dq_inter
  dk_total = dk_c + dk_inter

  # Gate gradient
  # dgk_inter = exp(gn) * sum_v(h * dh) + sum_t(dk_inter * k)
  # Note: for negative g_gamma, exp(gn) underflows toward 0 (not NaN).
  # This is mathematically correct — the gate gradient is small because the
  # hidden state from the chunk start is mostly forgotten by the end.
  dgk_inter = jnp.exp(gn) * jnp.einsum(
    "bnhkv,bnhkv->bnhk", h, dh, precision=lax.Precision.HIGHEST
  ) + jnp.sum(dk_inter * k_c, axis=2)  # [B, NT, H, K]

  # dg_raw = q * dq_total - k * dk_total
  dg_raw = q_c * dq_total - k_c * dk_total  # [B, NT, C, H, K]

  # Reverse cumsum over time dimension within each chunk
  dg = (
    jnp.cumsum(dg_raw[:, :, ::-1, :, :], axis=2)[:, :, ::-1, :, :]
    + dgk_inter[:, :, None, :, :]
  )

  dq_out = dq_total.reshape(B, T, H, K)
  dk_out = dk_total.reshape(B, T, H, K)
  dg_out = dg.reshape(B, T, H, K)
  return dq_out, dk_out, dg_out


# =============================================================================
# Varlen padding helpers (JAX)
# =============================================================================


def _pad_varlen_seqs_jax(
  tensors: list[jax.Array],
  cu_seqlens_cpu: jax.Array,
  chunk_size: int,
) -> tuple[list[jax.Array], jax.Array, list[int] | None, list[int] | None]:
  """Pad each variable-length segment along dim=1 to a multiple of chunk_size.

  Args:
      tensors: list of [1, T_total, ...] arrays sharing the same sequence layout
      cu_seqlens_cpu: [N+1] cumulative sequence lengths on CPU
      chunk_size: block size

  Returns:
      (padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens)
      If no padding is needed, returns (tensors, cu_seqlens, None, None).
  """
  N = len(cu_seqlens_cpu) - 1
  C = chunk_size
  orig_seqlens = jnp.diff(cu_seqlens_cpu).tolist()
  padded_seqlens = [((L + C - 1) // C) * C for L in orig_seqlens]

  if orig_seqlens == padded_seqlens:
    return tensors, cu_seqlens_cpu, None, None

  padded = [[] for _ in tensors]
  for i in range(N):
    bos = int(cu_seqlens_cpu[i])
    L = orig_seqlens[i]
    pad = padded_seqlens[i] - L
    for j, t in enumerate(tensors):
      seg = t[:, bos : bos + L]
      if pad > 0:
        pw = ((0, 0), (0, pad)) + ((0, 0),) * (t.ndim - 2)
        seg = jnp.pad(seg, pw)
      padded[j].append(seg)

  padded_tensors = [jnp.concatenate(p, axis=1) for p in padded]
  offsets = [0]
  for pl in padded_seqlens:
    offsets.append(offsets[-1] + pl)
  new_cu_seqlens = jnp.array(offsets, dtype=jnp.int32)

  return padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens


def _unpad_varlen_seqs_jax(
  tensor: jax.Array,
  orig_seqlens: list[int],
  padded_seqlens: list[int],
) -> jax.Array:
  """Remove per-segment padding from a variable-length array along dim=1."""
  parts = []
  offset = 0
  for L, PL in zip(orig_seqlens, padded_seqlens):
    parts.append(tensor[:, offset : offset + L])
    offset += PL
  return jnp.concatenate(parts, axis=1)


# =============================================================================
# Orchestrator: chunk_gla_bwd
# =============================================================================


def chunk_gla_bwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array | None,
  g_gamma: jax.Array | None,
  g_cumsum: jax.Array | None,
  scale: float,
  initial_state: jax.Array | None,
  h: jax.Array | None,
  A: jax.Array | None,
  do: jax.Array,
  dht: jax.Array | None,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Chunk GLA backward orchestrator.

  Follows the FLA/Triton convention: h and A are passed from the forward
  pass to avoid recomputation. If None, they are recomputed internally.

  Pads inputs to a multiple of chunk_size, then calls the sub-functions.
  For varlen mode, each segment is individually padded to a multiple of
  chunk_size and cu_seqlens is updated accordingly.

  Args:
      q:  [B, T, H, K]
      k:  [B, T, H, K]
      v:  [B, T, H, V]
      g:  [B, T, H, K] — raw log-space gates (or None if g_gamma is used)
      g_gamma: broadcastable to [B, T, H, K] or None — constant gate
      g_cumsum: [B, T, H, K] or None — pre-computed chunk-local cumsum
      scale: scaling factor
      initial_state: [B, H, K, V] or [N, H, K, V] (varlen) or None
      h:  [B, NT, H, K, V] or None — hidden states from forward
      A:  [B, T, H, C] or None — intra-chunk attention from forward
      do: [B, T, H, V] — output gradient
      dht: [B, H, K, V] or [N, H, K, V] (varlen) or None — terminal state gradient
      cu_seqlens: [N+1] cumulative sequence lengths for variable-length mode.
          When provided, B must be 1. Segment lengths need not be multiples
          of chunk_size — they are padded internally.
      chunk_size: block size

  Returns:
      (dq, dk, dv, dg, dh0)
  """

  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size

  # Record if g was originally None
  g_orig = g

  # Broadcast g_gamma into full g if g is not provided
  if g is None:
    if g_gamma is not None:
      g = jnp.broadcast_to(g_gamma, q.shape)
    else:
      g = jnp.zeros_like(q)

  # --- padding ---
  orig_seqlens = None
  padded_seqlens = None

  if cu_seqlens is not None:
    assert B == 1
    [q, k, v, g, do], cu_seqlens, orig_seqlens, padded_seqlens = _pad_varlen_seqs_jax(
      [q, k, v, g, do], cu_seqlens, C
    )
  else:
    NT = (T + C - 1) // C
    T_padded = NT * C
    if T_padded > T:
      pad = T_padded - T
      pad_width = ((0, 0), (0, pad), (0, 0), (0, 0))
      q = jnp.pad(q, pad_width)
      k = jnp.pad(k, pad_width)
      v = jnp.pad(v, pad_width)
      g = jnp.pad(g, pad_width)
      do = jnp.pad(do, pad_width)

  # 1. Chunk-local cumsum
  if g_cumsum is None:
    g_cumsum = chunk_local_cumsum_ref(g, C, cu_seqlens_cpu=cu_seqlens)

  # 2. Forward replay to get h
  if h is None:
    h, _ = chunk_fwd_h_ref(
      k,
      v,
      gk=g_cumsum,
      h0=initial_state,
      output_final_state=False,
      cu_seqlens_cpu=cu_seqlens,
      chunk_size=C,
    )

  # 3. Backward hidden state gradients
  dh, dh0 = chunk_bwd_dh_ref(
    q,
    k,
    v,
    g=None,
    g_gamma=None,
    gk=g_cumsum,
    do=do,
    h0=initial_state,
    dht=dht,
    scale=scale,
    output_dh0=(initial_state is not None or dht is not None),
    cu_seqlens_cpu=cu_seqlens,
    chunk_size=C,
  )

  # 4. dv (uses A from forward)
  if A is None:
    A = chunk_gla_fwd_intra_gk_ref(q, k, g_cumsum, scale, chunk_size=C)
  dv = chunk_gla_bwd_dv_ref(
    k, g_cumsum, A, do, dh, cu_seqlens_cpu=cu_seqlens, chunk_size=C
  )

  # 5. dA
  dA = chunk_gla_bwd_dA_ref(v, do, scale, cu_seqlens_cpu=cu_seqlens, chunk_size=C)

  # 6. Intra-chunk dq, dk
  dq, dk = chunk_gla_bwd_dqk_intra_ref(
    q, k, g_cumsum, dA, cu_seqlens_cpu=cu_seqlens, chunk_size=C
  )

  # 7. Inter-chunk dq, dk + gate gradient
  dq, dk, dg = chunk_gla_bwd_dqkg_ref(
    q,
    k,
    v,
    h,
    g_cumsum,
    do,
    dh,
    dq,
    dk,
    scale,
    cu_seqlens_cpu=cu_seqlens,
    chunk_size=C,
  )

  # --- unpadding ---
  if orig_seqlens is not None:
    dq = _unpad_varlen_seqs_jax(dq, orig_seqlens, padded_seqlens)
    dk = _unpad_varlen_seqs_jax(dk, orig_seqlens, padded_seqlens)
    dv = _unpad_varlen_seqs_jax(dv, orig_seqlens, padded_seqlens)
    dg = _unpad_varlen_seqs_jax(dg, orig_seqlens, padded_seqlens)
  else:
    dq = dq[:, :T]
    dk = dk[:, :T]
    dv = dv[:, :T]
    dg = dg[:, :T]

  if g_orig is None and g_gamma is not None:
    # Sum-reduce dg to match g_gamma's shape
    for i, (d_full, d_gamma) in enumerate(zip(dg.shape[::-1], g_gamma.shape[::-1])):
      if d_full != d_gamma:
        dg = jnp.sum(dg, axis=len(dg.shape) - 1 - i, keepdims=True)
    missing_dims = len(dg.shape) - len(g_gamma.shape)
    if missing_dims > 0:
      dg = jnp.sum(dg, axis=tuple(range(missing_dims)))
    dg = dg.reshape(g_gamma.shape)

  return dq, dk, dv, dg, dh0


def chunk_gla_bwd_with_pl(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array | None,
  g_gamma: jax.Array | None,
  g_cumsum: jax.Array | None,
  scale: float,
  initial_state: jax.Array | None,
  h: jax.Array | None,
  A: jax.Array | None,
  do: jax.Array,
  dht: jax.Array | None,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Chunk GLA backward orchestrator using Pallas kernels."""

  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size

  # Record if g was originally None
  g_orig = g

  # Broadcast g_gamma into full g if g is not provided
  if g is None:
    if g_gamma is not None:
      g = jnp.broadcast_to(g_gamma, q.shape)
    else:
      g = jnp.zeros_like(q)

  assert T % C == 0, "T must be a multiple of chunk_size for chunk_gla_bwd_with_pl"
  assert (cu_seqlens is None) or (cu_seqlens % C == 0).all(), (
    "cu_seqlens must be multiples of chunk_size for chunk_gla_bwd_with_pl"
  )

  # 1. Chunk-local cumsum
  if g_cumsum is None:
    if g_gamma is not None and cu_seqlens is None:
      _, T_pad, _, _ = q.shape
      pos = jnp.arange(1, C + 1, dtype=jnp.float32)
      pos = jnp.tile(pos, T_pad // C).reshape(1, T_pad, 1, 1)
      g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)
    else:
      g_cumsum = _gla_chunk_local_cumsum_vector(g, C, cu_seqlens=cu_seqlens)

  # 2. Forward replay to get h
  if h is None:
    h, _ = chunk_fwd_h_kernel(
      k,
      v,
      gk=g_cumsum,
      h0=initial_state,
      output_final_state=False,
      cu_seqlens=cu_seqlens,
      chunk_size=C,
    )
    if cu_seqlens is None:
      h = h.reshape(B, T // C, H, K, V)

  # 3. Backward hidden state gradients
  dh, dh0 = chunk_bwd_dh_kernel(
    q,
    k,
    v,
    gk=g_cumsum,
    do=do,
    dht=dht,
    scale=scale,
    output_dh0=(initial_state is not None or dht is not None),
    cu_seqlens=cu_seqlens,
    chunk_size=C,
  )
  if cu_seqlens is None:
    dh = dh.reshape(B, T // C, H, K, V)
    if dh0 is not None:
      dh0 = dh0.reshape(B, H, K, V)

  # 4. Fused backward pass for dq, dk, dv, dg
  dq, dk, dv, dg = chunk_gla_bwd_fused_pl(
    q, k, v, g_cumsum, h, do, dh, scale=scale, chunk_size=C
  )

  if g_orig is None and g_gamma is not None:
    # Sum-reduce dg to match g_gamma's shape
    for i, (d_full, d_gamma) in enumerate(zip(dg.shape[::-1], g_gamma.shape[::-1])):
      if d_full != d_gamma:
        dg = jnp.sum(dg, axis=len(dg.shape) - 1 - i, keepdims=True)
    missing_dims = len(dg.shape) - len(g_gamma.shape)
    if missing_dims > 0:
      dg = jnp.sum(dg, axis=tuple(range(missing_dims)))
    dg = dg.reshape(g_gamma.shape)

  return dq, dk, dv, dg, dh0


# =============================================================================
# Orchestrator: chunk_gla_fwd
# =============================================================================


def chunk_gla_fwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array | None,
  g_gamma: jax.Array | None,
  g_cumsum: jax.Array | None,
  scale: float,
  initial_state: jax.Array | None,
  output_final_state: bool,
  cu_seqlens: jax.Array | None = None,
  chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array]:
  """Chunk GLA forward orchestrator.

  Pads inputs to a multiple of chunk_size (T dim) and 128 (K/V dims),
  then calls the 4 sub-functions.
  For varlen mode, each segment is individually padded to a multiple of
  chunk_size and cu_seqlens is updated accordingly.

  Returns:
      (g_cumsum, A, h, ht, o)
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  C = chunk_size

  # --- padding ---
  orig_seqlens = None
  padded_seqlens = None

  if g is None:
    if g_gamma is not None:
      g = jnp.broadcast_to(g_gamma, q.shape)
    else:
      g = jnp.zeros_like(q)

  if cu_seqlens is not None:
    assert B == 1
    [q, k, v, g], cu_seqlens, orig_seqlens, padded_seqlens = _pad_varlen_seqs_jax(
      [q, k, v, g], cu_seqlens, C
    )
  elif T % C != 0:
    q, k, v, g = (pad_to_multiple(x, C, axis=1, val=0) for x in (q, k, v, g))

  # K/V padding (chunk_fwd_h_kernel requires K%128==0, V%128==0)
  q, k, g = (pad_to_multiple(x, 128, axis=3, val=0) for x in (q, k, g))
  v = pad_to_multiple(v, 128, axis=3, val=0)
  if initial_state is not None:
    initial_state = pad_to_multiple(initial_state, [128, 128], axis=[2, 3], val=0)

  if g_cumsum is None:
    if g_gamma is not None and cu_seqlens is None:
      # Constant g_gamma: compute chunk-local cumsum analytically.
      # For constant g, cumsum within each chunk = g_gamma * [1, 2, ..., C].
      # This avoids _gla_chunk_local_cumsum_vector whose pallas kernel uses
      # no_block_spec and loads the full tensor into VMEM.
      _, T_pad, _, _ = q.shape
      pos = jnp.arange(1, C + 1, dtype=jnp.float32)
      pos = jnp.tile(pos, T_pad // C).reshape(1, T_pad, 1, 1)
      g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)
    else:
      g_cumsum = _gla_chunk_local_cumsum_vector(g, C, cu_seqlens=cu_seqlens)

  h, ht = chunk_fwd_h_kernel(
    k=k,
    v=v,
    g=None,
    g_gamma=None,
    gk=g_cumsum,
    h0=initial_state,
    output_final_state=output_final_state,
    cu_seqlens=cu_seqlens,
    chunk_size=C,
  )
  if cu_seqlens is None:
    h = h.reshape(k.shape[0], -1, k.shape[2], k.shape[3], v.shape[-1])
  else:
    h = h.reshape(1, -1, h.shape[1], h.shape[2], h.shape[3])

  A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
  o = chunk_gla_fwd_o_gk(
    q,
    v,
    g_cumsum,
    A,
    h,
    scale,
    chunk_size=C,
  )

  # --- unpadding ---
  o = o[..., :V]
  g_cumsum = g_cumsum[..., :K]
  h = h[..., :K, :V]
  if ht is not None:
    ht = ht[..., :K, :V]
  # T unpadding
  if orig_seqlens is not None:
    o = _unpad_varlen_seqs_jax(o, orig_seqlens, padded_seqlens)
  else:
    o = o[:, :T]
  return g_cumsum, A, h, ht, o


# =============================================================================
# Public API: chunk_gla
# =============================================================================


def chunk_gla(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array | None = None,
  g_gamma: jax.Array | None = None,
  scale: float | None = None,
  initial_state: jax.Array | None = None,
  output_final_state: bool = False,
  cu_seqlens: np.ndarray | None = None,
  chunk_size: int = 16,
) -> tuple[jax.Array, jax.Array | None]:
  """Chunked GLA — pure JAX implementation.

  Splits the sequence into blocks of chunk_size and computes in parallel
  within each block, propagating hidden states across blocks.

  Either ``g`` or ``g_gamma`` (or both) may be provided:
  - ``g``: [B, T, H, K] — per-step log-space gates.
  - ``g_gamma``: broadcastable to [B, T, H, K] — constant gate that is
    broadcast across the sequence.  Used only when ``g is None``.
  If neither is given, gates default to zero (no decay).

  Args:
      q: [B, T, H, K]
      k: [B, T, H, K]
      v: [B, T, H, V]
      g: [B, T, H, K] or None — gates (log-space, after logsigmoid)
      g_gamma: broadcastable to [B, T, H, K] or None — constant gate
      scale: scaling factor, default K^{-0.5}
      initial_state: [N, H, K, V]
      output_final_state: whether to return final state
      cu_seqlens: [N+1] variable-length cumulative lengths
      chunk_size: block size, default 16

  Returns:
      o: [B, T, H, V]
      final_state: [N, H, K, V] or None
  """
  dtype = q.dtype
  q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
  if g is not None:
    g = g.astype(jnp.float32)
  if g_gamma is not None:
    g_gamma = g_gamma.astype(jnp.float32)
  B, T, H, K = q.shape

  if scale is None:
    scale = K**-0.5

  _, _, _, ht, o = chunk_gla_fwd(
    q,
    k,
    v,
    g,
    g_gamma=g_gamma,
    g_cumsum=None,
    scale=scale,
    initial_state=initial_state,
    output_final_state=output_final_state,
    cu_seqlens=cu_seqlens,
    chunk_size=chunk_size,
  )
  final_state = ht if output_final_state else None
  return o.astype(dtype), final_state


# =============================================================================
# Pallas kernel: chunk_gla_fwd_o_gk (unified, handles both varlen and non-varlen)
# =============================================================================


def chunk_gla_fwd_o_gk_pl_kernel(
  q_ref,
  v_ref,
  g_ref,
  h_ref,
  A_ref,
  o_ref,
  *,
  BT,
  scale,
  USE_EXP2,
):
  """Unified GLA forward O+GK Pallas kernel.

  Block specs deliver exactly one chunk's data per grid point.
  No varlen logic, no K/V tiling, no pl.ds needed.

  Grid: (H, total_NT) where total_NT = B * NT.
  Refs (after block spec indexing):
    q_ref: (1, 1, BT, K)   g_ref: (1, 1, BT, K)
    v_ref: (1, 1, BT, V)   A_ref: (1, 1, BT, BT)
    h_ref: (1, 1, K, V)    o_ref: (1, 1, BT, V)
  """
  b_q = q_ref[0, 0]  # (BT, K)
  b_g = g_ref[0, 0]  # (BT, K)
  b_v = v_ref[0, 0]  # (BT, V)
  b_h = h_ref[0, 0]  # (K, V)
  b_A = A_ref[0, 0]  # (BT, BT)

  # Inter-chunk: scale * (q * exp(g)) @ h
  # Numerically-stable max-subtraction (softmax-style):
  #   (q * exp(g - g_max)) @ (h * exp(g_max)) == q @ h * exp(g)
  # but exp(g - g_max) <= 1 always, eliminating the +inf that arose
  # from the prior midpoint scheme when chunk-internal cumsum span
  # exceeded the bf16 exp2 range (~+/-127). When exp(g_max) underflows
  # to 0, the product is finite*0 = 0 (correct) instead of inf*0 = NaN.
  b_g_f32 = b_g.astype(jnp.float32)
  b_q_f32 = b_q.astype(jnp.float32)
  g_max = jnp.max(b_g_f32, axis=0, keepdims=True)  # (1, K)
  _exp_fn = exp2 if USE_EXP2 else exp
  b_qg = b_q_f32 * _exp_fn(b_g_f32 - g_max)
  b_h_scaled = b_h.astype(jnp.float32) * _exp_fn(g_max[0, :])[:, None]
  b_o = jnp.dot(
    b_qg,
    b_h_scaled,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  b_o *= scale

  # Intra-chunk: tril(A) @ v
  m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  b_A_f32 = jnp.where(m_s, b_A, 0.0).astype(jnp.float32)
  b_o += jnp.dot(
    b_A_f32,
    b_v.astype(jnp.float32),
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  o_ref[0, 0] = b_o.astype(o_ref.dtype)


@functools.partial(
  jax.jit,
  static_argnames=[
    "scale",
    "chunk_size",
    "use_exp2",
  ],
)
def chunk_gla_fwd_o_gk_pl(
  q: jax.Array,  # [H, B, T, K]
  v: jax.Array,  # [H, B, T, V]
  g: jax.Array,  # [H, B, T, K]
  A: jax.Array,  # [H, B, T, BT]
  h: jax.Array,  # [H, B, NT, K, V]
  scale: float,
  chunk_size: int,
  use_exp2: bool,
) -> jax.Array:
  """Unified launcher for chunk_gla_fwd_o_gk Pallas kernel.

  Pre-reshapes all inputs to (H, total_NT, ...) so the kernel is
  completely agnostic to batch/varlen structure.
  Works for both varlen (B=1, cu_seqlens aligned to BT) and non-varlen.
  """
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  total_NT = B * NT

  # Reshape: [B, T, H, K] -> [B, NT, BT, H, K] -> [H, B*NT, BT, K]
  _q = q.reshape(H, B, NT, BT, K).reshape(H, total_NT, BT, K)
  _g = g.reshape(H, B, NT, BT, K).reshape(H, total_NT, BT, K)
  _v = v.reshape(H, B, NT, BT, V).reshape(H, total_NT, BT, V)
  _A = A.reshape(H, B, NT, BT, BT).reshape(H, total_NT, BT, BT)
  # h: [H, B, NT, K, V] -> [H, B*NT, K, V]
  _h = h.reshape(H, total_NT, K, V)

  # Block specs — grid = (H, total_NT)
  q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
  g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
  v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
  A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
  h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))

  o_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)
  o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

  grid = (H, total_NT)
  o = pl.pallas_call(
    functools.partial(
      chunk_gla_fwd_o_gk_pl_kernel,
      BT=BT,
      scale=scale,
      USE_EXP2=use_exp2,
    ),
    grid=grid,
    out_shape=o_shape,
    in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
    out_specs=o_spec,
    compiler_params=pltpu.CompilerParams(
      # vmem_limit_bytes=32 * 1024 * 1024,
      disable_bounds_checks=True,
    ),
    interpret=get_interpret(),
  )(_q, _v, _g, _h, _A)

  # Post-process: (H, total_NT, BT, V) -> (H, B, T, V)
  # total_NT = B * NT
  o = o.reshape(H, B, NT, BT, V).reshape(H, B, NT * BT, V)
  return o


def chunk_gla_fwd_o_gk(
  q: jax.Array,  # [H, B, T, K]
  v: jax.Array,  # [H, B, T, V]
  g: jax.Array,  # [H, B, T, K]
  A: jax.Array,  # [H, B, T, BT]
  h: jax.Array,  # [H, B, NT, K, V]
  scale: float,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
  chunk_size: int = 64,
  use_exp2: bool = False,
  _cu_seqlens: jax.Array | None = None,
  _chunk_indices: jax.Array | None = None,
) -> jax.Array:
  """Dispatch chunk_gla_fwd_o_gk to the unified Pallas kernel.

  Both varlen and non-varlen take the same path — the launcher
  reshapes inputs so the kernel sees (H, total_NT, ...).
  """
  _, _, T, _ = q.shape
  assert T % chunk_size == 0
  # Stage3+4 cherry-pick: accept private aliases.
  if cu_seqlens is None and _cu_seqlens is not None:
    cu_seqlens = _cu_seqlens
  if chunk_indices is None and _chunk_indices is not None:
    chunk_indices = _chunk_indices
  if cu_seqlens is None:
    return chunk_gla_fwd_o_gk_pl(
      q,
      v,
      g,
      A,
      h,
      scale,
      chunk_size,
      use_exp2,
    )
  else:
    return chunk_kda_fwd_o_gk_varlen(
      q=q,
      v=v,
      g=g,
      A=A,
      h=h,
      scale=scale,
      chunk_size=chunk_size,
      use_exp2=use_exp2,
      cu_seqlens=cu_seqlens,
      chunk_indices=chunk_indices,
    )


def _chunk_kda_fwd_o_gk_varlen_kernel(
  q_ref,
  v_ref,
  g_ref,
  h_ref,
  A_ref,
  o_ref,
  *,
  BT,
  scale,
  USE_EXP2,
):
  b_q = q_ref[0, 0, 0]
  b_g = g_ref[0, 0, 0]
  b_v = v_ref[0, 0, 0]
  b_h = h_ref[0, 0, 0]
  b_A = A_ref[0, 0, 0]

  b_g_f32 = b_g.astype(jnp.float32)
  b_q_f32 = b_q.astype(jnp.float32)
  # Compute inter-chunk output: o = scale * q * exp2(g) @ h.
  # Use g[0] (first position, largest cumsum) as reference to avoid overflow/underflow:
  #   exp2(g[t]) = exp2(g[t] - g[0]) * exp2(g[0])
  # g[t] - g[0] ≤ 0 for all t (cumsum is monotonically decreasing), so exp2 is safe.
  # Factor exp2(g[0]) into h to preserve the matmul structure.
  _exp_fn = exp2 if USE_EXP2 else exp
  b_g_ref = b_g_f32[0:1, :]  # [1, K] — reference point
  b_qg = b_q_f32 * _exp_fn(jnp.maximum(b_g_f32 - b_g_ref, -126.0))
  # Scale h rows: h_scaled[k, v] = h[k, v] * exp2(g_ref[k])
  b_h_scaled = (
    b_h.astype(jnp.float32) * _exp_fn(jnp.maximum(b_g_ref[0], -126.0))[:, None]
  )
  b_o = jnp.dot(
    b_qg,
    b_h_scaled,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  b_o *= scale

  m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  b_A_f32 = jnp.where(m_s, b_A, 0.0).astype(jnp.float32)
  b_o += jnp.dot(
    b_A_f32,
    b_v.astype(jnp.float32),
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  o_ref[0, 0, 0] = b_o.astype(o_ref.dtype)


def chunk_kda_fwd_o_gk_varlen(
  q,
  v,
  g,
  A,
  h,
  scale,
  *,
  cu_seqlens,
  chunk_indices=None,
  chunk_size=64,
  use_exp2=False,
):
  assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
  assert chunk_indices is not None, "chunk_indices is not None"
  H, B, T, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  assert T % BT == 0
  NT = T // BT

  # Reshape to [H, B, NT, BT, X] for per-chunk block access.
  _q = q.reshape(H, B, NT, BT, K)
  _v = v.reshape(H, B, NT, BT, V)
  _g = g.reshape(H, B, NT, BT, K)
  _A = A.reshape(H, B, NT, BT, BT)
  # h: [H, B, NT, K, V] — already correct shape.
  _h = h

  q_spec = pl.BlockSpec([1, 1, 1, BT, K], index_map=lambda h, b, nt: (h, b, nt, 0, 0))
  g_spec = pl.BlockSpec([1, 1, 1, BT, K], index_map=lambda h, b, nt: (h, b, nt, 0, 0))
  v_spec = pl.BlockSpec([1, 1, 1, BT, V], index_map=lambda h, b, nt: (h, b, nt, 0, 0))
  A_spec = pl.BlockSpec([1, 1, 1, BT, BT], index_map=lambda h, b, nt: (h, b, nt, 0, 0))
  h_spec = pl.BlockSpec([1, 1, 1, K, V], index_map=lambda h, b, nt: (h, b, nt, 0, 0))
  o_shape = jax.ShapeDtypeStruct([H, B, NT, BT, V], v.dtype)
  o_spec = pl.BlockSpec([1, 1, 1, BT, V], index_map=lambda h, b, nt: (h, b, nt, 0, 0))

  o_r = pl.pallas_call(
    functools.partial(
      _chunk_kda_fwd_o_gk_varlen_kernel, BT=BT, scale=scale, USE_EXP2=use_exp2
    ),
    grid=(H, B, NT),
    out_shape=o_shape,
    in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
    out_specs=o_spec,
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    interpret=get_interpret(),
  )(_q, _v, _g, _h, _A)

  # Reshape back: [H, B, NT, BT, V] → [H, B, T, V]
  return o_r.reshape(H, B, T, V)


# =============================================================================
# Pallas kernel: chunk_gla_bwd_fused
# =============================================================================


def chunk_gla_bwd_fused_kernel(
  q_ref,
  k_ref,
  v_ref,
  g_ref,
  h_ref,
  a_ref,
  do_ref,
  dh_ref,
  dq_ref,
  dk_ref,
  dv_ref,
  dg_ref,
  *,
  BT: int,
  scale: float,
):
  b_q = q_ref[0, 0]
  b_k = k_ref[0, 0]
  b_v = v_ref[0, 0]
  b_g = g_ref[0, 0].astype(jnp.float32)
  b_h = h_ref[0, 0].astype(jnp.float32)
  b_a = a_ref[0, 0].astype(jnp.float32)  # [BT, BT] — true intra-chunk attention
  b_do = do_ref[0, 0]
  b_dh = dh_ref[0, 0].astype(jnp.float32)

  b_gn = b_g[BT - 1, :]

  # Midpoint stabilization: halves exponent range to prevent exp overflow
  # at large chunk sizes. Identity: exp(g[i])*exp(-g[j]) = exp(g[i]-m)*exp(m-g[j])
  g_mid = (b_g[0:1, :] + b_g[-1:, :]) * 0.5
  exp_g_s = jnp.exp(b_g - g_mid)  # stable exp(g), bounded by exp(range/2)
  exp_neg_g_s = jnp.exp(g_mid - b_g)  # stable exp(-g), bounded by exp(range/2)

  # 1. dA = do @ v^T * scale  (gradient of loss w.r.t. A; used for dq, dk)
  b_A_do = b_do.astype(b_v.dtype)
  b_dA = (
    jnp.dot(
      b_A_do,
      b_v.T,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )
  mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  b_dA = jnp.where(mask, b_dA, 0.0)

  # 2. dv — intra uses true A (not dA): dv[j] = sum_{i>=j} A[i,j] * do[i]
  b_a_masked = jnp.where(mask, b_a, 0.0)
  b_dv_intra = jnp.dot(
    b_a_masked.T.astype(b_do.dtype),
    b_do,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  k_decay = (b_k * jnp.exp(b_gn[None, :] - b_g)).astype(b_k.dtype)
  b_dv_inter = jnp.dot(
    k_decay,
    b_dh.astype(b_k.dtype),
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )
  b_dv = b_dv_intra + b_dv_inter
  dv_ref[0, 0] = b_dv.astype(dv_ref.dtype)

  # 3. dq
  k_neg = (b_k * exp_neg_g_s).astype(b_k.dtype)
  b_dq_intra = (
    jnp.dot(
      b_dA.astype(k_neg.dtype),
      k_neg,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * exp_g_s
  )
  b_dq_inter = jnp.dot(
    b_do,
    (b_h * jnp.exp(g_mid[0, :])[:, None]).astype(b_do.dtype).T,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  ) * (scale * exp_g_s)
  b_dq = b_dq_intra + b_dq_inter
  dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

  # 4. dk
  q_pos = (b_q * exp_g_s).astype(b_q.dtype)
  b_dk_intra = (
    jnp.dot(
      b_dA.T.astype(q_pos.dtype),
      q_pos,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * exp_neg_g_s
  )
  b_dk_inter = jnp.dot(
    b_v,
    b_dh.astype(b_v.dtype).T,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  ) * jnp.exp(b_gn[None, :] - b_g)
  b_dk = b_dk_intra + b_dk_inter
  dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)

  # 5. dg
  dgk_inter = jnp.exp(b_gn) * jnp.sum(b_h * b_dh, axis=1) + jnp.sum(
    b_dk_inter * b_k.astype(jnp.float32), axis=0
  )
  dg_raw = b_q.astype(jnp.float32) * b_dq - b_k.astype(jnp.float32) * b_dk

  # Use upper triangular matrix multiplication for reverse cumsum
  mask_upper = jnp.arange(BT)[None, :] >= jnp.arange(BT)[:, None]
  M_upper = jnp.where(mask_upper, 1.0, 0.0).astype(jnp.float32)
  dg_rev_cumsum = jnp.dot(
    M_upper,
    dg_raw,
    precision=jax.lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  b_dg = dg_rev_cumsum + dgk_inter[None, :]
  dg_ref[0, 0] = b_dg.astype(dg_ref.dtype)


def chunk_gla_bwd_fused_pl(
  q: jax.Array,  # [B, T, H, K]
  k: jax.Array,  # [B, T, H, K]
  v: jax.Array,  # [B, T, H, V]
  g: jax.Array,  # [B, T, H, K]
  h: jax.Array,  # [B, NT, H, K, V]
  do: jax.Array,  # [B, T, H, V]
  dh: jax.Array,  # [B, NT, H, K, V]
  scale: float,
  chunk_size: int,
  A: jax.Array | None = None,  # [B, T, H, chunk_size] — intra-chunk attention from fwd
):
  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = T // BT
  total_NT = B * NT

  if A is None:
    A = chunk_gla_fwd_intra_gk_ref(q, k, g, scale, chunk_size=BT)

  # Reshape
  _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
  _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
  _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
  _g = g.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
  _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
  _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
  _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
  # A: [B, T, H, BT] -> [B, NT, BT, H, BT] -> [H, total_NT, BT, BT]
  _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)

  # BlockSpecs
  grid = (H, total_NT)
  spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
  spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
  spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
  spec_A = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))

  dq_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype)
  dk_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype)
  dv_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)
  dg_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], g.dtype)

  dq, dk, dv, dg = pl.pallas_call(
    functools.partial(chunk_gla_bwd_fused_kernel, BT=BT, scale=scale),
    grid=grid,
    out_shape=[dq_shape, dk_shape, dv_shape, dg_shape],
    in_specs=[spec_K, spec_K, spec_V, spec_K, spec_h, spec_A, spec_V, spec_h],
    out_specs=[spec_K, spec_K, spec_V, spec_K],
    compiler_params=pltpu.CompilerParams(
      # vmem_limit_bytes=32 * 1024 * 1024, # 32 MB limit to be safe
      disable_bounds_checks=True,
    ),
    interpret=get_interpret(),
  )(_q, _k, _v, _g, _h, _A, _do, _dh)

  # Post-process
  def _unreshape(x, shape):
    x = x.reshape(H, B, NT, BT, shape[-1])
    x = x.transpose(1, 0, 2, 3, 4)
    x = x.reshape(B, H, T, shape[-1])
    return x.transpose(0, 2, 1, 3)

  dq = _unreshape(dq, [B, T, H, K])
  dk = _unreshape(dk, [B, T, H, K])
  dv = _unreshape(dv, [B, T, H, V])
  dg = _unreshape(dg, [B, T, H, K])

  return dq, dk, dv, dg



# =============================================================================
# KDA chunked forward orchestrator
# =============================================================================

"""KDA chunked forward pass orchestrator.

Decomposes Kernelized Delta Attention (KDA) into four pipelined stages:
  1. Gate cumsum       -- chunk-local cumsum in log2 space
  2. Intra-chunk solve -- Neumann-series triangular solve
  3. State propagation -- delta-rule recurrence across chunks
  4. Output            -- inter-chunk state + intra-chunk attention

Mirrors the four-stage pipeline of FLA's ``chunk_kda_fwd`` from
``fla.ops.kda.chunk_fwd``, using Neumann approximation for intra-chunk solve.
"""


import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tokamax._src.ops.experimental.kda.utils import get_tpu_config
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
  get_interpret,
  l2norm_fwd,
  normalize_initial_state,
  prepare_chunk_indices,
  segment_ids_to_cu_seqlens,
  segment_ids_to_seqlens,
)

_RCP_LN2 = 1.0 / math.log(2)


# =====================================================================
# === DEBUG: KDA NaN/Inf probe — default OFF, opt-in via env var ======
# Enable with KDA_DEBUG_NAN=1. When disabled, _probe() is a no-op and
# adds zero runtime cost (no jax.debug.print dispatches).
# Probes are inserted ONLY at HBM tensor boundaries in chunk_kda_fwd
# (kernel inputs/outputs). They do NOT modify any computation, do NOT
# enter Pallas kernel bodies.
# =====================================================================
_KDA_DEBUG_NAN = os.environ.get("KDA_DEBUG_NAN", "0") == "1"


def _probe(name: str, x):
  """Print NaN/Inf summary statistics of an HBM tensor at runtime.

  Uses jax.debug.print so it works inside jit / custom_vjp without
  breaking lowering. Must NEVER be called from inside a Pallas kernel
  body (TPU lowering does not support host callbacks).

  Args:
      name: Short label printed alongside the stats (≤30 chars recommended).
      x:    A JAX array on HBM, or None (in which case probe is skipped).
  """
  if not _KDA_DEBUG_NAN or x is None:
    return
  x_f32 = x.astype(jnp.float32)
  is_nan = jnp.isnan(x_f32)
  is_inf = jnp.isinf(x_f32)
  is_fin = jnp.isfinite(x_f32)
  n_nan = is_nan.sum()
  n_inf = is_inf.sum()
  finite = jnp.where(is_fin, x_f32, 0.0)
  f_min = jnp.where(is_fin, x_f32, jnp.inf).min()
  f_max = jnp.where(is_fin, x_f32, -jnp.inf).max()
  abs_max = jnp.abs(finite).max()
  jax.debug.print(
    "[KDA-PROBE] {n:32s} shape={s} dtype={d} | nan={nn} inf={ni} | "
    "fin_min={mn:.4e} fin_max={mx:.4e} abs_max={am:.4e}",
    n=name,
    s=x.shape,
    d=x.dtype.name,
    nn=n_nan,
    ni=n_inf,
    mn=f_min,
    mx=f_max,
    am=abs_max,
  )


def pallas_kda_gate_cumsum(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float = _RCP_LN2,
  head_first: bool = False,
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
    scale=scale,
    head_first=True,
    output_dtype=jnp.float32,
  )



# ---------------------------------------------------------------------------
# Fused Stage 3+4 kernel for VARIABLE-LENGTH sequences
# Adapted from b099ab6b's fixed-length _chunk_kda_fwd_h_o_kernel by adding:
#   - scalar prefetch (seqlens_ref, chunk_offsets_ref)
#   - per-sequence real_NT bounds via @pl.when(idx_nt < real_NT)
#   - per-sequence init (idx_nt == 0) and final-state store (idx_nt == real_NT - 1)
#   - _t_index_map mapping (n, h, nt) -> (0, h, bos // BT + nt, 0)
# ---------------------------------------------------------------------------


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


def chunk_kda_fwd(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g: jax.Array,
  beta: jax.Array,
  scale: float,
  initial_state: jax.Array,
  output_final_state: bool,
  use_qk_l2norm_in_kernel: bool = False,
  cu_seqlens: jax.Array | None = None,
  chunk_indices: jax.Array | None = None,
  chunk_size: int = 64,
  safe_gate: bool = True,
  lower_bound: float | None = None,
  use_gate_in_kernel: bool = False,
  A_log: jax.Array | None = None,
  dt_bias: jax.Array | None = None,
  disable_recompute: bool = False,
  cp_context: CPContext | None = None,
  _skip_align: bool = False,
  segment_ids: jax.Array | None = None,
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

  # === DEBUG: Stage 0 — input tensors ===========================
  _probe("0a.in.q", q)
  _probe("0b.in.k", k)
  _probe("0c.in.v", v)
  _probe("0d.in.g_raw", g)
  _probe("0e.in.beta", beta)
  if A_log is not None:
    _probe("0f.in.A_log", A_log)
    # Critical: exp(A_log) overflow risk (issue #1 in NaN report)
    _probe("0g.in.exp(A_log)", jnp.exp(A_log.astype(jnp.float32)))
  if dt_bias is not None:
    _probe("0h.in.dt_bias", dt_bias)
  if initial_state is not None:
    _probe("0i.in.initial_state", initial_state)
  # === END DEBUG ===============================================

  # --- Unsupported parameters ---
  assert use_qk_l2norm_in_kernel is False, (
    "use_qk_l2norm_in_kernel not yet supported in Pallas"
  )

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

  assert_shape(q, (H, B, T, K), "q")
  assert_shape(k, (H, B, T, K), "k")
  assert_shape(v, (H, B, T, V), "v")
  assert_shape(g, (H, B, T, K), "g")
  assert_shape(beta, (H, B, T), "beta")
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

  # === DEBUG: Stage 1+2 — fused gate cumsum + intra-chunk solve ==
  _probe("1a.s12.g_cumsum", g_cumsum)
  _g_step = g_cumsum[:, 1:] - g_cumsum[:, :-1]
  _probe("1b.s12.g_cumsum.step_diff", _g_step)
  _probe("2a.s12.w", w)
  _probe("2b.s12.u", u)
  _probe("2c.s12.qg", qg)
  _probe("2d.s12.kg", kg)
  _probe("2e.s12.Aqk", Aqk)
  _probe("2f.s12.Akk", Akk)
  # === END DEBUG ===============================================

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
  # Stage 1/2 already operate in BT-aligned layout (q/k/v/g were aligned at
  # L413-L418, derived tensors kg/w/u/Aqk/g_cumsum inherit that layout).
  # Reuse cu_seqlens / chunk_indices directly — no second gather needed.
  # The final _unalign_output at L596-L598 maps `o` back to original T.
  #
  # Varlen path: use fused Stage 3+4 (chunk_kda_fwd_h_o_varlen) which keeps
  # the hidden state in VMEM scratch and v_new in registers, eliminating
  # the intermediate h [B,NT,H,K,V] and v_new [B,T,H,V] HBM tensors.
  # When disable_recompute=True we additionally spill h/v_new for bwd reuse.
  # Non-varlen path keeps the unfused two-kernel chain.
  if cu_seqlens is not None:
    o, final_state, h_fused, v_new_fused = chunk_kda_fwd_h_o_varlen(
      w=w,
      u=u,
      kg=kg,
      gk=g_cumsum,
      q=q,
      A=Aqk,
      cu_seqlens=cu_seqlens,
      chunk_indices=chunk_indices,
      initial_state=initial_state,
      output_final_state=output_final_state,
      scale=scale,
      chunk_size=BT,
      store_h=disable_recompute,
      store_v_new=False,
    )
    # When disable_recompute=True, fused path spills h for bwd reuse.
    # v_new is NOT spilled — bwd Stage 0 derives it cheaply in parallel
    # via compute_v_new_from_h_pallas, avoiding the sequential recurrence.
    h = h_fused
    v_new = v_new_fused

    # === DEBUG: Stage 3+4 — fused output =========================
    _probe("4.s4.o", o)
    if final_state is not None:
      _probe("3c.s3.final_state", final_state)
    # === END DEBUG ===============================================
  else:
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
      k=kg,
      w=w,
      u=u,
      gk=g_cumsum,
      initial_state=initial_state,
      output_final_state=output_final_state,
      chunk_size=BT,
      use_exp2=True,
      _cu_seqlens=cu_seqlens,
      _chunk_indices=chunk_indices if cu_seqlens is not None else None,
    )

    # === DEBUG: Stage 3 — inter-chunk state ======================
    _probe("3a.s3.h", h)
    _probe("3b.s3.v_new", v_new)
    if final_state is not None:
      _probe("3c.s3.final_state", final_state)
    # === END DEBUG ===============================================

    o = chunk_gla_fwd_o_gk(
      q=q,
      v=v_new,
      g=g_cumsum,
      A=Aqk,
      h=h,
      scale=scale,
      chunk_size=BT,
      use_exp2=True,
      _cu_seqlens=cu_seqlens,
      _chunk_indices=chunk_indices if cu_seqlens is not None else None,
    )

    # === DEBUG: Stage 4 — output =================================
    _probe("4.s4.o", o)
    # === END DEBUG ===============================================

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
