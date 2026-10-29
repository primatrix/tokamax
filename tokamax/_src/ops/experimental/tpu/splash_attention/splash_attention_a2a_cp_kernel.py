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

"""Splash attention with the Ulysses (a2a) context-parallel exchange fused in.

This module is a copy of ``splash_attention_kernel.py``.  The attention math
(``flash_attention_kernel`` body, mask processing and the backward kernels) is
unchanged; only the forward driver differs:

* the native forward is a gridded ``pallas_call`` over (head, active block);
* here a single ``pallas_call`` owns, per local head, the q/k/v sequence-to-head
  all-to-all, a manual active-block loop with double-buffered tile DMA, and the
  o head-to-sequence all-to-all.  Incoming(h+1) overlaps compute(h) and
  outgoing(h) overlaps compute(h+1).

With ``cp_size == 1`` the same block loop runs without any exchange.

Calling convention for ``cp_size > 1`` (inside the caller's ``shard_map`` over
``cp_axis``): q/k/v are the local sequence shards ``[heads, N / cp, d]``,
segment ids cover the full sequence ``[N]``.  ``o`` is returned sequence
sharded ``[heads, N / cp, d_v]``; residual stats are head sharded
``[heads / cp, N]``, exactly as the native kernel returns them after the
pre-kernel all-to-all.
"""

from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import math
from typing import Any, NamedTuple

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as mask_info_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as _native


P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
SUPPORTS_GROUPED_SEGMENT_IDS = True
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed
TN_DIM_NUMBERS = (((0,), (0,)), ((), ()))  # LHS transposed

LOG2E = math.log2(math.e)
LOG2E_INV = 1 / LOG2E

# mypy: ignore-errors


def _not(x: jax.Array | bool) -> jax.Array | bool:
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


# Shared with the native kernel so callers pass the same types to either
# implementation.
SegmentIds = _native.SegmentIds


MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ["dq", "dkv", "fwd"]
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = "_residuals" if save_residuals else "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}"


# Splash attention implementation


QKVLayout = _native.QKVLayout


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


def _group_segment_spec(ids, num_heads, tile_size, seq_minor, width, sequence_axis, unravel):
  """Load one ID array for each contiguous group of attention heads."""
  groups, length = ids.shape
  block_shape = (width, tile_size) if seq_minor else (tile_size, width)
  array_shape = (groups, width, length) if seq_minor else (groups, length, width)
  expanded = ids[:, None, :] if seq_minor else ids[:, :, None]

  def index_map(h, i, j):
    tile = (i, j)[sequence_axis]
    group = _div(h, num_heads // groups)
    return (group, 0, tile) if seq_minor else (group, tile, 0)

  return (
      pl.BlockSpec((None, *block_shape), unravel(index_map)),
      jnp.broadcast_to(expanded, array_shape),
  )


def _block_mask_value(ref, grid_idx, head, num_heads):
  """Read a shared mask or a mask for a contiguous group of heads."""
  if ref.ndim == 2:
    group = head // (num_heads // ref.shape[0])
    return ref[group, grid_idx].astype(jnp.int32)
  return ref[grid_idx].astype(jnp.int32)


SplashConfig = _native.SplashConfig


to_i32 = lambda x: x.astype(jnp.int32)


def _use_native_layout(
    config, q, k, v, segment_ids, mask_info, mask_function, *, is_mqa
):
  """Select the sequence-minor layout without changing attention semantics.

  This is a compile-time implementation choice, not a caller tuning flag.
  In particular, never turn online softmax into fixed-shift softmax here.
  Retain the caller's tile sizes: MaskInfo was constructed for those tiles.
  """
  return (
      not is_mqa
      and q.ndim == k.ndim == v.ndim == 3
      and q.shape[0] == k.shape[0] == v.shape[0]
      and q.dtype == k.dtype == v.dtype == jnp.bfloat16
      and config.q_layout == config.k_layout == config.v_layout
      == QKVLayout.SEQ_MINOR
      and config.max_logit_const == 0.0
      and config.use_base2_exp
      and config.combine_log2_scale
      and config.softmax_scale is not None
      and config.attn_logits_soft_cap is None
      and segment_ids is not None
      and mask_info.partial_mask_blocks is None
      and mask_function is None
  )


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
    has_partial_mask: bool = False,
    kv_segment_ids_seq_minor: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask)
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (bq, k_slice.size), 1
        )

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(
            q_sequence_ref[...], (1, repeats)
        )  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (k_slice.size, bq), 0
        )
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            "Mask function must return a boolean-valued array, but got:"
            f" {computed_mask.dtype}"
        )
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      if kv_segment_ids_seq_minor:
        kv_ids = jnp.broadcast_to(
            kv_segment_ids_ref[:1, k_slice].T,
            (k_slice.size, bq),
        )
      else:
        repeats, rem = divmod(bq, NUM_LANES)
        if rem:
          raise NotImplementedError(
              f"block_q must be a multiple of {NUM_LANES}"
          )
        kv_ids = jnp.tile(
            kv_segment_ids_ref[k_slice, :], (1, repeats)
        )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    l_linear_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
    native_layout: bool = False,
    grid_idx: jax.Array | None = None,
    head_idx: jax.Array | None = None,
    num_heads: int | None = None,
):
  # The fused driver has no grid: it passes the active-block index, the head
  # index within the global head numbering and the head count explicitly.
  # Without them the body reads the native grid exactly as the native kernel.
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  if grid_idx is None:
    grid_idx = pl.program_id(1)
  h = pl.program_id(0) if head_idx is None else head_idx
  if num_heads is None:
    num_heads = pl.num_programs(0)

  should_run = True
  if block_mask_ref is not None:
    kind = _block_mask_value(block_mask_ref, grid_idx, h, num_heads)
    should_not_mask = kind != 1
    if block_mask_ref.ndim == 2:
      # A union tile can be inactive for this head group. Initialization and
      # output stores still follow the shared grid's row boundaries.
      should_run = kind != 0
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
  else:
    should_not_mask = False
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    max_logit_estimate = max_logit_value_ref[0, h]

  if config.use_base2_exp and max_logit_estimate is not None:
    max_logit_estimate *= LOG2E

  @pl.when(should_initialize)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    sink = None
    if sinks_ref is not None:
      sink = sinks_ref[0, h].astype(m_scratch_ref.dtype)
      if config.use_base2_exp:
        sink *= LOG2E

    if sinks_ref is None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is None and max_logit_estimate is not None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is not None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:  # sinks_ref is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = exp(
          sink - jnp.full_like(l_scratch_ref, max_logit_estimate)
      )

  def store_output_stat(ref, value):
    ref[...] = (
        jnp.broadcast_to(value[:, :1].T, (NUM_SUBLANES, bq))
        if config.compact_stats_output
        else value
    ).astype(ref.dtype)

  resident_q = q_ref[...] if native_layout else None

  def kvmajor_body(kv_compute_index, has_partial_mask):
    # Keep P as [KV, Q] so V @ P accumulates directly into [D, Q].
    # This avoids padding D in the minor axis of the output scratch buffer.
    window = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    logits = lax.dot_general(
        k_ref[:, window], resident_q, TN_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    logits *= jnp.float32(config.softmax_scale * LOG2E)
    if not config.segment_mask_on_partial_only or has_partial_mask:
      q_ids = (
          q_segment_ids_ref[:1, :]
          if q_segment_ids_ref.shape[0] == 1
          else q_segment_ids_ref[:, :1].T
      )
      kv_ids = kv_segment_ids_ref[:1, window].T
      logits = jnp.where(kv_ids == q_ids, logits, mask_value)
    probabilities = jnp.exp2(logits - max_logit_estimate)
    current_l = jnp.sum(probabilities, axis=0, keepdims=True)
    l_scratch_ref[...] += jnp.broadcast_to(current_l, l_scratch_ref.shape)
    # The reference PV consumes FP32 P; do not introduce a BF16 cast here.
    values = v_ref[:, window]
    # Pad only the non-reducing PV dimension; keep stored values and gradients
    # at their original width while improving narrow-head MXU mapping.
    pv_width = 120 if values.shape[0] == 72 else NUM_LANES
    values = jnp.pad(values, ((0, max(0, pv_width - values.shape[0])), (0, 0)))
    output_t = lax.dot_general(
        values, probabilities, NN_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    o_scratch_ref[...] += output_t[:v_ref.shape[0], :]

  def body(kv_compute_index, _, has_partial_mask=False):
    if native_layout:
      kvmajor_body(kv_compute_index, has_partial_mask)
      return
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (bq, NUM_LANES)
    assert l_prev.shape == (bq, NUM_LANES)

    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
    if config.use_base2_exp and config.softmax_scale is None:
      q *= LOG2E

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
    else:
      k = k_ref[:, slice_k]
    qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)
    if config.softmax_scale is not None:
      if config.use_base2_exp and config.combine_log2_scale:
        qk *= jnp.float32(config.softmax_scale * LOG2E)
      else:
        qk *= jnp.float32(config.softmax_scale)
        if config.use_base2_exp:
          qk *= jnp.float32(LOG2E)

    assert qk.shape == (bq, bkv_compute)
    apply_mask_and_soft_cap = functools.partial(
        _apply_mask_and_soft_cap,
        qk,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref
        if (not config.segment_mask_on_partial_only or has_partial_mask)
        else None,
        kv_segment_ids_ref
        if (not config.segment_mask_on_partial_only or has_partial_mask)
        else None,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )

    qk = apply_mask_and_soft_cap()

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[:, None]  # pytype: disable=attribute-error
      assert m_curr.shape == (bq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (bq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(
          f"{bkv_compute=} should be a multiple of {NUM_LANES}"
      )

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(qk - jnp.tile(m_next, (1, bkv_repeats)))
    else:
      s_curr = exp(qk - max_logit_estimate)
    assert s_curr.shape == (bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    assert l_curr.shape == (bq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[...] = l_curr + l_prev

    sv_dims = (
        NN_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
    else:
      v = v_ref[:, slice_k]
    o_curr = lax.dot_general(s_curr, v, sv_dims)

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, head_dim_v_repeats))
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = (
      k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  )

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(jnp.logical_not(should_not_mask), should_run))
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
    )

  @pl.when(should_write)
  def end():
    if native_layout:
      # Keep the exact reciprocal -> multiply -> BF16 cast sequence. Avoid
      # transposing FP32 output or broadcasting statistics to [Q, 128].
      l_native = l_scratch_ref[...]
      m_native = m_scratch_ref[...]
      output_native = o_scratch_ref[...]
      if fuse_reciprocal:
        # Empty group rows retain their initialized zero output.
        denominator = jnp.where(l_native[:1, :] == 0, 1.0, l_native[:1, :])
        output_native *= 1.0 / denominator
      output_native = output_native.astype(o_ref.dtype)
      o_ref[...] = output_native
      if logsumexp_ref is not None:
        log = jnp.log2 if config.use_base2_exp else jnp.log
        logsumexp_ref[...] = (m_native + log(l_native)).astype(logsumexp_ref.dtype)
      if l_linear_ref is not None:
        l_linear_ref[...] = l_native.astype(l_linear_ref.dtype)
      if max_logits_ref is not None:
        max_logits_ref[...] = m_native.astype(max_logits_ref.dtype)
      return
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      denominator = jnp.where(l == 0, 1.0, l)
      l_inv = jnp.tile(1.0 / denominator, (1, head_dim_v_repeats))
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (
          (NUM_SUBLANES, bq) if config.compact_stats_output else (bq, NUM_LANES)
      )
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      store_output_stat(logsumexp_ref, logsumexp)
    if l_linear_ref is not None:
      assert l_linear_ref.shape == (
          (NUM_SUBLANES, bq) if config.compact_stats_output else (bq, NUM_LANES)
      )
      store_output_stat(l_linear_ref, l)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (
          (NUM_SUBLANES, bq) if config.compact_stats_output else (bq, NUM_LANES)
      )
      store_output_stat(max_logits_ref, m)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
  if x is None:
    return 0

  if jnp.issubdtype(x.dtype, jnp.floating):
    info = jnp.finfo
  elif jnp.issubdtype(x.dtype, jnp.integer):
    info = jnp.iinfo
  else:
    raise ValueError(f"Unsupported dtype: {x.dtype}")
  return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)


_DEFAULT_FWD_VMEM_LIMIT_BYTES = 56 * 1024 * 1024


def _whole_spec(shape, memory_space):
  rank = len(shape)
  return pl.BlockSpec(shape, lambda *_: (0,) * rank, memory_space=memory_space)


def _flat(x: jax.Array, layout: QKVLayout) -> jax.Array:
  """[heads, seq, dim] -> the flattened 2D physical layout of ``layout``.

  HEAD_DIM_MINOR is [heads * seq, dim]; SEQ_MINOR is [heads * dim, seq].
  """
  heads, seq, dim = x.shape
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return x.reshape(heads * seq, dim)
  return x.mT.reshape(heads * dim, seq)


def _unflat(x: jax.Array, layout: QKVLayout, heads: int) -> jax.Array:
  """Inverse of ``_flat``."""
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return x.reshape(heads, x.shape[0] // heads, x.shape[1])
  return x.reshape(heads, x.shape[0] // heads, x.shape[1]).mT


def _seq_tile(ref, layout, head, dim, seq_len, start, size):
  """``size`` sequence positions of one head inside a ``_flat`` ref."""
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return ref.at[pl.ds(head * seq_len + start, size), :]
  return ref.at[pl.ds(head * dim, dim), pl.ds(start, size)]


def _stat_tile(ref, compact, head, seq_len, start, size):
  """``size`` query positions of one head inside a flattened stats ref."""
  if compact:
    return ref.at[pl.ds(head * NUM_SUBLANES, NUM_SUBLANES), pl.ds(start, size)]
  return ref.at[pl.ds(head * seq_len + start, size), :]


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    cp_size: int,
    cp_axis: str | None,
    max_logit_value: jax.Array | None = None,
    save_max_logits: bool = True,
):
  """Forward pass with the Ulysses exchange fused into one pallas call.

  With ``cp_size > 1`` this runs inside the caller's ``shard_map`` over
  ``cp_axis``: q/k/v are local sequence shards ``[heads, N / cp, d]`` and the
  segment ids cover the full sequence.  Each rank attends ``heads / cp`` heads
  over the full sequence, exactly like the native kernel after a pre-kernel
  all-to-all, and returns ``o`` sequence sharded.

  Returns ``(out, stats, buffers)``.  ``stats`` is None unless
  ``save_residuals``.  ``buffers`` holds the head-sharded q/k/v (``_flat``
  layout when ``cp_size > 1``) and the head-sharded ``o`` for the backward.
  """
  del fwd_mask_sparsity  # Only used by the native cost estimate.
  native_layout = (
      _use_native_layout(
          config, q, k, v, segment_ids, mask_info, mask_function, is_mqa=is_mqa
      )
      and sinks is None
      and max_logit_value is None
  )
  # TODO(step 2): native_layout, partial mask blocks (NumPy / dynamic masks),
  # grouped segment ids, grouped block masks, GQA and the unfused reciprocal
  # under cp > 1 still need their tile DMA in the fused driver.
  if native_layout:
    raise NotImplementedError("a2a cp: native_layout is not implemented yet.")
  if mask_info.partial_mask_blocks is not None:
    raise NotImplementedError("a2a cp: partial mask blocks are not implemented yet.")
  if segment_ids is not None and segment_ids.q.ndim == 2:
    raise NotImplementedError("a2a cp: grouped segment ids are not implemented yet.")
  if mask_info.block_mask is not None and mask_info.block_mask.ndim == 2:
    raise NotImplementedError("a2a cp: grouped block masks are not implemented yet.")
  if is_mqa:
    raise ValueError("a2a cp shards heads; MQA has a single kv head.")

  if cp_size < 1:
    raise ValueError(f"{cp_size=} must be >= 1.")
  if cp_size > 1:
    if cp_axis is None:
      raise ValueError("cp_axis is required when cp_size > 1.")
    axis_size = lax.axis_size(cp_axis)
    if axis_size != cp_size:
      raise ValueError(f"{cp_size=} != size of mesh axis {cp_axis!r} ({axis_size}).")

  num_q_heads, local_seq_len, head_dim_qk = q.shape
  if k.ndim != 3 or k.shape[-1] != head_dim_qk:
    raise ValueError(f"Expected 'key' [heads, seq, {head_dim_qk}], got {k.shape}.")
  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"'key' {k.shape} and 'value' {v.shape} leading dims differ.")
  if k.shape[0] != num_q_heads:
    raise NotImplementedError("a2a cp: GQA is not implemented yet.")
  if k.shape[1] != local_seq_len:
    raise ValueError("a2a cp needs equal q and kv sequence shards.")
  head_dim_v = v.shape[-1]
  if num_q_heads % cp_size:
    raise ValueError(f"{cp_size=} must divide {num_q_heads=}.")
  heads_local = num_q_heads // cp_size
  seq_len = local_seq_len * cp_size

  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  if seq_len % bq or seq_len % bkv:
    raise ValueError(f"{bq=} and {bkv=} must divide {seq_len=}.")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")
  kv_steps = seq_len // bkv
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  save_max_logits = save_max_logits or not fuse_reciprocal
  if not fuse_reciprocal and cp_size > 1:
    raise NotImplementedError(
        "a2a cp: fuse_reciprocal=False needs the division before the o exchange."
    )
  compact = config.compact_stats_output

  if segment_ids is not None:
    for name, ids in (("q", segment_ids.q), ("kv", segment_ids.kv)):
      if ids.shape != (seq_len,):
        raise ValueError(
            f"{name} segment_ids must cover the full sequence ({seq_len},), "
            f"got {ids.shape}."
        )
  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(
        f"Only one of {config.max_logit_const=} and {max_logit_value=} can be set."
    )
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(
        jnp.atleast_1d(max_logit_value).astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    sinks = jnp.broadcast_to(
        sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )

  q_layout, k_layout, v_layout = config.q_layout, config.k_layout, config.v_layout
  head_dim_minor = QKVLayout.HEAD_DIM_MINOR
  dynamic_grid = mask_info.active_rows is not None
  if dynamic_grid:
    num_blocks = mask_info.active_rows.shape[0]
    bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  else:
    num_blocks = (seq_len // bq) * kv_steps

  if cp_size > 1:
    rank = lax.axis_index(cp_axis).astype(jnp.int32).reshape((1,))
  else:
    rank = jnp.zeros((1,), jnp.int32)

  # Operands are keyed by name; absent optional operands are simply omitted.
  prefetch = {"rank": rank}
  if dynamic_grid:
    prefetch.update(
        active_rows=mask_info.active_rows,
        active_cols=mask_info.active_cols,
        bounds_start=bounds_start,
        bounds_end=bounds_end,
    )
  if mask_info.block_mask is not None:
    prefetch["block_mask"] = mask_info.block_mask

  hbm_in = {
      "q": _flat(q, q_layout),
      "k": _flat(k, k_layout),
      "v": _flat(v, v_layout),
  }
  if segment_ids is not None:
    hbm_in["q_ids"] = lax.broadcast_in_dim(
        segment_ids.q, (seq_len, NUM_LANES), (0,)
    )
    hbm_in["kv_ids"] = lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, seq_len), (1,)
    )
  if mask_info.q_sequence is not None:
    hbm_in["q_seq"] = lax.broadcast_in_dim(
        mask_info.q_sequence, (seq_len, NUM_LANES), (0,)
    )
  smem_in = {}
  if sinks is not None:
    smem_in["sinks"] = sinks
  if max_logit_value is not None:
    smem_in["max_logit"] = max_logit_value

  def flat_shape(layout, heads, seq, dim):
    return (heads * seq, dim) if layout == head_dim_minor else (heads * dim, seq)

  out_shapes = {}
  if cp_size > 1:
    out_shapes["q_recv"] = jax.ShapeDtypeStruct(
        flat_shape(q_layout, heads_local, seq_len, head_dim_qk), q.dtype)
    out_shapes["k_recv"] = jax.ShapeDtypeStruct(
        flat_shape(k_layout, heads_local, seq_len, head_dim_qk), k.dtype)
    out_shapes["v_recv"] = jax.ShapeDtypeStruct(
        flat_shape(v_layout, heads_local, seq_len, head_dim_v), v.dtype)
  # o of the local heads over the full sequence; with cp == 1 it is the output.
  out_shapes["o_head"] = jax.ShapeDtypeStruct(
      (heads_local * seq_len, head_dim_v), q.dtype)
  if cp_size > 1:
    out_shapes["out"] = jax.ShapeDtypeStruct(
        (num_q_heads * local_seq_len, head_dim_v), q.dtype)
  stat_shape = (
      (heads_local * NUM_SUBLANES, seq_len) if compact
      else (heads_local * seq_len, NUM_LANES)
  )
  stat_names = []
  if save_residuals:
    stat_names.append("logsumexp" if fuse_reciprocal else "l_linear")
    if save_max_logits:
      stat_names.append("max_logits")
  for name in stat_names:
    out_shapes[name] = jax.ShapeDtypeStruct(stat_shape, jnp.float32)

  stat_tile = (NUM_SUBLANES, bq) if compact else (bq, NUM_LANES)
  in_tiles = {
      "q": (from_head_minor((bq, head_dim_qk), q_layout), q.dtype),
      "k": (from_head_minor((bkv, head_dim_qk), k_layout), k.dtype),
      "v": (from_head_minor((bkv, head_dim_v), v_layout), v.dtype),
  }
  if segment_ids is not None:
    in_tiles["q_ids"] = ((bq, NUM_LANES), hbm_in["q_ids"].dtype)
    in_tiles["kv_ids"] = ((NUM_SUBLANES, bkv), hbm_in["kv_ids"].dtype)
  if "q_seq" in hbm_in:
    in_tiles["q_seq"] = ((bq, NUM_LANES), hbm_in["q_seq"].dtype)
  out_tiles = {"o": ((bq, head_dim_v), q.dtype)}
  for name in stat_names:
    out_tiles[name] = (stat_tile, jnp.float32)

  scratch = {
      "m_scratch": pltpu.VMEM((bq, NUM_LANES), jnp.float32),
      "l_scratch": pltpu.VMEM((bq, NUM_LANES), jnp.float32),
      "o_scratch": pltpu.VMEM((bq, head_dim_v), jnp.float32),
  }
  for name, (shape, dtype) in in_tiles.items():
    scratch[f"{name}_tile"] = pltpu.VMEM((2, *shape), dtype)
  for name, (shape, dtype) in out_tiles.items():
    scratch[f"{name}_out_tile"] = pltpu.VMEM((2, *shape), dtype)

  body_kwargs = dict(
      mask_value=mask_value,
      kv_steps=kv_steps,
      bq=bq,
      bkv=bkv,
      bkv_compute=bkv_compute,
      head_dim_v=head_dim_v,
      mask_function=mask_function,
      fuse_reciprocal=fuse_reciprocal,
      config=config,
      native_layout=False,
  )
  names = [*prefetch, *hbm_in, *smem_in, *out_shapes, *scratch]

  def kernel(*refs):
    r = dict(zip(names, refs, strict=True))
    get = r.get
    rank_ = r["rank"][0]
    if cp_size > 1:
      q_buf, k_buf, v_buf = r["q_recv"], r["k_recv"], r["v_recv"]
    else:
      q_buf, k_buf, v_buf = r["q"], r["k"], r["v"]
    o_buf = r["o_head"]

    def block_ij(g):
      if dynamic_grid:
        return to_i32(r["active_rows"][g]), to_i32(r["active_cols"][g])
      return g // kv_steps, g % kv_steps

    def should_write(g):
      if dynamic_grid:
        return r["bounds_end"][g].astype(jnp.bool_)
      return block_ij(g)[1] == kv_steps - 1

    def in_copies(h, g, slot, in_sem):
      i, j = block_ij(g)
      src = {
          "q": lambda: _seq_tile(q_buf, q_layout, h, head_dim_qk, seq_len,
                                 i * bq, bq),
          "k": lambda: _seq_tile(k_buf, k_layout, h, head_dim_qk, seq_len,
                                 j * bkv, bkv),
          "v": lambda: _seq_tile(v_buf, v_layout, h, head_dim_v, seq_len,
                                 j * bkv, bkv),
          "q_ids": lambda: r["q_ids"].at[pl.ds(i * bq, bq), :],
          "kv_ids": lambda: r["kv_ids"].at[:, pl.ds(j * bkv, bkv)],
          "q_seq": lambda: r["q_seq"].at[pl.ds(i * bq, bq), :],
      }
      srcs = {name: src[name]() for name in in_tiles}
      return [
          pltpu.make_async_copy(
              srcs[name], r[f"{name}_tile"].at[slot], in_sem.at[slot, idx])
          for idx, name in enumerate(in_tiles)
      ]

    def out_copies(h, i, slot, out_sem):
      dsts = {"o": _seq_tile(o_buf, head_dim_minor, h, head_dim_v, seq_len,
                             i * bq, bq)}
      for name in stat_names:
        dsts[name] = _stat_tile(r[name], compact, h, seq_len, i * bq, bq)
      return [
          pltpu.make_async_copy(
              r[f"{name}_out_tile"].at[slot], dsts[name], out_sem.at[slot, idx])
          for idx, name in enumerate(out_tiles)
      ]

    def tile(name, slot):
      ref = get(f"{name}_tile")
      return None if ref is None else ref.at[slot]

    def out_tile(name, slot):
      ref = get(f"{name}_out_tile")
      return None if ref is None else ref.at[slot]

    def compute_head(h, in_sem, out_sem):
      """All active blocks of local head ``h`` with double-buffered DMA."""
      head_idx = rank_ * heads_local + h
      for copy in in_copies(h, 0, 0, in_sem):
        copy.start()

      def block(g, out_count):
        slot = g % 2

        # Prefetch block g+1 into the other slot; block g-1 already used it.
        @pl.when(g + 1 < num_blocks)
        def _():
          for copy in in_copies(h, g + 1, 1 - slot, in_sem):
            copy.start()

        for copy in in_copies(h, g, slot, in_sem):
          copy.wait()

        write = should_write(g)
        oslot = out_count % 2

        # Drain the writeback that used this output slot two rows ago.
        @pl.when(jnp.logical_and(write, out_count >= 2))
        def _():
          for copy in out_copies(h, 0, oslot, out_sem):
            copy.wait()

        flash_attention_kernel(
            get("active_rows"), get("active_cols"), None,
            get("bounds_start"), get("bounds_end"), get("block_mask"),
            tile("q", slot), tile("k", slot), tile("v", slot),
            tile("q_ids", slot), tile("kv_ids", slot), get("sinks"), None,
            tile("q_seq", slot), get("max_logit"),
            out_tile("o", oslot), out_tile("logsumexp", oslot),
            out_tile("l_linear", oslot), out_tile("max_logits", oslot),
            r["m_scratch"], r["l_scratch"], r["o_scratch"],
            grid_idx=g, head_idx=head_idx, num_heads=num_q_heads,
            **body_kwargs,
        )

        # Start the finished row's writeback; the next row computes meanwhile.
        @pl.when(write)
        def _():
          i, _ = block_ij(g)
          for copy in out_copies(h, i, oslot, out_sem):
            copy.start()

        return out_count + write.astype(jnp.int32)

      out_count = lax.fori_loop(0, num_blocks, block, jnp.int32(0))

      # Drain the last two writebacks so o is complete before its exchange.
      @pl.when(out_count >= 1)
      def _():
        for copy in out_copies(h, 0, (out_count - 1) % 2, out_sem):
          copy.wait()

      @pl.when(out_count >= 2)
      def _():
        for copy in out_copies(h, 0, out_count % 2, out_sem):
          copy.wait()

    if cp_size == 1:
      def scope(in_sem, out_sem):
        def head(h, carry):
          compute_head(h, in_sem, out_sem)
          return carry
        lax.fori_loop(0, heads_local, head, None)

      pl.run_scoped(
          scope,
          pltpu.SemaphoreType.DMA((2, len(in_tiles))),
          pltpu.SemaphoreType.DMA((2, len(out_tiles))),
      )
      return

    # Every peer must have entered the kernel before remote writes start.
    barrier = pltpu.get_barrier_semaphore()
    for step in range(1, cp_size):
      pl.semaphore_signal(
          barrier, inc=1, device_id={cp_axis: (rank_ + step) % cp_size},
          device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, cp_size - 1)

    streams = (
        (r["q"], q_buf, q_layout, head_dim_qk),
        (r["k"], k_buf, k_layout, head_dim_qk),
        (r["v"], v_buf, v_layout, head_dim_v),
    )

    def remote(src, dst, send_sem, recv_sem, device):
      return pltpu.make_async_remote_copy(
          src, dst, send_sem, recv_sem, device_id={cp_axis: device},
          device_id_type=pl.DeviceIdType.MESH)

    def in_exchange(h, in_send, in_recv, local_sem):
      """q/k/v sequence-to-head exchange of local head ``h``."""
      locals_, sends, recvs = [], [], []
      for s, (src, dst, layout, dim) in enumerate(streams):
        mine = _seq_tile(dst, layout, h, dim, seq_len, rank_ * local_seq_len,
                         local_seq_len)
        locals_.append(pltpu.make_async_copy(
            _seq_tile(src, layout, rank_ * heads_local + h, dim, local_seq_len,
                      0, local_seq_len),
            mine, local_sem.at[h, s]))
        for step in range(1, cp_size):
          peer_out = (rank_ + step) % cp_size
          peer_in = (rank_ + cp_size - step) % cp_size
          sends.append(remote(
              _seq_tile(src, layout, peer_out * heads_local + h, dim,
                        local_seq_len, 0, local_seq_len),
              mine, in_send.at[h, step, s], in_recv.at[h, step, s], peer_out))
          # Shape-only source; waits for the shard arriving from peer_in.
          recvs.append(remote(
              _seq_tile(src, layout, 0, dim, local_seq_len, 0, local_seq_len),
              _seq_tile(dst, layout, h, dim, seq_len, peer_in * local_seq_len,
                        local_seq_len),
              in_send.at[h, step, s], in_recv.at[h, step, s], peer_out))
      return locals_, sends, recvs

    def o_exchange(h, o_send, o_recv, local_sem):
      """o head-to-sequence exchange of local head ``h``."""
      out = r["out"]
      mine = _seq_tile(out, head_dim_minor, rank_ * heads_local + h, head_dim_v,
                       local_seq_len, 0, local_seq_len)
      local = pltpu.make_async_copy(
          _seq_tile(o_buf, head_dim_minor, h, head_dim_v, seq_len,
                    rank_ * local_seq_len, local_seq_len),
          mine, local_sem.at[h, 3])
      sends, recvs = [], []
      for step in range(1, cp_size):
        peer_out = (rank_ + step) % cp_size
        peer_in = (rank_ + cp_size - step) % cp_size
        sends.append(remote(
            _seq_tile(o_buf, head_dim_minor, h, head_dim_v, seq_len,
                      peer_out * local_seq_len, local_seq_len),
            mine, o_send.at[h, step], o_recv.at[h, step], peer_out))
        recvs.append(remote(
            _seq_tile(o_buf, head_dim_minor, h, head_dim_v, seq_len, 0,
                      local_seq_len),
            _seq_tile(out, head_dim_minor, peer_in * heads_local + h,
                      head_dim_v, local_seq_len, 0, local_seq_len),
            o_send.at[h, step], o_recv.at[h, step], peer_out))
      return [local], sends, recvs

    def start(descs):
      locals_, sends, _ = descs
      for d in (*locals_, *sends):
        d.start()

    def wait(descs):
      locals_, sends, recvs = descs
      for d in locals_:
        d.wait()
      for d in sends:
        d.wait_send()
      for d in recvs:
        d.wait_recv()

    def scope(in_send, in_recv, o_send, o_recv, local_sem, in_sem, out_sem):
      ins = lambda h: in_exchange(h, in_send, in_recv, local_sem)
      outs = lambda h: o_exchange(h, o_send, o_recv, local_sem)
      start(ins(0))
      # incoming(h+1) overlaps compute(h); outgoing(h) overlaps compute(h+1).
      # Heads are unrolled statically (heads / cp is small), matching the
      # measured schedule.
      for h in range(heads_local):
        wait(ins(h))
        if h + 1 < heads_local:
          start(ins(h + 1))
        compute_head(h, in_sem, out_sem)
        if h > 0:
          wait(outs(h - 1))
        start(outs(h))
      wait(outs(heads_local - 1))

    pl.run_scoped(
        scope,
        pltpu.SemaphoreType.DMA((heads_local, cp_size, 3)),
        pltpu.SemaphoreType.DMA((heads_local, cp_size, 3)),
        pltpu.SemaphoreType.DMA((heads_local, cp_size)),
        pltpu.SemaphoreType.DMA((heads_local, cp_size)),
        pltpu.SemaphoreType.DMA((heads_local, 4)),
        pltpu.SemaphoreType.DMA((2, len(in_tiles))),
        pltpu.SemaphoreType.DMA((2, len(out_tiles))),
    )

  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if segment_ids is not None else ""
  residuals = "_residuals" if save_residuals else "_no_residuals"
  kernel_name = f"splash_{attention_type}_fwd_a2a_cp{segments}{residuals}"
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            grid=(),
            in_specs=[
                *(_whole_spec(x.shape, pltpu.HBM) for x in hbm_in.values()),
                *(_whole_spec(x.shape, pltpu.SMEM) for x in smem_in.values()),
            ],
            out_specs=[_whole_spec(s.shape, pltpu.HBM)
                       for s in out_shapes.values()],
            scratch_shapes=list(scratch.values()),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(),
            collective_id=0 if cp_size > 1 else None,
            vmem_limit_bytes=(
                config.fwd_vmem_limit_bytes or _DEFAULT_FWD_VMEM_LIMIT_BYTES
            ),
            flags={
                "XLA_TPU_FORCE_LP_LLO_SCHEDULER": (
                    config.use_experimental_scheduler
                )
            },
        ),
        out_shape=list(out_shapes.values()),
        name=kernel_name,
        cost_estimate=config.fwd_cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*prefetch.values(), *hbm_in.values(), *smem_in.values())
  outs = dict(zip(out_shapes, all_out, strict=True))

  o_head = outs["o_head"].reshape(heads_local, seq_len, head_dim_v)
  out = (
      outs["out"].reshape(num_q_heads, local_seq_len, head_dim_v)
      if cp_size > 1 else o_head
  )

  # Same as the native kernel: with a dynamic grid and no active block the
  # outputs were never written, so report the empty-attention defaults.
  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x
    return jnp.where(mask_info.num_active_blocks[0] == 0, value, x)

  out = init_if_empty(out, 0.0)

  def public_stat(x):
    if compact:
      return x.reshape(heads_local, NUM_SUBLANES, seq_len)[:, 0, :]
    return x.reshape(heads_local, seq_len, NUM_LANES)[..., 0]

  logsumexp = max_logits = None
  if save_residuals:
    if save_max_logits:
      max_logits = init_if_empty(public_stat(outs["max_logits"]), mask_value)
    if fuse_reciprocal:
      logsumexp = init_if_empty(public_stat(outs["logsumexp"]), mask_value)
    else:
      log = jnp.log2 if config.use_base2_exp else jnp.log
      l = public_stat(outs["l_linear"])
      logsumexp = max_logits + log(l)
      denominator = jnp.where(l == 0, 1.0, l)
      out = (out / denominator[..., None]).astype(out.dtype)
  else:
    # If we're not saving residuals, then we can't fuse the reciprocal
    # out of the kernel.
    assert fuse_reciprocal

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(
        out, name=config.residual_checkpoint_name
    )
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(
          logsumexp, name=config.residual_checkpoint_name
      )
  if cp_size > 1:
    buffers = {"q": outs["q_recv"], "k": outs["k_recv"], "v": outs["v_recv"],
               "o": o_head}
  else:
    buffers = {"q": q, "k": k, "v": v, "o": out}
  stats = None
  if save_residuals:
    stats = {"logsumexp": logsumexp, "max_logits": max_logits}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
  return out, stats, buffers


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "save_residuals",
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "cp_size",
        "cp_axis",
    ),
)
def _splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    cp_size: int,
    cp_axis: str | None,
    max_logit_value: jax.Array | None = None,
) -> base.SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dkv_mask_info

  out, stats, _ = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      cp_size=cp_size,
      cp_axis=cp_axis,
      max_logit_value=max_logit_value,
  )
  if save_residuals:
    if config.use_base2_exp:  # for user, output values in natural base
      stats["logsumexp"] = stats["logsumexp"] / LOG2E
      stats["max_logits"] = stats["max_logits"] / LOG2E
    return out, stats
  else:
    return out


def _splash_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    cp_size: int,
    cp_axis: str | None,
    max_logit_value: jax.Array | None = None,
) -> tuple[tuple[jax.Array], base.SplashResidualsType]:

  # TODO: add some higher order AD check that isn't save_residuals based.
  # if save_residuals:
  #   raise NotImplementedError("Higher-order AD not supported.")

  out, stats, buffers = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      save_max_logits=save_residuals or not config.omit_unused_max_logits,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      cp_size=cp_size,
      cp_axis=cp_axis,
      max_logit_value=max_logit_value,
  )
  logsumexp = stats["logsumexp"]  # save in the config base for the bwd pass
  if config.use_base2_exp:  # for user, output values in natural base
    stats["logsumexp"] = stats["logsumexp"] / LOG2E
    if stats["max_logits"] is not None:
      stats["max_logits"] = stats["max_logits"] / LOG2E
  # The backward runs on the head-sharded buffers the forward already holds,
  # i.e. exactly the native kernel's inputs after the pre-kernel all-to-all.
  heads_local = q.shape[0] // cp_size
  if cp_size > 1:
    q_res = _unflat(buffers["q"], config.q_layout, heads_local)
    k_res = _unflat(buffers["k"], config.k_layout, heads_local)
    v_res = _unflat(buffers["v"], config.v_layout, heads_local)
    rank = lax.axis_index(cp_axis)
    sinks_res = (
        None if sinks is None
        else lax.dynamic_slice_in_dim(sinks, rank * heads_local, heads_local)
    )
  else:
    q_res, k_res, v_res, sinks_res = q, k, v, sinks
  residuals = (
      q_res, k_res, v_res, segment_ids, sinks_res, buffers["o"], logsumexp,
      dkv_mask_info,
  )
  if save_residuals:
    return (out, stats), residuals
  else:
    return out, residuals


def _flash_attention_dq_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  grid_idx = pl.program_id(1)
  if block_mask_ref is not None:
    kv_index = active_cols_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index = grid_idx % kv_steps
    should_not_mask = False
    should_initialize = kv_index == 0
    should_write = kv_index == kv_steps - 1

  @pl.when(should_initialize)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def body(has_partial_mask: bool = False):
    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
    if config.use_base2_exp and config.softmax_scale is None:
      q *= LOG2E
    # We keep k and v possibly transposed, since they are RHS of dots.
    k = k_ref[...]
    v = v_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    do = do_ref[...]
    di = jnp.expand_dims(di_ref[0], -1)

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)
    if config.softmax_scale is not None:
      qk_uncapped *= jnp.float32(config.softmax_scale)
      if config.use_base2_exp:
        qk_uncapped *= jnp.float32(LOG2E)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)
    dp_dims = (
        NT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    dp = lax.dot_general(
        do.astype(v.dtype),
        v,
        dp_dims,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    if config.softmax_scale is not None:
      ds *= jnp.float32(config.softmax_scale)

    dq_dims = (
        NN_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dq_scratch_ref[...] += lax.dot_general(
        ds.astype(k.dtype),
        k,
        dq_dims,
        preferred_element_type=jnp.float32,
    )

  @pl.when(should_not_mask)
  def _():
    body()

  @pl.when(jnp.logical_not(should_not_mask))
  def _():
    body(has_partial_mask=True)

  @pl.when(should_write)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # aliases
    dq_alias,
    dk_alias,
    dv_alias,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    # Scratch
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    mask_value: float,
    q_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    q_heads_per_kv_head: int,
    config: SplashConfig,
    native_layout: bool = False,
):
  del mask_next_ref, active_cols_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    q_head = pl.program_id(0)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index, q_head, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    grid_idx = (kv_index * q_steps) + q_index
    should_initialize = q_index == 0
    should_write = True if q_steps <= 2 else q_index == q_steps - 1
    if q_heads_per_kv_head > 1:
      q_head_index_per_kv_head = lax.rem(q_head, q_heads_per_kv_head)
      should_initialize = jnp.logical_and(
          should_initialize, q_head_index_per_kv_head == 0
      )
      should_write = jnp.logical_and(
          should_write, q_head_index_per_kv_head == q_heads_per_kv_head - 1
      )

  if block_mask_ref is not None:
    head_axis = 0 if active_rows_ref is not None else 1
    kind = _block_mask_value(
        block_mask_ref, grid_idx, q_head, pl.num_programs(head_axis)
    )
    should_not_mask = kind != 1
    should_run = kind != 0
  else:
    should_not_mask = False
    should_run = True

  # TODO: Update docstring explaining the accumulation logic

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  resident_logsumexp = logsumexp_ref[:1, :] if native_layout else None
  resident_di = di_ref[:1, :] if native_layout else None

  def body(i, _, has_partial_mask=False):
    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]
    if config.use_base2_exp and config.softmax_scale is None:
      scaled_q = q * LOG2E
    else:
      scaled_q = q

    def _load_kv(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return ref[slice_k, :]
      return ref[:, slice_k].T

    k = _load_kv(k_ref, config.k_layout)
    v = _load_kv(v_ref, config.v_layout)
    logsumexp = resident_logsumexp if native_layout else logsumexp_ref[:1, :]
    do = do_ref[...]
    di = resident_di if native_layout else di_ref[:1, :]
    if native_layout:
      dp_dims = NN_DIM_NUMBERS
    else:
      dp_dims = NT_DIM_NUMBERS

    def compute_dp():
      return lax.dot_general(
          v,
          do,
          dp_dims,
          preferred_element_type=jnp.float32,
      )

    qk_dims = (
        NT_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    qk_uncapped = lax.dot_general(
        k, scaled_q, qk_dims, preferred_element_type=jnp.float32
    )
    if config.softmax_scale is not None:
      if config.use_base2_exp and config.combine_log2_scale:
        qk_uncapped *= jnp.float32(config.softmax_scale * LOG2E)
      else:
        qk_uncapped *= jnp.float32(config.softmax_scale)
        if config.use_base2_exp:
          qk_uncapped *= jnp.float32(LOG2E)
    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref
        if (not config.segment_mask_on_partial_only or has_partial_mask)
        else None,
        kv_segment_ids_ref
        if (not config.segment_mask_on_partial_only or has_partial_mask)
        else None,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
        kv_segment_ids_seq_minor=native_layout,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    # A backward tile can mix empty and nonempty forward rows. Empty rows
    # have logsumexp=-inf and must contribute zero probability and gradients.
    p = exp(jnp.where(logsumexp == -jnp.inf, -jnp.inf, qk - logsumexp))
    p_bf16 = p.astype(do.dtype)

    dv = lax.dot_general(
        p_bf16, do,
        NT_DIM_NUMBERS if native_layout else NN_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    scratch_ref = dv_scratch_ref
    if native_layout:
      dv = dv.astype(dv_scratch_ref.dtype) + scratch_ref[:, slice_k].T
      scratch_ref[:, slice_k] = dv.T
    else:
      dv = dv.astype(dv_scratch_ref.dtype) + scratch_ref[slice_k, :]
      scratch_ref[slice_k, :] = dv

    dp = compute_dp()
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    if config.softmax_scale is not None and not config.bwd_scale_after_dot:
      ds *= jnp.float32(config.softmax_scale)

    def compute_dk():
      dk_dims = (
          NN_DIM_NUMBERS
          if config.q_layout == HEAD_DIM_MINOR
          else NT_DIM_NUMBERS
      )
      dk = lax.dot_general(
          ds.astype(do.dtype),
          q,
          dk_dims,
          preferred_element_type=jnp.float32,
      )
      if config.softmax_scale is not None and config.bwd_scale_after_dot:
        dk *= jnp.float32(config.softmax_scale)
      scratch_ref = dk_scratch_ref
      if native_layout:
        dk = dk.astype(dk_scratch_ref.dtype) + scratch_ref[:, slice_k].T
        scratch_ref[:, slice_k] = dk.T
      else:
        dk = dk.astype(dk_scratch_ref.dtype) + scratch_ref[slice_k, :]
        scratch_ref[slice_k, :] = dk

    if not config.bwd_dq_first:
      compute_dk()
    if dq_scratch_ref is not None or dq_ref is not None:
      if native_layout:
        dq_dims = TN_DIM_NUMBERS
        dq_transposed = lax.dot_general(
            k, ds.astype(k.dtype), dq_dims,
            preferred_element_type=jnp.float32,
        )
        # Sequence-minor scratch cancels this logical-output transpose.
        dq = dq_transposed.T
      else:
        dq = lax.dot_general(
            ds.astype(k.dtype).T
            if config.bwd_cast_before_transpose
            else ds.T.astype(k.dtype),
            k,
            NN_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
      if config.softmax_scale is not None and config.bwd_scale_after_dot:
        dq *= jnp.float32(config.softmax_scale)
      if dq_scratch_ref is not None:
        # Compute block size != memory block size
        scratch_update = dq.T if native_layout else dq
        dq_scratch_ref[...] += scratch_update
      else:
        if dq_alias is not None:
          dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
        else:
          dq_ref[...] = dq.astype(dq_ref.dtype)

    if config.bwd_dq_first:
      compute_dk()

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_alias is not None:
    dq_ref[...] = dq_alias[...]
  else:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  k_seq_axis = 0 if config.k_layout is HEAD_DIM_MINOR else 1
  num_iters = k_ref.shape[k_seq_axis] // bkv_compute

  def run_inner_loop(has_partial_mask):
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=has_partial_mask), None,
        unroll=config.bwd_kv_unroll,
    )

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    run_inner_loop(False)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _():
    run_inner_loop(True)

  if dq_scratch_ref is not None:
    if dq_alias is not None:
      dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)
    else:
      dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  if dk_alias is None:
    assert dv_alias is None

    @pl.when(should_write)
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  else:
    q_head = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _():
      dk_ref[...] = dk_alias[...] + dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_alias[...] + dv_scratch_ref[...].astype(dv_ref.dtype)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
    dkv_mask_sparsity: float,
):
  native_layout = (
      _use_native_layout(
          config, q, k, v, segment_ids, mask_info, mask_function, is_mqa=is_mqa
      )
      and bkv > bkv_compute
      and do.dtype == q.dtype
  )
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  dynamic_grid = mask_info.active_rows is not None

  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if dynamic_grid:

    def unravel(f):
      def index_map(h, grid_idx, rows_ref, cols_ref, *_):
        j = to_i32(rows_ref[grid_idx])
        i = to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    grid_size = mask_info.num_active_blocks[0]
    grid = (num_q_heads, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads, q_steps)

    def mask_index_map(j, h, i, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      grid_idx = j * q_steps + i
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  q_index_map = unravel(
      lambda h, i, j: from_head_minor((h, i, 0), config.q_layout)
  )
  o_index_map = unravel(lambda h, i, j: (h, i, 0))

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map
  )

  o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    layout = (
        QKVLayout.SEQ_MINOR
        if native_layout
        else QKVLayout.HEAD_DIM_MINOR
    )
    return from_head_minor((*prefix, j, 0), layout)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk)
          if is_mqa
          else (None, bkv, head_dim_qk),
          QKVLayout.SEQ_MINOR
          if native_layout
          else QKVLayout.HEAD_DIM_MINOR,
      ),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v)
          if is_mqa
          else (None, bkv, head_dim_v),
          QKVLayout.SEQ_MINOR
          if native_layout
          else QKVLayout.HEAD_DIM_MINOR,
      ),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None and segment_ids.q.ndim == 2:
    q_segment_spec, q_segment_ids = _group_segment_spec(
        segment_ids.q, num_q_heads, bq, True,
        1 if native_layout else NUM_SUBLANES, 0, unravel,
    )
    kv_segment_spec, kv_segment_ids = _group_segment_spec(
        segment_ids.kv, num_q_heads, bkv, native_layout,
        1 if native_layout else NUM_LANES, 1, unravel,
    )
  elif segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))
    if native_layout:
      # MaskInfo already fixes the DMA tiles; only physical buffer layouts
      # change here, not the visited blocks or segment equality semantics.
      q_segment_spec = pl.BlockSpec((1, bq), q_segment_ids_index_map)
      kv_segment_spec = pl.BlockSpec(
          (1, bkv), unravel(lambda h, i, j: (0, j))
      )
      q_segment_ids = segment_ids.q[None, :]
      kv_segment_ids = segment_ids.kv[None, :]
    else:
      q_segment_spec = pl.BlockSpec(
          (NUM_SUBLANES, bq), q_segment_ids_index_map
      )
      kv_segment_spec = pl.BlockSpec(
          (bkv, NUM_LANES), kv_segment_ids_index_map
      )
      q_segment_ids = jax.lax.broadcast_in_dim(
          segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
      )
      kv_segment_ids = jax.lax.broadcast_in_dim(
          segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
      )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  if native_layout:
    do_spec = pl.BlockSpec(
        (None, head_dim_v, bq), unravel(lambda h, i, j: (h, 0, i))
    )
  else:
    do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_sequence = None
    in_specs.append(None)

  dq_reduction_steps = config.dq_reduction_steps
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq = dq_alias_spec = None
  dq_layout = (
      QKVLayout.SEQ_MINOR if native_layout
      else QKVLayout.HEAD_DIM_MINOR
  )
  dq_block_shape = (None, None, *from_head_minor((bq, head_dim_qk), dq_layout))
  dq_output_shape = from_head_minor(q.shape, dq_layout)
  if dq_reduction_steps == 3:
    dq_index_map = unravel(
        lambda h, i, j: from_head_minor((j % 3, h, i, 0), dq_layout)
    )
    dq_spec = pl.BlockSpec(dq_block_shape, dq_index_map)
    dq_alias_spec = dq_spec
    dq_shape = jax.ShapeDtypeStruct((3, *dq_output_shape), q.dtype)
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(
        lambda h, i, j: from_head_minor((j, h, i, 0), dq_layout)
    )
    dq_spec = pl.BlockSpec(dq_block_shape, dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    q_dtype = q.dtype if kv_steps <= 4 else jnp.float32
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *dq_output_shape), q_dtype)

  in_specs += [dq_alias_spec]

  if bkv == bkv_compute:
    dq_scratch = None
  else:
    dq_scratch = pltpu.VMEM(
        (
            (head_dim_qk, bq)
            if native_layout
            else (bq, head_dim_qk)
        ),
        jnp.float32,
    )

  if dynamic_grid and q_heads_per_kv_head != 1:
    # in/out aliasing to accumulate within kv groups.
    in_specs += [dk_spec, dv_spec]
    dk = lax.empty(k.shape, dtype=jnp.float32)
    dv = lax.empty(v.shape, dtype=jnp.float32)
    # Keep gradients in fp32 when accumulating over head groups.
    dk_type = dv_type = jnp.float32
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = k.dtype
    dv_type = v.dtype

  dk_shape = k.mT.shape if native_layout else k.shape
  dv_shape = v.mT.shape if native_layout else v.shape
  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(dk_shape, dk_type),
      jax.ShapeDtypeStruct(dv_shape, dv_type),
  ]
  out_specs = [dq_spec, dk_spec, dv_spec]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      q_heads_per_kv_head=q_heads_per_kv_head,
      native_layout=native_layout,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
              use_experimental_scheduler=config.use_experimental_scheduler,
          ),
      )
  }
  args = [
      # scalar prefetch
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      # inputs
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do.mT if native_layout else do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  input_fusion_candidates = [
      False,  # active_rows
      False,  # active_cols
      False,  # mask_next
      False,  # bounds_start
      False,  # bounds_end
      False,  # block_mask
      False,  # q
      False,  # k
      False,  # v
      # Group-indexed ID loads cannot always be dematerialized by input fusion.
      segment_ids is None or segment_ids.q.ndim == 1,  # q_segment_ids
      segment_ids is None or segment_ids.kv.ndim == 1,  # kv_segment_ids
      False,  # logsumexp
      False,  # do
      False,  # di
      False,  # partial_mask_blocks
      False,  # q_sequence
  ]
  num_args = sum(1 for x in args if x is not None)
  input_output_aliases = {}
  if dq_reduction_steps == 3:
    if dynamic_grid and q_heads_per_kv_head != 1:
      input_output_aliases = {num_args: 0, num_args + 1: 1, num_args + 2: 2}
    else:
      input_output_aliases = {num_args: 0}
  elif dynamic_grid and q_heads_per_kv_head != 1:
    input_output_aliases = {num_args: 1, num_args + 1: 2}

  scratch_shapes = [
      dq_scratch,
      pltpu.VMEM(
          (
              (head_dim_qk, bkv)
              if native_layout
              else (bkv, head_dim_qk)
          ),
          jnp.float32,
      ),
      pltpu.VMEM(
          (
              (head_dim_v, bkv)
              if native_layout
              else (bkv, head_dim_v)
          ),
          jnp.float32,
      ),
  ]

  def _bwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      logsumexp: jax.Array,
      do: jax.Array,
      di: jax.Array,
      partial_mask_blocks: jax.Array | None,
      q_sequence: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity_factor: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    total_matmul_flops_per_head = (
        2 * q_seq_len * kv_seq_len * head_dim_qk  # qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dv
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dp
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dq
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dk
    )

    estimated_flops = int(
        total_matmul_flops_per_head * num_q_heads * mask_sparsity_factor
    )

    exp_flops = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    if config.attn_logits_soft_cap is None:
      tanh_flops = 0
    else:
      tanh_flops = (
          2 * num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
      )
    estimated_transcendentals = int(exp_flops + tanh_flops)

    inputs_ = [
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        partial_mask_blocks,
        q_sequence,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))

    estimated_bytes = input_bytes + output_bytes

    return pl.CostEstimate(
        flops=estimated_flops,
        transcendentals=estimated_transcendentals,
        bytes_accessed=estimated_bytes,
    )

  cost_estimate = config.bwd_cost_estimate or _bwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  allow_input_fusion = None
  if native_layout:
    # The dynamic grid bound is the first custom-call operand. Pallas then
    # removes None arguments while preserving the order of the remaining
    # scalar-prefetch and regular inputs, followed by aliased outputs.
    allow_input_fusion = (
        *((False,) if dynamic_grid else ()),
        *(
            can_fuse
            for arg, can_fuse in zip(args, input_fusion_candidates)
            if arg is not None
        ),
        *(False for value in (dq, dk, dv) if value is not None),
    )

  with jax.named_scope(kernel_name):
    dq_unreduced, dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        # We set all dimensions to arbitrary because:
        # 1) for heads, we are reducing over heads
        # 2) for kv_seq_len, the splash attention prefetch schedule assumes no
        #     megacore
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * len(grid),
            flags={}
            if config.bwd_scheduler is None
            else {"XLA_TPU_FORCE_LP_LLO_SCHEDULER": config.bwd_scheduler},
            vmem_limit_bytes=config.bwd_vmem_limit_bytes,
            allow_input_fusion=allow_input_fusion,
        ),
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv)
  with jax.named_scope("splash_bwd_epilogue"):
    dq = dq_unreduced.sum(axis=0)
    dq = dq.astype(q.dtype)
    if native_layout:
      dq = dq.mT
      dk = dk.mT
      dv = dv.mT
    dk = dk.astype(k.dtype)
    dv = dv.astype(v.dtype)
  return dq, dk, dv


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    cp_size: int,
    cp_axis: str | None,
    res: base.SplashResidualsType,
    grads: jax.Array | tuple[jax.Array, dict[str, jax.Array]],
) -> tuple[
    MaskInfo | None,  # fwd_mask_info
    MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    base.SegmentIds | None,  # segment_ids
    jax.Array | None,  # segment_ids
    jax.Array | None,  # max_logit_estimate
]:
  # If `save_residuals` is True, `_splash_attention_fwd` returns `(out, stats)`,
  # so we unpack the gradients, otherwise it returns `out` and `grads` is just
  # `do`.
  if save_residuals:
    do, _ = grads
  else:
    do = grads
  del save_residuals, fwd_mask_sparsity
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
      config.block_q_dkv,
      config.block_kv_dkv,
      config.block_kv_dkv_compute,
  )
  q, k, v, segment_ids, sinks, o, logsumexp, dkv_mask_info = res
  if cp_size > 1:
    # dO arrives sequence sharded; move it to the head-sharded layout of o.
    do = lax.all_to_all(do, cp_axis, 0, 1, tiled=True)

  # di: [num_heads, q_seq_len]
  with jax.named_scope("splash_bwd_di"):
    di = jnp.einsum(
        "hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32)
    )  # pytype: disable=attribute-error
  with jax.named_scope("splash_bwd_fused_dq_dkv"):
    dq, dk, dv = _splash_attention_bwd_dkv(
        q,
        k,
        v,
        segment_ids,
        logsumexp,
        do,
        di,
        bq=bq_dkv,
        bkv=bkv_dkv_memory,
        bkv_compute=bkv_dkv_compute,
        is_mqa=is_mqa,
        mask_info=dkv_mask_info,
        mask_value=mask_value,
        mask_function=mask_function,
        config=config,
        dkv_mask_sparsity=dkv_mask_sparsity,
    )
  dsinks = None
  if sinks is not None:
    with jax.named_scope("splash_bwd_sinks"):
      logsumexp_ = (logsumexp / LOG2E) if config.use_base2_exp else logsumexp
      sinks_exp = -jnp.exp(
          sinks[..., None, None].astype(jnp.float32)
          - logsumexp_[..., None].astype(jnp.float32)
      )
      dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))
  if cp_size > 1:
    # Back to the caller's sequence-sharded layout.
    dq, dk, dv = (lax.all_to_all(x, cp_axis, 1, 0, tiled=True)
                  for x in (dq, dk, dv))
    if dsinks is not None:
      # Each rank holds the gradient of its own head slice only.
      heads_local = dsinks.shape[0]
      rank = lax.axis_index(cp_axis)
      dsinks = lax.dynamic_update_slice_in_dim(
          jnp.zeros((heads_local * cp_size,), dsinks.dtype), dsinks,
          rank * heads_local, 0)
  # Match the signature of the fwd function.
  assert dq is not None
  return (
      None,  # fwd_mask_info
      None,  # dvk_mak_info
      dq,  # q
      dk,  # k
      dv,  # v
      None,  # segment_ids
      dsinks,  # sinks
      None,  # max_logit_estimate
  )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "cp_size",
        "cp_axis",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: base.SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    cp_size: int,
    cp_axis: str | None,
) -> base.SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      config=config,
      max_logit_value=max_logit_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
      cp_size=cp_size,
      cp_axis=cp_axis,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionA2ACpKernel:
  """Splash with shared or grouped runtime segment IDs.

  IDs shaped [groups, sequence] share one ID array across each contiguous
  group of heads. The group count must divide both Q and KV head counts.
  A [groups, grid] block mask can describe their union grid: zero skips work
  for that group while preserving grid-boundary initialization and stores.
  Grouped IDs and masks require the fused backward kernel.
  """

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> base.SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if len(sharding.spec) != 1:
      raise ValueError("Only q sequence sharding is supported.")

    _resolve_spec = lambda x: sharding.spec if x is not None else None

    def mask_info_spec(mask_info):
      block_mask_spec = None
      if mask_info.block_mask is not None:
        block_mask_spec = sharding.spec
        if mask_info.block_mask.ndim == 2:
          # Grouped masks are [groups, grid]: replicate groups and partition
          # the grid just like the shared one-dimensional metadata arrays.
          block_mask_spec = jax.sharding.PartitionSpec(None, *sharding.spec)
        block_mask_sharding = jax.sharding.NamedSharding(
            sharding.mesh, block_mask_spec
        )
        try:
          block_mask_sharding.shard_shape(mask_info.block_mask.shape)
        except ValueError as exc:
          raise ValueError(
              "The sharding must divide the mask blocks evenly between devices"
          ) from exc

      return MaskInfo(  # pytype: disable=wrong-arg-types
          mask_next=_resolve_spec(mask_info.mask_next),
          active_rows=_resolve_spec(mask_info.active_rows),
          active_cols=_resolve_spec(mask_info.active_cols),
          num_active_blocks=_resolve_spec(mask_info.num_active_blocks),
          block_mask=block_mask_spec,
          partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated
          if mask_info.partial_mask_blocks is not None
          else None,
          q_sequence=_resolve_spec(mask_info.q_sequence),
      )

    return SplashAttentionA2ACpKernel(
        mask_info_spec(self.fwd_mask_info),
        mask_info_spec(self.dkv_mask_info)
        if self.dkv_mask_info is not None
        else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return ((self.fwd_mask_info, self.dkv_mask_info), self.kwargs)

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dkv_mask_info = (
        MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    )
    return SplashAttentionA2ACpKernel(
        MaskInfo(*fwd_mask_info), dkv_mask_info, **kwargs
    )


def _make_splash_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int = 1,
    cp_size: int = 1,
    cp_axis: str | None = None,
):
  # After the fused exchange every rank attends the full sequence, so the mask
  # is never sequence sharded.
  if q_seq_shards != 1:
    raise ValueError(f"{q_seq_shards=} must be 1; the a2a exchange is fused.")
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = mask_lib.NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )

    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)
  else:
    dkv_mask_sparsity = 1.0

  return SplashAttentionA2ACpKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
      cp_size=cp_size,
      cp_axis=cp_axis,
  )


def _make_dynamic_splash_attention(
    mask: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
    mask_spec: jax.sharding.PartitionSpec | None = None,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    cp_size: int = 1,
    cp_axis: str | None = None,
):
  if (mesh is not None) != (mask_spec is not None):
    raise ValueError(
        "Either both or neither of mesh and mask_spec must be specified."
    )

  if mask_spec is not None and len(mask_spec) != 1:
    raise ValueError("Only shard over the query sequence dimension.")

  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if config is None:
    config = SplashConfig.get_default()

  # This is the only mode that supports the dynamic grid.
  config = dataclasses.replace(config, dq_reduction_steps=3)

  def process_mask_shard(mask):
    process_mask_fn = functools.partial(
        mask_info_lib._process_dynamic_mask,
        downcast_smem_data=downcast_smem_data,
        partial_mask_blocks_dtype=partial_mask_blocks_dtype,
    )

    fwd_mask_info = process_mask_fn(
        mask, (config.block_q, config.block_kv), is_dkv=False
    )

    dkv_mask_info = None
    if config.has_backward_blocks:
      dkv_mask_info = process_mask_fn(
          mask, (config.block_q_dkv, config.block_kv_dkv), is_dkv=True
      )

    return fwd_mask_info, dkv_mask_info

  kwargs = dict(
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=None,
      fwd_mask_sparsity=1.0,
      dkv_mask_sparsity=1.0,
      cp_size=cp_size,
      cp_axis=cp_axis,
  )

  # If the input mask is replicated we don't need to call shard_map.
  if mask_spec is None:
    fwd_mask_info, dkv_mask_info = process_mask_shard(mask)
    kernel = SplashAttentionA2ACpKernel(fwd_mask_info, dkv_mask_info, **kwargs)
    return kernel

  mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
      mask_next=mask_spec,
      active_rows=None,
      active_cols=None,
      num_active_blocks=None,
      block_mask=mask_spec,
      partial_mask_blocks=mask_spec,
      q_sequence=None,
  )
  out_specs = (
      mask_info_specs,
      mask_info_specs if config.has_backward_blocks else None,
  )

  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=mask_spec,
      out_specs=out_specs,
      check_vma=False,
  )
  def process_all_shards(mask):
    return process_mask_shard(mask)

  fwd_mask_info, dkv_mask_info = process_all_shards(mask)
  kernel = SplashAttentionA2ACpKernel(fwd_mask_info, dkv_mask_info, **kwargs)
  kernel_spec = SplashAttentionA2ACpKernel(*out_specs, **kwargs)

  return (kernel, kernel_spec)


make_splash_mha_a2a_cp = partial(_make_splash_attention, is_mqa=False)
make_dynamic_splash_mha_a2a_cp = partial(
    _make_dynamic_splash_attention, is_mqa=False
)
