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

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from collections.abc import Callable
import contextlib
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


P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed
TN_DIM_NUMBERS = (((0,), (0,)), ((), ()))  # LHS transposed
TT_DIM_NUMBERS = (((0,), (1,)), ((), ()))  # Both operands transposed

LOG2E = math.log2(math.e)
LOG2E_INV = 1 / LOG2E

# mypy: ignore-errors


def _not(x: jax.Array | bool) -> jax.Array | bool:
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


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


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  # TODO: Remove these 3 params, they're only kept for backwards compatibility.
  block_q_dq: int | None = None
  block_kv_dq: int | None = None
  use_fused_bwd_kernel: bool = True

  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  fwd_cost_estimate: pl.CostEstimate | None = None
  bwd_cost_estimate: pl.CostEstimate | None = None

  residual_checkpoint_name: str | None = None  # whether to checkpoint outputs
  attn_logits_soft_cap: float | None = None
  fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  interpret: bool = False
  # The fused bwd kernel accumulates dq at every grid step. To safely avoid
  # read/write conflicts we conservatively avoid *any* in-kernel reductions.
  # This parameter allows to override this behavior and specifies the number of
  # reduction steps. For now, only 3 or all the kv steps are supported.
  dq_reduction_steps: int | None = None
  # Opt-in ViT tuning. Defaults preserve the existing execution paths.
  # Arithmetic reordering may change floating-point rounding.
  combine_log2_scale: bool = False
  # Integers use lax.fori_loop's partial-unroll factor (1 is rolled).
  bwd_kv_unroll: bool | int = True
  # Segment-only diagnostic: one masked loop body avoids duplicating the
  # expanded full/partial branches. Full tiles perform redundant exact masking.
  bwd_single_segment_mask_body: bool = False
  # ViT diagnostic: prepare the next KV block before consuming prior P/dS.
  # Carried P/dS are cast only where the reference gradient dots already cast.
  bwd_staged_kv_pipeline: bool = False
  # Scheduling probe: interleave next-tile producers with native-dQ/dK-first
  # consumers. Source ordering alone does not guarantee hardware overlap.
  bwd_staged_kv_interleave: bool = False
  # Remove producer conditionals by preparing an unused, valid tile 0 at the
  # tail. This adds one producer tile; only measured schedules may justify it.
  bwd_staged_kv_wrap_tail: bool = False
  # Split Q compute independently of the outer Q DMA tile to bound P/dS live
  # ranges. Optional producer/consumer carry holds only BF16 P and dS.
  bwd_block_q_compute: int | None = None
  bwd_qtile_pipeline: bool = False
  # Nested KV/Q compute loops permit reusing each dK/dV accumulator across
  # the inner Q sweep. Keep a scratch-updating nested control for ablation.
  bwd_qtile_nested: bool = False
  bwd_qtile_accumulator_carry: bool = False
  bwd_dq_first: bool = False
  bwd_dv_last: bool = False
  # Complete the gradient-consumer ordering screen without changing dots.
  bwd_dv_between_dq_dk: bool = False
  bwd_cast_before_transpose: bool = False
  bwd_dq_contract_ds_axis0: bool = False
  # Diagnostic: produce dQ.T directly, avoiding the large dS transpose.
  # Reassociation can change TPU rounding; never infer accuracy from CPU alone.
  bwd_dq_transposed_output: bool = False
  # Independent diagnostic controls for the other two gradient contractions.
  # Keep BF16 dot inputs and FP32 accumulation exactly as in the normal path.
  bwd_dk_transposed_output: bool = False
  bwd_dv_transposed_output: bool = False
  # ViT-only diagnostic: form P/dS as [Q, KV] throughout the inner loop.
  # dK/dV then write sequence-minor accumulators without transposing P/dS.
  bwd_qmajor_probabilities: bool = False
  bwd_keep_kv_seq_minor: bool = False
  # Feed dO directly in sequence-minor physical layout to dP/dV dots.
  # This changes operand preparation, not the logical contraction or dtype.
  bwd_do_seq_minor: bool = False
  bwd_dp_before_qk: bool = False
  bwd_reuse_bf16_probabilities: bool = False
  # Use singleton broadcast axes for backward ID inputs. The KV-major
  # orientation may still incur lane padding; see the native layout below.
  bwd_compact_segment_ids: bool = False
  # Put the KV token axis in lanes rather than padding each int32 token to
  # NUM_LANES columns. This changes only backward mask input layout.
  bwd_kv_segment_ids_seq_minor: bool = False
  # Store FP32 dK/dV accumulators with sequence as the minor dimension. This
  # avoids padding a non-MXU-facing head dimension in VMEM.
  bwd_dkv_scratch_seq_minor: bool = False
  # Store the FP32 dQ accumulator with sequence as the minor dimension. This
  # avoids the same head-dimension padding without changing the public layout.
  bwd_dq_scratch_seq_minor: bool = False
  # Keep the dQ partial output and its alias in sequence-minor layout too.
  # Preserve partial dtype/rounding and restore public layout after reduction.
  bwd_dq_output_seq_minor: bool = False
  # Return dK/dV from the Pallas call in sequence-minor physical layout, then
  # restore the public logical layout outside the custom call.
  bwd_dkv_output_seq_minor: bool = False
  # Let XLA fuse the segment-ID broadcast producers into the custom call.
  bwd_fuse_segment_id_inputs: bool = False
  # Process multiple independent MHA heads in one Pallas program. Dots remain
  # 2D (Mosaic TPU does not support a rank-3 batched dot); the larger program
  # gives the scheduler independent BF16 dot/vector work to interleave.
  bwd_head_group_size: int = 1
  bwd_scale_after_dot: bool = False
  omit_unused_max_logits: bool = False
  compact_stats_output: bool = False
  compact_softmax_scratch: bool = False
  # Keep inner-loop softmax/output state as SSA loop carry. Scratch is still
  # used across memory tiles, but not explicitly loaded/stored each compute tile.
  fwd_loop_carry: bool = False
  # Diagnostic orientations: keep the head-width axis out of the MXU columns.
  # PV still consumes the original FP32 probabilities.
  fwd_pv_transposed_output: bool = False
  fwd_output_scratch_seq_minor: bool = False
  # Normalize sequence-minor FP32 scratch before casting/transposing output.
  # Keep compact statistics in their native layout through the drain.
  fwd_native_output_normalization: bool = False
  # Return a sequence-minor physical Pallas output, then restore public layout.
  # Any outer conversion must be included in end-to-end kernel timing.
  fwd_output_seq_minor: bool = False
  # Fixed-shift ViT diagnostic: produce P as [KV, Q] from QK onward,
  # feed V @ P directly, and keep output/state scratch sequence-minor.
  fwd_kvmajor_probabilities: bool = False
  # Share dot/exp/PV between full/partial tiles; branch only around masking.
  # This controls instruction footprint without changing mask semantics.
  fwd_kvmajor_single_loop: bool = False
  # Branchless shared-loop control: check segment equality on every tile.
  # Forward-only flag so the backward mask policy is not changed.
  fwd_kvmajor_mask_all_tiles: bool = False
  # Accuracy/scheduling control: express sum on the reference's logical axis.
  # This does not guarantee identical lowering or floating-point association.
  fwd_kvmajor_sum_in_qmajor: bool = False
  # Diagnostic: append constant-one rows to V so PV also computes sum(P).
  # Requires HIGHEST contraction precision, not just an FP32 P array: the
  # default TPU matmul approximation is not accurate enough for the denominator.
  fwd_kvmajor_fuse_normalizer: bool = False
  fwd_kv_unroll: bool | int = True
  # Fixed-logit-shift ViT diagnostic: QK/exp for i before PV/accum for i-1.
  fwd_staged_kv_pipeline: bool = False
  bwd_parallel_heads: bool = False
  bwd_scheduler: bool | None = None
  # Caller contract: every full tile in MaskInfo must also be fully allowed
  # by the runtime segment IDs. Partial tiles retain exact segment checks.
  segment_mask_on_partial_only: bool = False
  fwd_vmem_limit_bytes: int | None = None
  bwd_vmem_limit_bytes: int | None = None
  # An experimental scheduler that sometimes produces better softmax overlap.
  use_experimental_scheduler: bool = False
  # If provided, scale FP32 QK logits inside the kernel. Keeping this optional
  # preserves the legacy path, where callers pre-scale Q before invoking Splash.
  softmax_scale: float | None = None
  # Diagnostic only: fine scopes can perturb the compiled pipeline when
  # custom-call region tracing is enabled. Coarse scopes leave inner-loop
  # MXU/vector scheduling unannotated. Keep all in-kernel scopes opt-in.
  region_trace_mode: str = "none"

  def __post_init__(self):
    if self.region_trace_mode not in ("none", "coarse", "fine"):
      raise ValueError(f"Invalid region_trace_mode: {self.region_trace_mode}")
    if self.bwd_dv_between_dq_dk and self.bwd_dv_last:
      raise ValueError("dV cannot be both between dQ/dK and last")
    if self.bwd_staged_kv_interleave and not (
        self.bwd_staged_kv_pipeline
        and self.bwd_dq_transposed_output and not self.bwd_dq_first
    ):
      raise ValueError("interleaved KV pipeline requires staged native-dQ/dK-first consumers")
    if self.bwd_staged_kv_wrap_tail and not self.bwd_staged_kv_pipeline:
      raise ValueError("wrapped KV tail requires the staged KV pipeline")
    if self.bwd_dq_output_seq_minor and not (
        self.bwd_dq_scratch_seq_minor and self.use_fused_bwd_kernel
    ):
      raise ValueError("sequence-minor dQ output requires native scratch and fused backward")
    if self.bwd_block_q_compute is not None and (
        self.bwd_block_q_compute <= 0
        or self.bwd_block_q_compute % NUM_LANES
        or self.block_q_dkv is None
        or self.block_q_dkv % self.bwd_block_q_compute
    ):
      raise ValueError("backward Q compute tile must be lane-aligned and divide block_q_dkv")
    if self.bwd_qtile_pipeline and self.bwd_block_q_compute is None:
      raise ValueError("Q-tile pipeline requires bwd_block_q_compute")
    if self.bwd_qtile_nested and self.bwd_block_q_compute is None:
      raise ValueError("nested Q-tile loops require bwd_block_q_compute")
    if self.bwd_qtile_accumulator_carry and not self.bwd_qtile_nested:
      raise ValueError("Q-tile accumulator carry requires nested Q-tile loops")
    if self.bwd_block_q_compute is not None and self.bwd_staged_kv_pipeline:
      raise ValueError("Q compute tiling cannot use the legacy KV pipeline")
    if self.fwd_native_output_normalization and not (
        self.fwd_output_scratch_seq_minor
        and self.compact_softmax_scratch and self.compact_stats_output
    ):
      raise ValueError("native output normalization requires sequence-minor output and compact statistics scratch")
    if self.fwd_output_seq_minor and not self.fwd_native_output_normalization:
      raise ValueError("sequence-minor forward output requires native normalization")
    if self.fwd_kvmajor_single_loop and not self.fwd_kvmajor_probabilities:
      raise ValueError("shared segment loop requires native KV-major probabilities")
    if self.fwd_kvmajor_mask_all_tiles and not self.fwd_kvmajor_single_loop:
      raise ValueError("unconditional forward masking requires a shared KV-major loop")
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)

    if self.dq_reduction_steps is not None and self.dq_reduction_steps != 3:
      raise ValueError(
          f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 3 or"
          " None are supported."
      )
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_kv_dkv_compute,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO: Select better parameters based on a heuristic.
    return SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )


def _attention_scope(config: SplashConfig, name: str, *, coarse: bool = False):
  if config.region_trace_mode == "fine" or (
      coarse and config.region_trace_mode == "coarse"
  ):
    return jax.named_scope(name)
  return contextlib.nullcontext()


to_i32 = lambda x: x.astype(jnp.int32)


def _kv_segment_column(ref, window, *, seq_minor: bool = False):
  """Load one int32 ID per KV token as a logical [KV, 1] column."""
  return ref[:1, window].T if seq_minor else ref[window, :1]


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
      computed_mask = mask_function(
          q_sequence, k_sequence
      )  # pytype: disable=wrong-arg-count
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
      if kv_segment_ids_seq_minor or kv_segment_ids_ref.shape[-1] == 1:
        kv_ids = jnp.broadcast_to(
            _kv_segment_column(
                kv_segment_ids_ref, k_slice,
                seq_minor=kv_segment_ids_seq_minor,
            ),
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
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
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

  if config.fwd_kvmajor_probabilities and not (
      config.max_logit_const == 0.0 and max_logit_value_ref is None
      and sinks_ref is None and attn_logits_soft_cap is None
      and config.use_base2_exp and config.combine_log2_scale
      and config.softmax_scale is not None
      and config.compact_softmax_scratch and config.fwd_output_scratch_seq_minor
      and not config.fwd_loop_carry and not config.fwd_staged_kv_pipeline
      and not config.fwd_pv_transposed_output
      and config.q_layout == config.k_layout == config.v_layout == QKVLayout.SEQ_MINOR
      and mask_ref is None and mask_function is None
      and q_segment_ids_ref is not None and kv_segment_ids_ref is not None
      and q_ref.dtype == k_ref.dtype == v_ref.dtype == jnp.bfloat16
  ):
    raise ValueError("fwd_kvmajor_probabilities requires fixed-shift segmented BF16 ViT attention and sequence-minor scratch")
  if config.fwd_kvmajor_fuse_normalizer and (
      not config.fwd_kvmajor_probabilities
      or config.fwd_kvmajor_sum_in_qmajor
      or head_dim_v % NUM_SUBLANES
  ):
    raise ValueError("fwd_kvmajor_fuse_normalizer requires native KV-major sum and sublane-aligned value width")

  @pl.when(should_initialize)
  def init():
    with _attention_scope(config, "splash_fwd_init", coarse=True):
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

  def load_state(ref):
    if config.compact_softmax_scratch:
      return jnp.broadcast_to(ref[:1, :].T, (bq, NUM_LANES))
    return ref[...]

  def store_state(ref, value):
    ref[...] = (
        jnp.broadcast_to(value[:, :1].T, (NUM_SUBLANES, bq))
        if config.compact_softmax_scratch
        else value
    )

  def load_output():
    value = o_scratch_ref[...]
    return value.T if config.fwd_output_scratch_seq_minor else value

  def store_output(value):
    o_scratch_ref[...] = value.T if config.fwd_output_scratch_seq_minor else value

  def compute_pv(probabilities, v):
    if config.fwd_pv_transposed_output:
      dims = TT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
      return lax.dot_general(v, probabilities, dims).T
    dims = NN_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    return lax.dot_general(probabilities, v, dims)

  def store_output_stat(ref, value):
    ref[...] = (
        jnp.broadcast_to(value[:, :1].T, (NUM_SUBLANES, bq))
        if config.compact_stats_output
        else value
    ).astype(ref.dtype)

  def kvmajor_body(kv_compute_index, has_partial_mask):
    window = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    with _attention_scope(config, "splash_fwd_qk_mxu"):
      logits = lax.dot_general(
          k_ref[:, window], q_ref[...], TN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      logits *= jnp.float32(config.softmax_scale * LOG2E)
    with _attention_scope(config, "splash_fwd_mask"):
      def mask_logits(value):
        q_ids = q_segment_ids_ref[:, :1].T
        kv_ids = kv_segment_ids_ref[:1, window].T
        return jnp.where(kv_ids == q_ids, value, mask_value)

      if (config.fwd_kvmajor_single_loop and config.segment_mask_on_partial_only
          and not config.fwd_kvmajor_mask_all_tiles):
        logits = lax.cond(should_not_mask, lambda value: value, mask_logits, logits)
      elif (config.fwd_kvmajor_mask_all_tiles
            or not config.segment_mask_on_partial_only or has_partial_mask):
        logits = mask_logits(logits)
    with _attention_scope(config, "splash_fwd_softmax"):
      probabilities = jnp.exp2(logits - max_logit_estimate)
      if not config.fwd_kvmajor_fuse_normalizer:
        current_l = (
            jnp.sum(probabilities.T, axis=-1)[None, :]
            if config.fwd_kvmajor_sum_in_qmajor
            else jnp.sum(probabilities, axis=0, keepdims=True)
        )
        l_scratch_ref[...] += jnp.broadcast_to(current_l, l_scratch_ref.shape)
    with _attention_scope(config, "splash_fwd_pv_mxu"):
      # The reference PV consumes FP32 P; do not introduce a BF16 cast here.
      values = v_ref[:, window]
      if config.fwd_kvmajor_fuse_normalizer:
        # Duplicate the unit row to match compact l_scratch's sublane layout.
        # These rows use the same FP32 P and MXU reduction as the output.
        values = jnp.concatenate(
            (values, jnp.ones((NUM_SUBLANES, bkv_compute), values.dtype)),
            axis=0,
        ).astype(jnp.float32)
        # HIGHEST's FP32 contraction requires both operands to be FP32.
        # Widening stored BF16 values is exact, not an input-precision change.
      output_t = lax.dot_general(
          values, probabilities, NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
          precision=(
              lax.Precision.HIGHEST
              if config.fwd_kvmajor_fuse_normalizer else None
          ),
      )
    with _attention_scope(config, "splash_fwd_output_accum"):
      if config.fwd_kvmajor_fuse_normalizer:
        o_scratch_ref[...] += output_t[:head_dim_v, :]
        l_scratch_ref[...] += output_t[head_dim_v:, :]
      else:
        o_scratch_ref[...] += output_t

  def body(kv_compute_index, carry, has_partial_mask=False):
    if config.fwd_kvmajor_probabilities:
      kvmajor_body(kv_compute_index, has_partial_mask)
      return
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    with _attention_scope(config, "splash_fwd_load_qk_state"):
      if config.fwd_loop_carry:
        m_prev, l_prev, _ = carry
      else:
        m_prev, l_prev = load_state(m_scratch_ref), load_state(l_scratch_ref)
      assert m_prev.shape == (bq, NUM_LANES)
      assert l_prev.shape == (bq, NUM_LANES)

      q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].T
      if config.use_base2_exp and config.softmax_scale is None:
        q *= LOG2E

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    with _attention_scope(config, "splash_fwd_qk_mxu"):
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

    with _attention_scope(config, "splash_fwd_mask"):
      qk = apply_mask_and_soft_cap()

    with _attention_scope(config, "splash_fwd_softmax"):
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

      l_curr = jax.lax.broadcast_in_dim(
          s_curr.sum(axis=-1), l_prev.shape, (0,)
      )
      assert l_curr.shape == (bq, NUM_LANES)

      if max_logit_estimate is None:
        alpha = exp(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        if not config.fwd_loop_carry:
          store_state(m_scratch_ref, m_next)
          store_state(l_scratch_ref, l_next)
      else:
        alpha = None
        l_next = l_curr + l_prev
        if not config.fwd_loop_carry:
          store_state(l_scratch_ref, l_next)

    with _attention_scope(config, "splash_fwd_pv_mxu"):
      if config.v_layout == HEAD_DIM_MINOR:
        v = v_ref[slice_k, :]
      else:
        v = v_ref[:, slice_k]
      o_curr = compute_pv(s_curr, v)

    with _attention_scope(config, "splash_fwd_output_accum"):
      o_prev = carry[2] if config.fwd_loop_carry else load_output()
      if max_logit_estimate is None:
        alpha_o = jnp.tile(alpha, (1, head_dim_v_repeats))
        alpha_o = alpha_o[..., :head_dim_v]
        o_next = alpha_o * o_prev + o_curr
      else:
        o_next = o_prev + o_curr
      if config.fwd_loop_carry:
        return m_prev if m_next is None else m_next, l_next, o_next
      store_output(o_next)

  assert bkv % bkv_compute == 0
  num_iters = (
      k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  )

  def run_inner_loop(has_partial_mask):
    loop_body = partial(body, has_partial_mask=has_partial_mask)
    if config.fwd_staged_kv_pipeline:
      if not (
          config.max_logit_const == 0.0
          and max_logit_value_ref is None
          and sinks_ref is None
          and attn_logits_soft_cap is None
          and config.use_base2_exp
          and config.combine_log2_scale
          and config.softmax_scale is not None
          and not config.fwd_loop_carry
          and config.q_layout == QKVLayout.SEQ_MINOR
          and config.k_layout == QKVLayout.SEQ_MINOR
          and config.v_layout == QKVLayout.SEQ_MINOR
          and q_ref.dtype == k_ref.dtype == v_ref.dtype == jnp.bfloat16
      ):
        raise ValueError("fwd_staged_kv_pipeline requires fixed-shift BF16 ViT attention")

      def prepare(i):
        window = pl.ds(i * bkv_compute, bkv_compute)
        with _attention_scope(config, "splash_fwd_qk_mxu"):
          logits = lax.dot_general(
              q_ref[...].T, k_ref[:, window], NN_DIM_NUMBERS,
              preferred_element_type=jnp.float32,
          )
          logits *= jnp.float32(config.softmax_scale * LOG2E)
        with _attention_scope(config, "splash_fwd_mask"):
          logits = _apply_mask_and_soft_cap(
              logits, mask_value, mask_ref, q_sequence_ref,
              q_segment_ids_ref if (not config.segment_mask_on_partial_only or has_partial_mask) else None,
              kv_segment_ids_ref if (not config.segment_mask_on_partial_only or has_partial_mask) else None,
              attn_logits_soft_cap=None, k_slice=window,
              k_offset=j * bkv + i * bkv_compute,
              bq=bq, mask_function=mask_function, has_partial_mask=has_partial_mask,
          )
        with _attention_scope(config, "splash_fwd_softmax"):
          return jnp.exp2(logits - max_logit_estimate)

      def consume(i, probabilities):
        window = pl.ds(i * bkv_compute, bkv_compute)
        with _attention_scope(config, "splash_fwd_softmax"):
          previous_l = load_state(l_scratch_ref)
          current_l = lax.broadcast_in_dim(
              probabilities.sum(axis=-1), previous_l.shape, (0,),
          )
          store_state(l_scratch_ref, current_l + previous_l)
        with _attention_scope(config, "splash_fwd_pv_mxu"):
          # Retain FP32 probabilities for PV, exactly as the reference does.
          output = compute_pv(probabilities, v_ref[:, window])
        with _attention_scope(config, "splash_fwd_output_accum"):
          store_output(load_output() + output)

      def step(i, previous):
        current = prepare(i)
        consume(i - 1, previous)
        return current

      probabilities = prepare(0)
      probabilities = lax.fori_loop(
          1, num_iters, step, probabilities, unroll=config.fwd_kv_unroll,
      )
      consume(num_iters - 1, probabilities)
    elif config.fwd_loop_carry:
      initial = (
          load_state(m_scratch_ref),
          load_state(l_scratch_ref),
          load_output(),
      )
      m, l, o = lax.fori_loop(0, num_iters, loop_body, initial, unroll=config.fwd_kv_unroll)
      store_state(m_scratch_ref, m)
      store_state(l_scratch_ref, l)
      store_output(o)
    else:
      lax.fori_loop(0, num_iters, loop_body, None, unroll=config.fwd_kv_unroll)

  if config.fwd_kvmajor_single_loop:
    # The native KV-major path only supports segment masks. Keep the full/
    # partial decision around masking, preserving ID 0 == 0, without emitting
    # two fully unrolled copies of the expensive dot/exp/PV.
    with _attention_scope(config, "splash_fwd_kv_loop", coarse=True):
      run_inner_loop(True)
  else:
    @pl.when(should_not_mask)
    def _():
      with _attention_scope(config, "splash_fwd_kv_loop", coarse=True):
        run_inner_loop(False)

    @pl.when(jnp.logical_not(should_not_mask))
    def _():
      with _attention_scope(config, "splash_fwd_kv_loop_partial", coarse=True):
        run_inner_loop(True)

  @pl.when(should_write)
  def end():
    with _attention_scope(config, "splash_fwd_output_drain", coarse=True):
      if config.fwd_native_output_normalization:
        # Keep the exact reciprocal -> multiply -> BF16 cast sequence. Avoid
        # transposing FP32 output or broadcasting statistics to [Q, 128].
        l_native = l_scratch_ref[...]
        m_native = m_scratch_ref[...]
        output_native = o_scratch_ref[...]
        if fuse_reciprocal:
          output_native *= 1.0 / l_native[:1, :]
        output_native = output_native.astype(o_ref.dtype)
        o_ref[...] = (
            output_native if config.fwd_output_seq_minor else output_native.T
        )
        if logsumexp_ref is not None:
          log = jnp.log2 if config.use_base2_exp else jnp.log
          logsumexp_ref[...] = (m_native + log(l_native)).astype(logsumexp_ref.dtype)
        if l_linear_ref is not None:
          l_linear_ref[...] = l_native.astype(l_linear_ref.dtype)
        if max_logits_ref is not None:
          max_logits_ref[...] = m_native.astype(max_logits_ref.dtype)
        return
      l = load_state(l_scratch_ref)
      m = load_state(m_scratch_ref)
      if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
        l_inv = jnp.tile(1.0 / l, (1, head_dim_v_repeats))
        l_inv = l_inv[..., :head_dim_v]
        o_ref[...] = (load_output() * l_inv).astype(o_ref.dtype)
      else:
        o_ref[...] = load_output().astype(o_ref.dtype)
      if logsumexp_ref is not None:
        assert logsumexp_ref.shape == (
            (NUM_SUBLANES, bq)
            if config.compact_stats_output
            else (bq, NUM_LANES)
        )
        log = jnp.log2 if config.use_base2_exp else jnp.log
        logsumexp = m + log(l)
        store_output_stat(logsumexp_ref, logsumexp)
      if l_linear_ref is not None:
        assert l_linear_ref.shape == (
            (NUM_SUBLANES, bq)
            if config.compact_stats_output
            else (bq, NUM_LANES)
        )
        store_output_stat(l_linear_ref, l)
      if max_logits_ref is not None:
        assert max_logits_ref.shape == (
            (NUM_SUBLANES, bq)
            if config.compact_stats_output
            else (bq, NUM_LANES)
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
    max_logit_value: jax.Array | None = None,
    save_max_logits: bool = True,
) -> base.SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  save_max_logits = save_max_logits or not fuse_reciprocal
  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
        f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
        f" {len(k.shape)}-dim one."
    )

  if k.shape[-1] != head_dim_qk:
    raise ValueError(
        f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
        f" {k.shape[-1]}."
    )

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

  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  dynamic_grid = mask_info.active_rows is not None

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)  # for pytype
    assert isinstance(segment_ids.kv, jax.Array)  # for pytype
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )
  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(
        f"Only one of {config.max_logit_const=} and"
        f" {max_logit_value=} can be set."
    )
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(
        jnp.atleast_1d(max_logit_value), (num_q_heads,)
    )

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout
  out_layout = (
      QKVLayout.SEQ_MINOR if config.fwd_output_seq_minor else QKVLayout.HEAD_DIM_MINOR
  )

  def unravel(f):
    def index_map(h, grid_idx, rows_ref, cols_ref, *_):
      if dynamic_grid:
        i = to_i32(rows_ref[grid_idx])
        j = to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h, i, j)

    return index_map

  def create_kv_index_map(layout):
    def index_map(h, i, j):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(lambda h, i, j: from_head_minor((h, i, 0), q_layout))
  out_index_map = unravel(lambda h, i, j: from_head_minor((h, i, 0), out_layout))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
    del h, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h, i, j: (0, j))

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
              k_layout,
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout
          ),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda h, i, j, *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
    sinks = jnp.broadcast_to(
        sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    in_specs += [None]

  if mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  if max_logit_value is not None:
    # reshape to allow sublane selection for vmap-ping and shard_map-ping
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
  else:
    in_specs.append(None)

  out_shapes = [
      jax.ShapeDtypeStruct(
          from_head_minor((num_q_heads, q_seq_len, head_dim_v), out_layout), q.dtype
      ),
  ]
  out_specs = [
      pl.BlockSpec(from_head_minor((None, bq, head_dim_v), out_layout), out_index_map),
  ]
  if save_residuals:
    logsumexp_index_map = unravel(
        lambda h, i, j, *_: (h, 0, i)
        if config.compact_stats_output
        else (h, i, 0)
    )
    stat_shape = (
        (num_q_heads, NUM_SUBLANES, q_seq_len)
        if config.compact_stats_output
        else (num_q_heads, q_seq_len, NUM_LANES)
    )
    stat_block = (
        (None, NUM_SUBLANES, bq)
        if config.compact_stats_output
        else (None, bq, NUM_LANES)
    )

    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct(stat_shape, jnp.float32)
        if fuse_reciprocal
        else None,
        # l_linear
        jax.ShapeDtypeStruct(stat_shape, jnp.float32)
        if not fuse_reciprocal
        else None,
        # max_logits
        jax.ShapeDtypeStruct(stat_shape, jnp.float32)
        if save_max_logits
        else None,
    ]
    out_specs += [
        pl.BlockSpec(stat_block, logsumexp_index_map)
        if fuse_reciprocal
        else None,
        pl.BlockSpec(stat_block, logsumexp_index_map)
        if not fuse_reciprocal
        else None,
        pl.BlockSpec(stat_block, logsumexp_index_map)
        if save_max_logits
        else None,
    ]
  else:
    out_shapes += [None, None, None]
    out_specs += [None, None, None]

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _fwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      partial_mask_blocks: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    matmul_flops = (
        2 * q_seq_len * kv_seq_len * head_dim_qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v
    )

    # This is an upper bound because `mask_sparsity` is actually the mean
    # sparsity of the non-fully masked **blocks**.
    total_flops = num_q_heads * matmul_flops * mask_sparsity

    # Count expensive exp() calls
    transcendentals = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity

    inputs_ = [q, k, v, q_segment_ids, kv_segment_ids, partial_mask_blocks]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  vmem_inputs = [
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      mask_info.partial_mask_blocks,
  ]
  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
      *vmem_inputs, out_shapes, fwd_mask_sparsity
  )

  if dynamic_grid:
    num_active_blocks = mask_info.num_active_blocks[0]
    grid = (num_q_heads, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (num_q_heads, kv_steps * (q_seq_len // bq))
    is_empty_attention_block = False

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM(
                    (NUM_SUBLANES, bq)
                    if config.compact_softmax_scratch
                    else (bq, NUM_LANES),
                    jnp.float32,
                ),  # m_scratch
                pltpu.VMEM(
                    (NUM_SUBLANES, bq)
                    if config.compact_softmax_scratch
                    else (bq, NUM_LANES),
                    jnp.float32,
                ),  # l_scratch
                pltpu.VMEM(
                    (head_dim_v, bq)
                    if config.fwd_output_scratch_seq_minor
                    else (bq, head_dim_v),
                    jnp.float32,
                ),  # o_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=config.fwd_vmem_limit_bytes,
            flags={
                "XLA_TPU_FORCE_LP_LLO_SCHEDULER": (
                    config.use_experimental_scheduler
                )
            },
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, l_linear, max_logits = all_out
  if config.fwd_output_seq_minor:
    out = out.mT

  # If there is no compute to do within an attention block, then we want to
  # initialize the output and residuals to default values. Otherwise, we will
  # read uninitialized memory. This is a common case in ring attention.
  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x

    return jnp.where(is_empty_attention_block, value, x)

  out = init_if_empty(out, 0.0)

  if save_residuals:
    if max_logits is not None:
      max_logits = init_if_empty(
          max_logits[:, 0, :]
          if config.compact_stats_output
          else max_logits[..., 0],
          mask_value,
      )

    if fuse_reciprocal:
      assert logsumexp is not None
      logsumexp = init_if_empty(
          logsumexp[:, 0, :]
          if config.compact_stats_output
          else logsumexp[..., 0],
          mask_value,
      )
    else:
      assert l_linear is not None
      log = jnp.log2 if config.use_base2_exp else jnp.log

      l = l_linear[:, 0, :] if config.compact_stats_output else l_linear[..., 0]
      logsumexp = max_logits + log(l)
      out = (out / l[..., None]).astype(out.dtype)
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
  if save_residuals:
    stats = {"logsumexp": logsumexp, "max_logits": max_logits}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return out, stats
  return out


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

  ret = _splash_attention_forward(  # pytype: disable=wrong-arg-types
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
      max_logit_value=max_logit_value,
  )
  if save_residuals:
    out, stats = ret
    if config.use_base2_exp:  # for user, output values in natural base
      stats["logsumexp"] = stats["logsumexp"] / LOG2E
      stats["max_logits"] = stats["max_logits"] / LOG2E
    return out, stats
  else:
    return ret


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
    max_logit_value: jax.Array | None = None,
) -> tuple[tuple[jax.Array], base.SplashResidualsType]:

  # TODO: add some higher order AD check that isn't save_residuals based.
  # if save_residuals:
  #   raise NotImplementedError("Higher-order AD not supported.")

  out, stats = _splash_attention_forward(  # pytype: disable=wrong-arg-types
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
      max_logit_value=max_logit_value,
  )
  logsumexp = stats["logsumexp"]  # save in the config base for the bwd pass
  if config.use_base2_exp:  # for user, output values in natural base
    stats["logsumexp"] = stats["logsumexp"] / LOG2E
    if stats["max_logits"] is not None:
      stats["max_logits"] = stats["max_logits"] / LOG2E
  residuals = q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info
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
):
  del mask_next_ref, active_cols_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E
  head_group_size = config.bwd_head_group_size

  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
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
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
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
    with _attention_scope(config, "splash_bwd_dkv_init", coarse=True):
      dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
      dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  def run_staged_pipeline(num_iters):
    # Keep this probe narrow: these are the production ViT arithmetic/layout
    # choices, not a replacement for all of Splash's supported configurations.
    if not (
        head_group_size == 1
        and config.q_layout == QKVLayout.SEQ_MINOR
        and config.k_layout == QKVLayout.SEQ_MINOR
        and config.v_layout == QKVLayout.SEQ_MINOR
        and config.bwd_do_seq_minor
        and config.bwd_dq_scratch_seq_minor
        and config.bwd_dkv_scratch_seq_minor
        and config.bwd_scale_after_dot
        and config.bwd_cast_before_transpose
        and not config.bwd_dp_before_qk
        and not config.bwd_keep_kv_seq_minor
        and not config.bwd_dq_contract_ds_axis0
        and not config.bwd_dk_transposed_output
        and not config.bwd_dv_transposed_output
        and not config.bwd_qmajor_probabilities
        and not config.bwd_dv_last
        and not config.bwd_dv_between_dq_dk
        and not config.bwd_reuse_bf16_probabilities
        and config.use_base2_exp
        and config.combine_log2_scale
        and config.softmax_scale is not None
        and attn_logits_soft_cap is None
        and dq_scratch_ref is not None
        and q_ref.dtype == k_ref.dtype == v_ref.dtype == do_ref.dtype == jnp.bfloat16
    ):
      raise ValueError("bwd_staged_kv_pipeline requires the ViT exact-layout configuration")

    def prepare_logits(i):
      window = pl.ds(i * bkv_compute, bkv_compute)
      q = q_ref[...]
      k = k_ref[:, window].T
      with _attention_scope(config, "splash_bwd_qk_recompute_mxu"):
        logits = lax.dot_general(k, q, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
        logits *= jnp.float32(config.softmax_scale * LOG2E)
      return logits

    def prepare_probabilities(i, logits):
      window = pl.ds(i * bkv_compute, bkv_compute)
      with _attention_scope(config, "splash_bwd_mask_softmax"):
        logits = _apply_mask_and_soft_cap(
            logits, mask_value, None, q_sequence_ref,
            q_segment_ids_ref, kv_segment_ids_ref,
            attn_logits_soft_cap=None, k_slice=window,
            k_offset=kv_index * bkv + i * bkv_compute,
            bq=bq, k_in_lanes=False, mask_function=None,
            has_partial_mask=True,
            kv_segment_ids_seq_minor=config.bwd_kv_segment_ids_seq_minor,
        )
        return jnp.exp2(logits - logsumexp_ref[:1, :])

    def prepare_dp(i):
      window = pl.ds(i * bkv_compute, bkv_compute)
      v, do = v_ref[:, window].T, do_ref[...]
      with _attention_scope(config, "splash_bwd_dp_mxu"):
        return lax.dot_general(v, do, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)

    def finish_prepare(p, dp):
      with _attention_scope(config, "splash_bwd_softmax_grad"):
        ds = (dp - di_ref[:1, :]) * p
      # Do not reuse BF16 P in dS: the reference consumes FP32 probabilities.
      return p.astype(do_ref.dtype), ds.astype(do_ref.dtype)

    def prepare(i):
      logits = prepare_logits(i)
      p = prepare_probabilities(i, logits)
      return finish_prepare(p, prepare_dp(i))

    def consume_dv(i, p):
      window = pl.ds(i * bkv_compute, bkv_compute)
      do = do_ref[...]
      with _attention_scope(config, "splash_bwd_dv_mxu_accum"):
        dv = lax.dot_general(p, do, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
        dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[:, window].T
        dv_scratch_ref[:, window] = dv.T

    def consume_dq(i, ds):
      window = pl.ds(i * bkv_compute, bkv_compute)
      k = k_ref[:, window].T
      with _attention_scope(config, "splash_bwd_dq_mxu_accum"):
        if config.bwd_dq_transposed_output:
          dq = lax.dot_general(k, ds, TN_DIM_NUMBERS, preferred_element_type=jnp.float32).T
        else:
          dq = lax.dot_general(ds.T, k, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
        dq *= jnp.float32(config.softmax_scale)
        dq_scratch_ref[...] += dq.T

    def consume_dk(i, ds):
      window = pl.ds(i * bkv_compute, bkv_compute)
      q = q_ref[...]
      with _attention_scope(config, "splash_bwd_dk_mxu_accum"):
        dk = lax.dot_general(ds, q, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
        dk *= jnp.float32(config.softmax_scale)
        dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[:, window].T
        dk_scratch_ref[:, window] = dk.T

    def consume(i, state):
      p, ds = state
      consume_dv(i, p)
      if config.bwd_dq_first:
        consume_dq(i, ds)
        consume_dk(i, ds)
      else:
        consume_dk(i, ds)
        consume_dq(i, ds)

    state = prepare(0)

    def step(i, previous):
      has_next = i + 1 < num_iters
      next_i = jnp.where(has_next, i + 1, 0) if config.bwd_staged_kv_wrap_tail else i + 1

      def produce(producer, fallback):
        if config.bwd_staged_kv_wrap_tail:
          return producer()
        return lax.cond(has_next, producer, fallback)

      if config.bwd_staged_kv_interleave:
        # Keep each gradient's KV accumulation order. Only independent next-
        # tile producer work is inserted between prior-tile consumers.
        empty = lambda: jnp.zeros(previous[0].shape, jnp.float32)
        logits = produce(lambda: prepare_logits(next_i), empty)
        consume_dv(i, previous[0])
        p = produce(lambda: prepare_probabilities(next_i, logits), empty)
        consume_dk(i, previous[1])
        dp = produce(lambda: prepare_dp(next_i), empty)
        consume_dq(i, previous[1])
        return finish_prepare(p, dp)
      else:
        current = produce(lambda: prepare(next_i), lambda: previous)
        consume(i, previous)
        return current

    # Use one consumer body, including the last tile, rather than duplicating
    # a drain outside the loop. Skip the tail producer or read valid tile 0;
    # in neither case may it access an out-of-bounds input window.
    lax.fori_loop(0, num_iters, step, state, unroll=False)

  def run_q_tiled_loop(num_kv_iters, has_partial_mask):
    # This probe retains outer DMA tiles and reduction/output handling. Only
    # internal compute is split; dK/dV reduction association may change.
    if not (
        head_group_size == q_heads_per_kv_head == 1
        and config.q_layout == config.k_layout == config.v_layout == QKVLayout.SEQ_MINOR
        and config.bwd_do_seq_minor and config.bwd_dq_scratch_seq_minor
        and config.bwd_dkv_scratch_seq_minor and config.bwd_scale_after_dot
        and config.bwd_dq_transposed_output and not config.bwd_dq_first
        and not config.bwd_staged_kv_pipeline and not config.bwd_dp_before_qk
        and not config.bwd_qmajor_probabilities and not config.bwd_keep_kv_seq_minor
        and not config.bwd_dk_transposed_output and not config.bwd_dv_transposed_output
        and not config.bwd_dv_last and not config.bwd_dv_between_dq_dk
        and not config.bwd_dq_contract_ds_axis0 and not config.bwd_reuse_bf16_probabilities
        and config.use_base2_exp and config.combine_log2_scale
        and config.softmax_scale is not None and attn_logits_soft_cap is None
        and dq_scratch_ref is not None and mask_ref is None and mask_function is None
        and q_segment_ids_ref is not None and kv_segment_ids_ref is not None
        and q_ref.dtype == k_ref.dtype == v_ref.dtype == do_ref.dtype == jnp.bfloat16
    ):
      raise ValueError("Q compute tiling requires the ViT transposed-dQ/dK-first layout")
    bq_compute = config.bwd_block_q_compute
    assert bq_compute is not None and bq % bq_compute == 0
    q_inner_steps = bq // bq_compute
    num_tiles = num_kv_iters * q_inner_steps

    def windows(i):
      return (
          pl.ds((i % q_inner_steps) * bq_compute, bq_compute),
          pl.ds((i // q_inner_steps) * bkv_compute, bkv_compute),
      )

    def prepare(i):
      q_window, kv_window = windows(i)
      q, k = q_ref[:, q_window], k_ref[:, kv_window]
      with _attention_scope(config, "splash_bwd_qk_recompute_mxu"):
        logits = lax.dot_general(k, q, TN_DIM_NUMBERS, preferred_element_type=jnp.float32)
        logits *= jnp.float32(config.softmax_scale * LOG2E)
      with _attention_scope(config, "splash_bwd_mask_softmax"):
        if not config.segment_mask_on_partial_only or has_partial_mask:
          q_ids = q_segment_ids_ref[:1, q_window]
          kv_ids = _kv_segment_column(
              kv_segment_ids_ref, kv_window,
              seq_minor=config.bwd_kv_segment_ids_seq_minor,
          )
          logits = jnp.where(kv_ids == q_ids, logits, mask_value)
        p = jnp.exp2(logits - logsumexp_ref[:1, q_window])
      with _attention_scope(config, "splash_bwd_dp_mxu"):
        dp = lax.dot_general(
            v_ref[:, kv_window], do_ref[:, q_window], TN_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
      with _attention_scope(config, "splash_bwd_softmax_grad"):
        ds = ((dp - di_ref[:1, q_window]) * p).astype(do_ref.dtype)
      # FP32 p is used for dS. Cast only at the reference's gradient-dot
      # boundaries, so the carried state does not lower softmax precision.
      return p.astype(do_ref.dtype), ds

    def consume(i, state, accumulators=None):
      q_window, kv_window = windows(i)
      p, ds = state
      if config.bwd_qtile_accumulator_carry:
        dk_acc, dv_acc = accumulators
      with _attention_scope(config, "splash_bwd_dv_mxu_accum"):
        dv = lax.dot_general(
            p, do_ref[:, q_window], NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        if config.bwd_qtile_accumulator_carry:
          dv_acc = dv_acc + dv.T
        else:
          dv_scratch_ref[:, kv_window] += dv.T
      with _attention_scope(config, "splash_bwd_dk_mxu_accum"):
        dk = lax.dot_general(
            ds, q_ref[:, q_window], NT_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        dk *= jnp.float32(config.softmax_scale)
        if config.bwd_qtile_accumulator_carry:
          dk_acc = dk_acc + dk.T
        else:
          dk_scratch_ref[:, kv_window] += dk.T
      with _attention_scope(config, "splash_bwd_dq_mxu_accum"):
        dq = lax.dot_general(
            k_ref[:, kv_window], ds, NN_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        dq *= jnp.float32(config.softmax_scale)
        dq_scratch_ref[:, q_window] += dq
      if config.bwd_qtile_accumulator_carry:
        return dk_acc, dv_acc

    if config.bwd_qtile_nested:
      def kv_step(kv_index, _):
        first = kv_index * q_inner_steps
        kv_window = pl.ds(kv_index * bkv_compute, bkv_compute)
        accumulators = None
        if config.bwd_qtile_accumulator_carry:
          with _attention_scope(config, "splash_bwd_qtile_acc_init"):
            accumulators = (
                dk_scratch_ref[:, kv_window], dv_scratch_ref[:, kv_window],
            )

        if config.bwd_qtile_pipeline:
          def q_step(q_index, state):
            previous, acc = state
            current = prepare(first + q_index)
            acc = consume(first + q_index - 1, previous, acc)
            return current, acc

          state, accumulators = lax.fori_loop(
              1, q_inner_steps, q_step, (prepare(first), accumulators),
              unroll=config.bwd_kv_unroll,
          )
          accumulators = consume(first + q_inner_steps - 1, state, accumulators)
        else:
          def q_step(q_index, acc):
            i = first + q_index
            return consume(i, prepare(i), acc)

          accumulators = lax.fori_loop(
              0, q_inner_steps, q_step, accumulators,
              unroll=config.bwd_kv_unroll,
          )

        if config.bwd_qtile_accumulator_carry:
          with _attention_scope(config, "splash_bwd_qtile_acc_drain"):
            dk_scratch_ref[:, kv_window], dv_scratch_ref[:, kv_window] = accumulators

      # Only the inner Q loop is unrolled. This bounds code expansion and
      # preserves the original Q-subtile accumulation order for each KV tile.
      lax.fori_loop(0, num_kv_iters, kv_step, None, unroll=False)
      return

    if config.bwd_qtile_pipeline:
      def step(i, previous):
        current = prepare(i)
        consume(i - 1, previous)
        return current

      state = lax.fori_loop(
          1, num_tiles, step, prepare(0), unroll=config.bwd_kv_unroll,
      )
      consume(num_tiles - 1, state)
    else:
      def step(i, _):
        consume(i, prepare(i))

      lax.fori_loop(0, num_tiles, step, None, unroll=config.bwd_kv_unroll)

  def qmajor_body(i, has_partial_mask):
    if not (
        head_group_size == 1
        and config.q_layout == config.k_layout == config.v_layout == QKVLayout.SEQ_MINOR
        and config.bwd_do_seq_minor
        and config.bwd_dq_scratch_seq_minor
        and config.bwd_dkv_scratch_seq_minor
        and config.bwd_scale_after_dot
        and not config.bwd_dk_transposed_output
        and not config.bwd_dv_transposed_output
        and not config.bwd_dq_contract_ds_axis0
        and not config.bwd_reuse_bf16_probabilities
        and config.use_base2_exp and config.combine_log2_scale
        and config.softmax_scale is not None
        and attn_logits_soft_cap is None
        and dq_scratch_ref is not None
        and mask_ref is None and mask_function is None
        and q_segment_ids_ref is not None and kv_segment_ids_ref is not None
        and q_ref.dtype == k_ref.dtype == v_ref.dtype == do_ref.dtype == jnp.bfloat16
    ):
      raise ValueError("bwd_qmajor_probabilities requires ViT sequence-minor BF16 inputs and scratch")
    window = pl.ds(i * bkv_compute, bkv_compute)
    q, k, v, do = q_ref[...], k_ref[:, window], v_ref[:, window], do_ref[...]

    def compute_dp():
      with _attention_scope(config, "splash_bwd_dp_mxu"):
        return lax.dot_general(do, v, TN_DIM_NUMBERS, preferred_element_type=jnp.float32)

    dp = compute_dp() if config.bwd_dp_before_qk else None
    with _attention_scope(config, "splash_bwd_qk_recompute_mxu"):
      logits = lax.dot_general(q, k, TN_DIM_NUMBERS, preferred_element_type=jnp.float32)
      logits *= jnp.float32(config.softmax_scale * LOG2E)
    with _attention_scope(config, "splash_bwd_mask_softmax"):
      if not config.segment_mask_on_partial_only or has_partial_mask:
        q_ids = q_segment_ids_ref[:1, :].T
        kv_ids = _kv_segment_column(
            kv_segment_ids_ref, window,
            seq_minor=config.bwd_kv_segment_ids_seq_minor,
        ).T
        logits = jnp.where(q_ids == kv_ids, logits, mask_value)
      p = jnp.exp2(logits - logsumexp_ref[:1, :].T)
      p_bf16 = p.astype(do.dtype)

    def compute_dv():
      with _attention_scope(config, "splash_bwd_dv_mxu_accum"):
        dv_t = lax.dot_general(do, p_bf16, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
        dv_scratch_ref[:, window] += dv_t.astype(dv_scratch_ref.dtype)

    if not config.bwd_dv_last and not config.bwd_dv_between_dq_dk:
      compute_dv()
    with _attention_scope(config, "splash_bwd_softmax_grad"):
      if dp is None:
        dp = compute_dp()
      ds = ((dp - di_ref[:1, :].T) * p).astype(do.dtype)

    def compute_dk():
      with _attention_scope(config, "splash_bwd_dk_mxu_accum"):
        dk_t = lax.dot_general(q, ds, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
        dk_t *= jnp.float32(config.softmax_scale)
        dk_scratch_ref[:, window] += dk_t.astype(dk_scratch_ref.dtype)

    if not config.bwd_dq_first:
      compute_dk()
      if config.bwd_dv_between_dq_dk:
        compute_dv()
    with _attention_scope(config, "splash_bwd_dq_mxu_accum"):
      if config.bwd_dq_transposed_output:
        dq_t = lax.dot_general(k, ds, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
      else:
        dq_t = lax.dot_general(ds, k, NT_DIM_NUMBERS, preferred_element_type=jnp.float32).T
      dq_t *= jnp.float32(config.softmax_scale)
      dq_scratch_ref[...] += dq_t
    if config.bwd_dq_first:
      if config.bwd_dv_between_dq_dk:
        compute_dv()
      compute_dk()
    if config.bwd_dv_last:
      compute_dv()

  def body(i, _, has_partial_mask=False):
    if config.bwd_qmajor_probabilities:
      qmajor_body(i, has_partial_mask)
      return

    slice_k = pl.ds(i * bkv_compute, bkv_compute)

    def per_head(head_offset):
      def head_ref(ref):
        return ref if head_group_size == 1 else ref.at[head_offset]

      # Keep the head selection as a transformed Ref. Loading the head first
      # would turn the subsequent KV-loop slice into a dynamic_slice primitive,
      # which Mosaic TPU does not lower.
      with _attention_scope(config, "splash_bwd_load_inputs"):
        q = head_ref(q_ref)[...]
        if config.use_base2_exp and config.softmax_scale is None:
          scaled_q = q * LOG2E
        else:
          scaled_q = q

      def _load_kv(ref, layout):
        ref = head_ref(ref)
        if layout == HEAD_DIM_MINOR:
          return ref[slice_k, :]
        value = ref[:, slice_k]
        return value if config.bwd_keep_kv_seq_minor else value.T

      with _attention_scope(config, "splash_bwd_load_inputs"):
        k = _load_kv(k_ref, config.k_layout)
        v = _load_kv(v_ref, config.v_layout)
        logsumexp = head_ref(logsumexp_ref)[:1, :]
        do = head_ref(do_ref)[...]
        di = head_ref(di_ref)[:1, :]

      v_is_seq_minor = (
          config.bwd_keep_kv_seq_minor
          and config.v_layout == QKVLayout.SEQ_MINOR
      )
      if config.bwd_do_seq_minor:
        dp_dims = TN_DIM_NUMBERS if v_is_seq_minor else NN_DIM_NUMBERS
      else:
        dp_dims = TT_DIM_NUMBERS if v_is_seq_minor else NT_DIM_NUMBERS

      def compute_dp():
        with _attention_scope(config, "splash_bwd_dp_mxu"):
          return lax.dot_general(
              v,
              do,
              dp_dims,
              preferred_element_type=jnp.float32,
          )

      dp = compute_dp() if config.bwd_dp_before_qk else None

      if config.bwd_keep_kv_seq_minor and config.k_layout == QKVLayout.SEQ_MINOR:
        qk_dims = (
            TT_DIM_NUMBERS
            if config.q_layout == HEAD_DIM_MINOR
            else TN_DIM_NUMBERS
        )
      else:
        qk_dims = (
            NT_DIM_NUMBERS
            if config.q_layout == HEAD_DIM_MINOR
            else NN_DIM_NUMBERS
        )
      with _attention_scope(config, "splash_bwd_qk_recompute_mxu"):
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

      with _attention_scope(config, "splash_bwd_mask_softmax"):
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
            kv_segment_ids_seq_minor=config.bwd_kv_segment_ids_seq_minor,
        )
        exp = jnp.exp2 if config.use_base2_exp else jnp.exp
        p = exp(qk - logsumexp)
        p_bf16 = p.astype(do.dtype)
        if config.bwd_reuse_bf16_probabilities:
          # dV already consumes BF16 probabilities. Reuse the same value for
          # dS to shorten the live range of the full-size FP32 probability tile.
          p = p_bf16.astype(jnp.float32)

      def compute_dv():
        with _attention_scope(config, "splash_bwd_dv_mxu_accum"):
          if config.bwd_dv_transposed_output:
            dv = lax.dot_general(
                do, p_bf16,
                NT_DIM_NUMBERS if config.bwd_do_seq_minor else TT_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            ).T
          else:
            dv = lax.dot_general(
                p_bf16, do,
                NT_DIM_NUMBERS if config.bwd_do_seq_minor else NN_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
          scratch_ref = head_ref(dv_scratch_ref)
          if config.bwd_dkv_scratch_seq_minor:
            dv = dv.astype(dv_scratch_ref.dtype) + scratch_ref[:, slice_k].T
            scratch_ref[:, slice_k] = dv.T
          else:
            dv = dv.astype(dv_scratch_ref.dtype) + scratch_ref[slice_k, :]
            scratch_ref[slice_k, :] = dv

      if not config.bwd_dv_last and not config.bwd_dv_between_dq_dk:
        compute_dv()

      with _attention_scope(config, "splash_bwd_softmax_grad"):
        if dp is None:
          dp = compute_dp()
        ds = (dp - di) * p
        if attn_logits_soft_cap is not None:
          normalized = qk_uncapped / attn_logits_soft_cap
          d = jnp.tanh(normalized)
          ds = ds * (1 - d * d)
        if config.softmax_scale is not None and not config.bwd_scale_after_dot:
          ds *= jnp.float32(config.softmax_scale)

      def compute_dk():
        with _attention_scope(config, "splash_bwd_dk_mxu_accum"):
          if config.bwd_dk_transposed_output:
            dk = lax.dot_general(
                q, ds.astype(do.dtype),
                TT_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            ).T
          else:
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
          scratch_ref = head_ref(dk_scratch_ref)
          if config.bwd_dkv_scratch_seq_minor:
            dk = dk.astype(dk_scratch_ref.dtype) + scratch_ref[:, slice_k].T
            scratch_ref[:, slice_k] = dk.T
          else:
            dk = dk.astype(dk_scratch_ref.dtype) + scratch_ref[slice_k, :]
            scratch_ref[slice_k, :] = dk

      if not config.bwd_dq_first:
        compute_dk()
        if config.bwd_dv_between_dq_dk:
          compute_dv()
      if dq_scratch_ref is not None or dq_ref is not None:
        with _attention_scope(config, "splash_bwd_dq_mxu_accum"):
          if config.bwd_dq_transposed_output:
            dq_dims = (
                NN_DIM_NUMBERS
                if config.bwd_keep_kv_seq_minor
                and config.k_layout == QKVLayout.SEQ_MINOR
                else TN_DIM_NUMBERS
            )
            dq_transposed = lax.dot_general(
                k, ds.astype(k.dtype), dq_dims,
                preferred_element_type=jnp.float32,
            )
            # Sequence-minor scratch cancels this logical-output transpose.
            dq = dq_transposed.T
          elif config.bwd_dq_contract_ds_axis0:
            dq_dims = (
                TT_DIM_NUMBERS
                if config.bwd_keep_kv_seq_minor
                and config.k_layout == QKVLayout.SEQ_MINOR
                else TN_DIM_NUMBERS
            )
            dq = lax.dot_general(
                ds.astype(k.dtype),
                k,
                dq_dims,
                preferred_element_type=jnp.float32,
            )
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
            scratch_update = (
                dq.T if config.bwd_dq_scratch_seq_minor else dq
            )
            if head_group_size == 1:
              dq_scratch_ref[...] += scratch_update
            else:
              head_ref(dq_scratch_ref)[...] += scratch_update
          else:
            # Compute block size == memory block size
            if head_group_size == 1:
              if dq_alias is not None:
                dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
              else:
                dq_ref[...] = dq.astype(dq_ref.dtype)
            else:
              dq_out_ref = head_ref(dq_ref)
              if dq_alias is not None:
                dq_out_ref[...] = (
                    head_ref(dq_alias)[...] + dq.astype(dq_ref.dtype)
                )
              else:
                dq_out_ref[...] = dq.astype(dq_ref.dtype)

      if config.bwd_dq_first:
        if config.bwd_dv_between_dq_dk:
          compute_dv()
        compute_dk()
      if config.bwd_dv_last:
        compute_dv()

    for head_offset in range(head_group_size):
      per_head(head_offset)

  if dq_scratch_ref is not None:
    with _attention_scope(config, "splash_bwd_dq_init", coarse=True):
      dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_alias is not None:
    with _attention_scope(config, "splash_bwd_dq_init", coarse=True):
      dq_ref[...] = dq_alias[...]
  else:
    with _attention_scope(config, "splash_bwd_dq_init", coarse=True):
      dq_ref[...] = jnp.zeros_like(dq_ref)

  k_seq_axis = 0 if config.k_layout is HEAD_DIM_MINOR else 1
  if head_group_size > 1:
    k_seq_axis += 1
  num_iters = k_ref.shape[k_seq_axis] // bkv_compute

  def run_inner_loop(has_partial_mask):
    if config.bwd_block_q_compute is not None:
      run_q_tiled_loop(num_iters, has_partial_mask)
    else:
      lax.fori_loop(
          0, num_iters, partial(body, has_partial_mask=has_partial_mask), None,
          unroll=config.bwd_kv_unroll,
      )

  if config.bwd_staged_kv_pipeline and not config.bwd_single_segment_mask_body:
    raise ValueError("bwd_staged_kv_pipeline requires a single segment-mask body")
  if config.bwd_single_segment_mask_body:
    if (
        mask_ref is not None
        or mask_function is not None
        or q_segment_ids_ref is None
        or kv_segment_ids_ref is None
    ):
      raise ValueError("bwd_single_segment_mask_body requires segment-only masks")

    @pl.when(should_run)
    def _():
      with _attention_scope(config, "splash_bwd_kv_loop_partial", coarse=True):
        if config.bwd_staged_kv_pipeline:
          run_staged_pipeline(num_iters)
        else:
          run_inner_loop(True)
  else:
    @pl.when(jnp.logical_and(should_not_mask, should_run))
    def _():
      with _attention_scope(config, "splash_bwd_kv_loop", coarse=True):
        run_inner_loop(False)

    @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
    def _():
      with _attention_scope(config, "splash_bwd_kv_loop_partial", coarse=True):
        run_inner_loop(True)

  if dq_scratch_ref is not None:
    with _attention_scope(config, "splash_bwd_dq_output_drain", coarse=True):
      dq_scratch = dq_scratch_ref[...]
      if config.bwd_dq_scratch_seq_minor != config.bwd_dq_output_seq_minor:
        dq_scratch = jnp.swapaxes(dq_scratch, -1, -2)
      if dq_alias is not None:
        dq_ref[...] = dq_alias[...] + dq_scratch.astype(dq_ref.dtype)
      else:
        dq_ref[...] = dq_scratch.astype(dq_ref.dtype)

  if dk_alias is None:
    assert dv_alias is None

    def format_dkv_output(scratch_ref):
      value = scratch_ref[...]
      if (
          config.bwd_dkv_scratch_seq_minor
          != config.bwd_dkv_output_seq_minor
      ):
        value = jnp.swapaxes(value, -1, -2)
      return value

    @pl.when(should_write)
    def _():
      with _attention_scope(config, "splash_bwd_dkv_output_drain", coarse=True):
        dk = format_dkv_output(dk_scratch_ref)
        dv = format_dkv_output(dv_scratch_ref)
        dk_ref[...] = dk.astype(dk_ref.dtype)
        dv_ref[...] = dv.astype(dv_ref.dtype)

  else:
    q_head = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _():
      with _attention_scope(config, "splash_bwd_dkv_output_drain", coarse=True):
        dk = dk_scratch_ref[...]
        dv = dv_scratch_ref[...]
        if config.bwd_dkv_scratch_seq_minor:
          dk = jnp.swapaxes(dk, -1, -2)
          dv = jnp.swapaxes(dv, -1, -2)
        dk_ref[...] = dk.astype(dk_ref.dtype)
        dv_ref[...] = dv.astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _():
      with _attention_scope(config, "splash_bwd_dkv_output_drain", coarse=True):
        dk = dk_scratch_ref[...]
        dv = dv_scratch_ref[...]
        if config.bwd_dkv_scratch_seq_minor:
          dk = jnp.swapaxes(dk, -1, -2)
          dv = jnp.swapaxes(dv, -1, -2)
        dk_ref[...] = dk_alias[...] + dk.astype(dk_ref.dtype)
        dv_ref[...] = dv_alias[...] + dv.astype(dv_ref.dtype)


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
  head_group_size = config.bwd_head_group_size
  if config.bwd_dq_output_seq_minor:
    if is_mqa or q_heads_per_kv_head != 1 or head_group_size != 1:
      raise NotImplementedError(
          "Sequence-minor dQ output currently supports ungrouped MHA only"
      )
    if bkv == bkv_compute:
      raise ValueError("sequence-minor dQ output requires a dQ scratch window")
  if head_group_size < 1:
    raise ValueError(f"{head_group_size=} must be positive")
  if head_group_size > 1:
    if is_mqa or q_heads_per_kv_head != 1:
      raise NotImplementedError("Backward head grouping currently supports MHA only")
    if num_q_heads % head_group_size:
      raise ValueError(
          f"{num_q_heads=} must be divisible by {head_group_size=}"
      )
  if config.bwd_dkv_output_seq_minor:
    if is_mqa or q_heads_per_kv_head != 1 or head_group_size != 1:
      raise NotImplementedError(
          "Sequence-minor dK/dV outputs currently support ungrouped MHA only"
      )

  if dynamic_grid:

    def unravel(f):
      def index_map(h, grid_idx, rows_ref, cols_ref, *_):
        j = to_i32(rows_ref[grid_idx])
        i = to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    grid_size = mask_info.num_active_blocks[0]
    grid = (num_q_heads // head_group_size, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads // head_group_size, q_steps)

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

  head_block = None if head_group_size == 1 else head_group_size
  q_spec = pl.BlockSpec(
      from_head_minor((head_block, bq, head_dim_qk), config.q_layout),
      q_index_map,
  )

  o_spec = pl.BlockSpec((head_block, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk)
          if is_mqa
          else (head_block, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v)
          if is_mqa
          else (head_block, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    layout = (
        QKVLayout.SEQ_MINOR
        if config.bwd_dkv_output_seq_minor
        else QKVLayout.HEAD_DIM_MINOR
    )
    return from_head_minor((*prefix, j, 0), layout)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk)
          if is_mqa
          else (head_block, bkv, head_dim_qk),
          QKVLayout.SEQ_MINOR
          if config.bwd_dkv_output_seq_minor
          else QKVLayout.HEAD_DIM_MINOR,
      ),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v)
          if is_mqa
          else (head_block, bkv, head_dim_v),
          QKVLayout.SEQ_MINOR
          if config.bwd_dkv_output_seq_minor
          else QKVLayout.HEAD_DIM_MINOR,
      ),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))
    if config.bwd_compact_segment_ids:
      q_segment_spec = pl.BlockSpec((1, bq), q_segment_ids_index_map)
      kv_segment_spec = pl.BlockSpec((bkv, 1), kv_segment_ids_index_map)
      q_segment_ids = segment_ids.q[None, :]
      kv_segment_ids = segment_ids.kv[:, None]
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
    if config.bwd_kv_segment_ids_seq_minor:
      kv_segment_spec = pl.BlockSpec(
          (1, bkv), unravel(lambda h, i, j: (0, j))
      )
      kv_segment_ids = segment_ids.kv[None, :]
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  if config.bwd_do_seq_minor:
    do_spec = pl.BlockSpec(
        (head_block, head_dim_v, bq), unravel(lambda h, i, j: (h, 0, i))
    )
  else:
    do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec(
      (head_block, NUM_SUBLANES, bq), logsumexp_index_map
  )
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec(
      (head_block, NUM_SUBLANES, bq), logsumexp_index_map
  )
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
      QKVLayout.SEQ_MINOR if config.bwd_dq_output_seq_minor
      else QKVLayout.HEAD_DIM_MINOR
  )
  dq_block_shape = (None, head_block, *from_head_minor((bq, head_dim_qk), dq_layout))
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
            if config.bwd_dq_scratch_seq_minor
            else (bq, head_dim_qk)
        )
        if head_group_size == 1
        else (
            (head_group_size, head_dim_qk, bq)
            if config.bwd_dq_scratch_seq_minor
            else (head_group_size, bq, head_dim_qk)
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

  dk_shape = k.mT.shape if config.bwd_dkv_output_seq_minor else k.shape
  dv_shape = v.mT.shape if config.bwd_dkv_output_seq_minor else v.shape
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
      do.mT if config.bwd_do_seq_minor else do,
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
      True,  # q_segment_ids
      True,  # kv_segment_ids
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
              if config.bwd_dkv_scratch_seq_minor
              else (bkv, head_dim_qk)
          )
          if head_group_size == 1
          else (
              (head_group_size, head_dim_qk, bkv)
              if config.bwd_dkv_scratch_seq_minor
              else (head_group_size, bkv, head_dim_qk)
          ),
          jnp.float32,
      ),
      pltpu.VMEM(
          (
              (head_dim_v, bkv)
              if config.bwd_dkv_scratch_seq_minor
              else (bkv, head_dim_v)
          )
          if head_group_size == 1
          else (
              (head_group_size, head_dim_v, bkv)
              if config.bwd_dkv_scratch_seq_minor
              else (head_group_size, bkv, head_dim_v)
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
  if config.bwd_fuse_segment_id_inputs:
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
            dimension_semantics=(
                ("parallel", "arbitrary")
                if config.bwd_parallel_heads
                and dynamic_grid
                and q_heads_per_kv_head == 1
                else ("arbitrary",) * len(grid)
            ),
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
    if config.bwd_dq_output_seq_minor:
      dq = dq.mT
    if config.bwd_dkv_output_seq_minor:
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
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

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
    if self.fwd_mask_info.block_mask is not None:
      block_mask_shape = self.fwd_mask_info.block_mask.shape
      try:
        sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
            "The sharding must divide the mask blocks evenly between devices"
        ) from exc

    if len(sharding.spec) != 1:
      raise ValueError("Only q sequence sharding is supported.")

    _resolve_spec = lambda x: sharding.spec if x is not None else None
    mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
        mask_next=_resolve_spec(self.fwd_mask_info.mask_next),
        active_rows=_resolve_spec(self.fwd_mask_info.active_rows),
        active_cols=_resolve_spec(self.fwd_mask_info.active_cols),
        num_active_blocks=_resolve_spec(self.fwd_mask_info.num_active_blocks),
        block_mask=_resolve_spec(self.fwd_mask_info.block_mask),
        partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=_resolve_spec(self.fwd_mask_info.q_sequence),
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dkv_mask_info is not None else None,
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
    return SplashAttentionKernel(
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
    q_seq_shards: int,
):
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

  return SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
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
  )

  # If the input mask is replicated we don't need to call shard_map.
  if mask_spec is None:
    fwd_mask_info, dkv_mask_info = process_mask_shard(mask)
    kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
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
  kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
  kernel_spec = SplashAttentionKernel(*out_specs, **kwargs)

  return (kernel, kernel_spec)


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(make_splash_mha, q_seq_shards=1)

make_splash_mqa_single_device = partial(make_splash_mqa, q_seq_shards=1)

make_dynamic_splash_mqa = partial(_make_dynamic_splash_attention, is_mqa=True)
make_dynamic_splash_mha = partial(_make_dynamic_splash_attention, is_mqa=False)
