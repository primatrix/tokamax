# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Full-shape BF16 Splash scheduling screen with explicit precision evidence.

Non-bitwise candidates are measured for diagnosis but are never marked accepted.
Backward tiling changes replace the residual MaskInfo, not just the config.
Backward screens share forward activations and residual statistics unchanged.
Forward screens also validate outputs, residual statistics, and all gradients.
Joint screens compile forward and backward together using candidate residuals.
"""

import argparse
import dataclasses
import functools
import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench
from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy


_EXACT_LAYOUT = dict(
    bwd_dq_scratch_seq_minor=True,
    bwd_dkv_scratch_seq_minor=True,
    bwd_dkv_output_seq_minor=True,
    bwd_fuse_segment_id_inputs=True,
    bwd_do_seq_minor=True,
)

_DQ_DK_FIRST = _EXACT_LAYOUT | dict(
    bwd_dq_transposed_output=True, bwd_dq_first=False,
)

_FWD_KVMAJOR = dict(
    fwd_kvmajor_probabilities=True,
    compact_softmax_scratch=True, fwd_output_scratch_seq_minor=True,
)

_FWD_FAST = _FWD_KVMAJOR | dict(block_q=2048, block_kv_compute=512)

_FWD_BRANCHLESS = _FWD_KVMAJOR | dict(
    fwd_native_output_normalization=True, fwd_output_seq_minor=True,
    fwd_kvmajor_single_loop=True, fwd_kvmajor_mask_all_tiles=True,
)


def variants(phase="backward"):
  if phase == "joint":
    return [
        ("pr13", {}),
        ("bwd_fast", _DQ_DK_FIRST),
        ("fwd_fast", _FWD_FAST),
        ("joint_fast", _FWD_FAST | _DQ_DK_FIRST),
        ("joint_native_drain", _FWD_FAST | _DQ_DK_FIRST | dict(fwd_native_output_normalization=True)),
        ("joint_native_output", _FWD_FAST | _DQ_DK_FIRST | dict(fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("joint_q4096_native_output", _FWD_KVMAJOR | _DQ_DK_FIRST | dict(block_q=4096, block_kv=4096, fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
    ]
  if phase == "forward":
    return [
        ("pr13", {}),
        ("branchless_q2048_c512", _FWD_BRANCHLESS | dict(block_q=2048, block_kv_compute=512)),
        ("branchless_q4096_k4096", _FWD_BRANCHLESS | dict(block_q=4096, block_kv=4096)),
        ("branchless_q4096", _FWD_BRANCHLESS | dict(block_q=4096)),
        ("branchless_q4096_c512", _FWD_BRANCHLESS | dict(block_q=4096, block_kv_compute=512)),
        ("single_loop", _FWD_FAST | dict(fwd_kvmajor_single_loop=True)),
        ("single_loop_native_drain", _FWD_FAST | dict(fwd_kvmajor_single_loop=True, fwd_native_output_normalization=True)),
        ("single_loop_native_output", _FWD_FAST | dict(fwd_kvmajor_single_loop=True, fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("single_loop_q4096", _FWD_KVMAJOR | dict(block_q=4096, fwd_kvmajor_single_loop=True, fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("single_loop_q4096_c512", _FWD_KVMAJOR | dict(block_q=4096, block_kv_compute=512, fwd_kvmajor_single_loop=True, fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("single_loop_q4096_k4096", _FWD_KVMAJOR | dict(block_q=4096, block_kv=4096, fwd_kvmajor_single_loop=True, fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("native_drain", _FWD_FAST | dict(fwd_native_output_normalization=True)),
        ("native_output", _FWD_FAST | dict(fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("q4096_k4096_native_drain", _FWD_KVMAJOR | dict(block_q=4096, block_kv=4096, fwd_native_output_normalization=True)),
        ("q4096_k4096_native_output", _FWD_KVMAJOR | dict(block_q=4096, block_kv=4096, fwd_native_output_normalization=True, fwd_output_seq_minor=True)),
        ("kvmajor", _FWD_KVMAJOR),
        ("kvmajor_qsum", _FWD_KVMAJOR | dict(fwd_kvmajor_sum_in_qmajor=True)),
        ("kvmajor_qsum_u4", _FWD_KVMAJOR | dict(fwd_kvmajor_sum_in_qmajor=True, fwd_kv_unroll=4)),
        ("kvmajor_fused_l", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True)),
        ("kvmajor_fused_l_u4", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, fwd_kv_unroll=4)),
        ("kvmajor_fused_l_u8", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, fwd_kv_unroll=8)),
        ("kvmajor_fused_l_q2048", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_q=2048)),
        ("kvmajor_fused_l_q4096", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_q=4096)),
        ("kvmajor_fused_l_q8192", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_q=8192)),
        ("kvmajor_fused_l_q4096_u8", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_q=4096, fwd_kv_unroll=8)),
        ("kvmajor_fused_l_c512", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_kv_compute=512)),
        ("kvmajor_u2", _FWD_KVMAJOR | dict(fwd_kv_unroll=2)),
        ("kvmajor_u4", _FWD_KVMAJOR | dict(fwd_kv_unroll=4)),
        ("kvmajor_u8", _FWD_KVMAJOR | dict(fwd_kv_unroll=8)),
        ("kvmajor_c512", _FWD_KVMAJOR | dict(block_kv_compute=512)),
        ("kvmajor_q512", _FWD_KVMAJOR | dict(block_q=512)),
        ("kvmajor_q2048", _FWD_KVMAJOR | dict(block_q=2048)),
        ("kvmajor_q4096", _FWD_KVMAJOR | dict(block_q=4096)),
        ("kvmajor_q8192", _FWD_KVMAJOR | dict(block_q=8192)),
        ("kvmajor_q4096_u8", _FWD_KVMAJOR | dict(block_q=4096, fwd_kv_unroll=8)),
        ("kvmajor_q4096_k4096", _FWD_KVMAJOR | dict(block_q=4096, block_kv=4096)),
        ("kvmajor_q8192_k2048", _FWD_KVMAJOR | dict(block_q=8192, block_kv=2048)),
        ("kvmajor_q2048_c512", _FWD_KVMAJOR | dict(block_q=2048, block_kv_compute=512)),
        ("kvmajor_fused_l_q4096_k4096", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_q=4096, block_kv=4096)),
        ("kvmajor_fused_l_q2048_c512", _FWD_KVMAJOR | dict(fwd_kvmajor_fuse_normalizer=True, block_q=2048, block_kv_compute=512)),
        ("kvmajor_scheduler", _FWD_KVMAJOR | dict(use_experimental_scheduler=True)),
        ("output_seqminor", dict(fwd_output_scratch_seq_minor=True)),
        ("pv_transposed", dict(fwd_pv_transposed_output=True)),
        ("pv_transposed_seqminor", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True)),
        ("pv_transposed_seqminor_q512", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, block_q=512)),
        ("pv_transposed_seqminor_q2048", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, block_q=2048)),
        ("pv_transposed_seqminor_c512", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, block_kv_compute=512)),
        ("pv_transposed_seqminor_scheduler", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, use_experimental_scheduler=True)),
        ("pv_u2", dict(fwd_pv_transposed_output=True, fwd_kv_unroll=2)),
        ("pv_u4", dict(fwd_pv_transposed_output=True, fwd_kv_unroll=4)),
        ("pv_u8", dict(fwd_pv_transposed_output=True, fwd_kv_unroll=8)),
        ("pv_seq_u2", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, fwd_kv_unroll=2)),
        ("pv_seq_u4", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, fwd_kv_unroll=4)),
        ("pv_seq_u8", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, fwd_kv_unroll=8)),
        ("pv_seq_q2048_u4", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, fwd_kv_unroll=4, block_q=2048)),
        ("pv_seq_c512_u4", dict(fwd_pv_transposed_output=True, fwd_output_scratch_seq_minor=True, fwd_kv_unroll=4, block_kv_compute=512)),
        ("rolled", dict(fwd_kv_unroll=False)),
        ("unroll2", dict(fwd_kv_unroll=2)),
        ("unroll4", dict(fwd_kv_unroll=4)),
        ("unroll8", dict(fwd_kv_unroll=8)),
        ("pipeline", dict(fwd_staged_kv_pipeline=True, fwd_kv_unroll=False)),
        ("pipeline_u2", dict(fwd_staged_kv_pipeline=True, fwd_kv_unroll=2)),
        ("pipeline_u4", dict(fwd_staged_kv_pipeline=True, fwd_kv_unroll=4)),
        ("pipeline_c512_u2", dict(fwd_staged_kv_pipeline=True, fwd_kv_unroll=2, block_kv_compute=512)),
        ("loop_carry", dict(fwd_loop_carry=True)),
        ("loop_carry_compact", dict(fwd_loop_carry=True, compact_softmax_scratch=True)),
        ("q2048_carry", dict(fwd_loop_carry=True, block_q=2048)),
        ("q512_carry", dict(fwd_loop_carry=True, block_q=512)),
        ("c512_carry", dict(fwd_loop_carry=True, block_kv_compute=512)),
        ("q2048_c512_carry", dict(fwd_loop_carry=True, block_q=2048, block_kv_compute=512)),
        ("scheduler_carry", dict(fwd_loop_carry=True, use_experimental_scheduler=True)),
        ("scheduler", dict(use_experimental_scheduler=True)),
    ]
  return [
      ("pr13", {}),
      ("qtile1024", _DQ_DK_FIRST | dict(bwd_block_q_compute=1024)),
      ("qtile1024_pipeline", _DQ_DK_FIRST | dict(bwd_block_q_compute=1024, bwd_qtile_pipeline=True)),
      ("qtile1024_u2", _DQ_DK_FIRST | dict(bwd_block_q_compute=1024, bwd_kv_unroll=2)),
      ("qtile1024_u4", _DQ_DK_FIRST | dict(bwd_block_q_compute=1024, bwd_kv_unroll=4)),
      ("qtile1024_pipeline_u2", _DQ_DK_FIRST | dict(bwd_block_q_compute=1024, bwd_qtile_pipeline=True, bwd_kv_unroll=2)),
      ("qtile1024_pipeline_u4", _DQ_DK_FIRST | dict(bwd_block_q_compute=1024, bwd_qtile_pipeline=True, bwd_kv_unroll=4)),
      ("qtile512", _DQ_DK_FIRST | dict(bwd_block_q_compute=512)),
      ("qtile512_pipeline", _DQ_DK_FIRST | dict(bwd_block_q_compute=512, bwd_qtile_pipeline=True)),
      ("qtile512_u4", _DQ_DK_FIRST | dict(bwd_block_q_compute=512, bwd_kv_unroll=4)),
      ("qtile512_pipeline_u4", _DQ_DK_FIRST | dict(bwd_block_q_compute=512, bwd_qtile_pipeline=True, bwd_kv_unroll=4)),
      ("qtile2048_c512", _DQ_DK_FIRST | dict(bwd_block_q_compute=2048, block_kv_dkv_compute=512)),
      ("qtile2048_c512_pipeline", _DQ_DK_FIRST | dict(bwd_block_q_compute=2048, block_kv_dkv_compute=512, bwd_qtile_pipeline=True)),
      ("seqminor", _EXACT_LAYOUT),
      ("dk_first", _EXACT_LAYOUT | dict(bwd_dq_first=False)),
      ("dv_last", _EXACT_LAYOUT | dict(bwd_dv_last=True)),
      ("dp_early", _EXACT_LAYOUT | dict(bwd_dp_before_qk=True)),
      ("dq_transposed", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True)),
      ("dq_transposed_dk_first", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_dq_first=False)),
      ("dq_dk_first_dv_last", _DQ_DK_FIRST | dict(bwd_dv_last=True)),
      ("dq_dk_first_dv_middle", _DQ_DK_FIRST | dict(bwd_dv_between_dq_dk=True)),
      ("dq_dq_first_dv_middle", _DQ_DK_FIRST | dict(bwd_dq_first=True, bwd_dv_between_dq_dk=True)),
      ("dq_dk_first_dp_early", _DQ_DK_FIRST | dict(bwd_dp_before_qk=True)),
      ("dq_dk_first_dp_early_dv_last", _DQ_DK_FIRST | dict(bwd_dp_before_qk=True, bwd_dv_last=True)),
      ("dq_dk_first_dp_early_dv_middle", _DQ_DK_FIRST | dict(bwd_dp_before_qk=True, bwd_dv_between_dq_dk=True)),
      ("dq_dk_first_all_trans_dv_last", _DQ_DK_FIRST | dict(bwd_dv_last=True, bwd_dk_transposed_output=True, bwd_dv_transposed_output=True)),
      ("dq_dk_first_all_trans_dv_middle", _DQ_DK_FIRST | dict(bwd_dv_between_dq_dk=True, bwd_dk_transposed_output=True, bwd_dv_transposed_output=True)),
      ("dq_dk_first_single", _DQ_DK_FIRST | dict(bwd_single_segment_mask_body=True)),
      ("dq_dk_first_single_u2", _DQ_DK_FIRST | dict(bwd_single_segment_mask_body=True, bwd_kv_unroll=2)),
      ("dq_dk_first_single_u4", _DQ_DK_FIRST | dict(bwd_single_segment_mask_body=True, bwd_kv_unroll=4)),
      ("dq_dk_first_q2048", _DQ_DK_FIRST | dict(block_q_dkv=2048)),
      ("dq_dk_first_c512", _DQ_DK_FIRST | dict(block_kv_dkv_compute=512)),
      ("dq_dk_first_transpose_dk", _DQ_DK_FIRST | dict(bwd_dk_transposed_output=True)),
      ("dq_dk_first_transpose_dv", _DQ_DK_FIRST | dict(bwd_dv_transposed_output=True)),
      ("dq_dk_first_transpose_all", _DQ_DK_FIRST | dict(bwd_dk_transposed_output=True, bwd_dv_transposed_output=True)),
      ("dq_dk_first_transpose_all_single_u2", _DQ_DK_FIRST | dict(bwd_dk_transposed_output=True, bwd_dv_transposed_output=True, bwd_single_segment_mask_body=True, bwd_kv_unroll=2)),
      ("qmajor", _EXACT_LAYOUT | dict(bwd_qmajor_probabilities=True)),
      ("qmajor_dk_first", _EXACT_LAYOUT | dict(bwd_qmajor_probabilities=True, bwd_dq_first=False)),
      ("qmajor_dqt", _EXACT_LAYOUT | dict(bwd_qmajor_probabilities=True, bwd_dq_transposed_output=True)),
      ("qmajor_dqt_dk_first", _DQ_DK_FIRST | dict(bwd_qmajor_probabilities=True)),
      ("qmajor_dqt_dk_first_dp_early", _DQ_DK_FIRST | dict(bwd_qmajor_probabilities=True, bwd_dp_before_qk=True)),
      ("qmajor_dqt_dk_first_c512", _DQ_DK_FIRST | dict(bwd_qmajor_probabilities=True, block_kv_dkv_compute=512)),
      ("qmajor_dqt_dk_first_q2048", _DQ_DK_FIRST | dict(bwd_qmajor_probabilities=True, block_q_dkv=2048)),
      ("dq_transposed_dv_last", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_dv_last=True)),
      ("dq_transposed_dp_early", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_dp_before_qk=True)),
      ("dq_transposed_scheduler", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_scheduler=True)),
      ("dq_transposed_single_body", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_single_segment_mask_body=True)),
      ("dq_transposed_u2", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_kv_unroll=2)),
      ("dq_transposed_u4", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_kv_unroll=4)),
      ("dq_transposed_single_u2", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_single_segment_mask_body=True, bwd_kv_unroll=2)),
      ("dq_transposed_single_u4", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, bwd_single_segment_mask_body=True, bwd_kv_unroll=4)),
      ("dq_transposed_c512_u2", _EXACT_LAYOUT | dict(bwd_dq_transposed_output=True, block_kv_dkv_compute=512, bwd_kv_unroll=2)),
      ("pipeline_c512", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_staged_kv_pipeline=True, block_kv_dkv_compute=512)),
      ("pipeline_c256", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_staged_kv_pipeline=True, block_kv_dkv_compute=256)),
      ("pipeline_q2048_c512", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_staged_kv_pipeline=True, block_q_dkv=2048, block_kv_dkv_compute=512)),
      ("pipeline_c512_scheduler", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_staged_kv_pipeline=True, block_kv_dkv_compute=512, bwd_scheduler=True)),
      ("pipeline_c1024", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_staged_kv_pipeline=True)),
      ("single_mask_body", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True)),
      ("single_mask_body_u2", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_kv_unroll=2)),
      ("single_mask_body_u4", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_kv_unroll=4)),
      ("single_mask_body_u8", _EXACT_LAYOUT | dict(bwd_single_segment_mask_body=True, bwd_kv_unroll=8)),
      ("unroll2", _EXACT_LAYOUT | dict(bwd_kv_unroll=2)),
      ("unroll4", _EXACT_LAYOUT | dict(bwd_kv_unroll=4)),
      ("compute512", _EXACT_LAYOUT | dict(block_kv_dkv_compute=512)),
      ("compute512_u2", _EXACT_LAYOUT | dict(block_kv_dkv_compute=512, bwd_kv_unroll=2)),
      ("q2048", _EXACT_LAYOUT | dict(block_q_dkv=2048)),
      ("q2048_u2", _EXACT_LAYOUT | dict(block_q_dkv=2048, bwd_kv_unroll=2)),
      ("q2048_c512_u2", _EXACT_LAYOUT | dict(block_q_dkv=2048, block_kv_dkv_compute=512, bwd_kv_unroll=2)),
      ("scheduler", _EXACT_LAYOUT | dict(bwd_scheduler=True)),
      ("compute512_scheduler", _EXACT_LAYOUT | dict(block_kv_dkv_compute=512, bwd_scheduler=True)),
      ("dp_early_dv_late", _EXACT_LAYOUT | dict(bwd_dp_before_qk=True, bwd_dv_last=True)),
  ]


@jax.jit
def _error_statistics(actual, expected):
  a, b = actual.astype(jnp.float32), expected.astype(jnp.float32)
  difference = a - b
  bits_dtype = jnp.uint16 if actual.dtype == jnp.bfloat16 else jnp.uint32
  actual_bits = jax.lax.bitcast_convert_type(actual, bits_dtype)
  expected_bits = jax.lax.bitcast_convert_type(expected, bits_dtype)
  return (
      jnp.sum(actual_bits != expected_bits, dtype=jnp.int32),
      jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)),
      jnp.max(jnp.abs(difference)),
      jnp.sqrt(jnp.sum(difference * difference)
               / jnp.maximum(jnp.sum(b * b), jnp.float32(1e-30))),
  )


def precision_statistics(actual, expected, names=("dq", "dk", "dv")):
  if len(actual) != len(expected) or len(actual) != len(names):
    raise ValueError("Precision inputs and names must have matching lengths")
  report = {}
  for name, value, reference in zip(names, actual, expected):
    if value.shape != reference.shape:
      raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {reference.shape}")
    if value.dtype not in (jnp.bfloat16, jnp.float32) or value.dtype != reference.dtype:
      raise ValueError("The scheduling screen expects matching BF16/F32 arrays")
    mismatches, finite, max_abs, rel_l2 = bench._ready(
        _error_statistics(value, reference)
    )
    report[name] = dict(
        bitwise_equal=int(mismatches) == 0,
        mismatch_count=int(mismatches), elements=value.size,
        finite=bool(finite), max_abs=float(max_abs), relative_l2=float(rel_l2),
    )
  return report


def oracle_statistics(values, oracles, *, phase):
  """Compare sampled full-length heads to an independent FP32 computation."""
  indices = (2, 3, 4) if phase == "backward" else (0, 1, 2, 3, 4)
  names = ("dq", "dk", "dv") if phase == "backward" else ("output", "logsumexp", "dq", "dk", "dv")
  return {
      str(head): {
          name: accuracy.accuracy_statistics(value[head], oracle[index])
          for name, value, index in zip(names, values, indices)
      }
      for head, oracle in oracles.items()
  }


def joint_values(kernel, q, k, v, ids, do):
  """Public custom-VJP composition, not summed independent timings.

  Return only the public output and three gradients. Residual LSE is checked
  separately outside timing, avoiding extra forward work in this function.
  This is an operator composition, not a model/remat simulation.
  """
  output, pullback = jax.vjp(lambda q, k, v: kernel(q, k, v, ids), q, k, v)
  return output, *pullback(do)


def _config(args):
  return splash.SplashConfig(
      block_q=min(1024, args.sequence // 2),
      block_kv=min(8192, args.sequence),
      block_kv_compute=min(256, args.sequence // 2),
      block_q_dkv=min(4096, args.sequence // 2),
      block_kv_dkv=min(8192, args.sequence),
      block_kv_dkv_compute=min(1024, args.sequence // 2),
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=args.head_dim**-0.5,
      max_logit_const=0.0, use_base2_exp=True, combine_log2_scale=True,
      bwd_kv_unroll=False, bwd_dq_first=True, bwd_cast_before_transpose=True,
      bwd_scale_after_dot=True, compact_stats_output=True,
      omit_unused_max_logits=True, segment_mask_on_partial_only=True,
      fwd_vmem_limit_bytes=60 * 1024**2,
      bwd_vmem_limit_bytes=63 * 1024**2,
      bwd_parallel_heads=False, bwd_scheduler=False,
      region_trace_mode=args.region_trace_mode, interpret=args.interpret,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("forward", "backward", "joint"), default="backward")
  parser.add_argument("--sequence", type=int, default=32768)
  parser.add_argument("--heads", type=int, default=32, help="Merged batch*heads")
  parser.add_argument("--head-dim", type=int, default=72)
  parser.add_argument("--seed", type=int, default=27)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeats", type=int, default=12)
  parser.add_argument("--variants", nargs="*")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--profile-variants", nargs="*", default=[])
  parser.add_argument("--profile-repeats", type=int, default=3)
  parser.add_argument("--oracle-heads", type=int, nargs="*", default=[],
                      help="Untimed independent FP32 oracle on these full-length merged heads")
  parser.add_argument("--oracle-block-q", type=int, default=512)
  parser.add_argument("--region-trace-mode", choices=("none", "coarse", "fine"), default="none")
  parser.add_argument("--interpret", action="store_true")
  args = parser.parse_args()
  if args.sequence < 256 or args.sequence & (args.sequence - 1):
    raise ValueError("sequence must be a power of two >= 256")
  if any(head < 0 or head >= args.heads for head in args.oracle_heads):
    raise ValueError("oracle-heads must index the merged head dimension")
  available = dict(variants(args.phase))
  selected = args.variants or list(available)
  if set(selected) - available.keys():
    raise ValueError(f"Unknown variants: {set(selected) - available.keys()}")
  out = Path(args.output_dir)
  (out / "benchmark").mkdir(parents=True, exist_ok=True)
  ids_np = np.repeat(
      np.array([1, 2, 3, 0], np.int32),
      [args.sequence // 2, args.sequence // 2 - 16, 8, 8],
  )
  ids = base.SegmentIds(jnp.asarray(ids_np), jnp.asarray(ids_np))
  cfg = _config(args)
  reference_kernel = bench._make_kernel(ids, cfg)
  shape = (args.heads, args.sequence, args.head_dim)
  q, k, v, do = [
      jax.random.normal(key, shape, jnp.bfloat16)
      for key in jax.random.split(jax.random.key(args.seed), 4)
  ]
  reference_forward = jax.jit(
      lambda q, k, v, ids: bench._forward(reference_kernel, q, k, v, ids)
  ).lower(q, k, v, ids).compile()
  forward_output, residuals = bench._ready(reference_forward(q, k, v, ids))
  reference_backward = jax.jit(
      lambda res, do: bench._backward(reference_kernel, res, do)
  ).lower(residuals, do).compile()
  reference_grads = bench._ready(reference_backward(residuals, do))
  reference_lse = residuals[6]
  reference_joint = None
  if args.phase == "joint":
    reference_joint = jax.jit(functools.partial(joint_values, reference_kernel)).lower(
        q, k, v, ids, do
    ).compile()
    joint_reference_values = bench._ready(reference_joint(q, k, v, ids, do))
    forward_output, *reference_grads = joint_reference_values
  oracles = {}
  oracle_fn = jax.jit(functools.partial(
      accuracy.fp32_attention_and_gradients,
      block_q=min(args.oracle_block_q, args.sequence),
  ))
  for head in args.oracle_heads:
    oracles[head] = bench._ready(oracle_fn(q[head], k[head], v[head], do[head], ids.q, ids.kv))
    accuracy.require_finite_oracle(oracles[head], head=head)
    print(json.dumps(dict(status="oracle_head_complete", head=head)), flush=True)
  reference_values = (
      reference_grads if args.phase == "backward"
      else (forward_output, reference_lse / splash.LOG2E, *reference_grads)
  )
  reference_oracle = oracle_statistics(reference_values, oracles, phase=args.phase)
  forward_timing = bench._summary(bench._measure(
      reference_forward, (q, k, v, ids), args.warmup, args.repeats
  ))

  metrics_path = out / "benchmark/metrics.jsonl"
  details_path = out / "benchmark/schedule-details.jsonl"
  with metrics_path.open("w") as metrics, details_path.open("w") as details:
    for name in selected:
      overrides = available[name].copy()
      # Small CPU smoke tests retain legal blocks without altering production.
      if args.interpret:
        for field in ("block_q", "block_kv", "block_kv_compute", "block_q_dkv", "block_kv_dkv", "block_kv_dkv_compute", "bwd_block_q_compute"):
          if field in overrides:
            overrides[field] = min(overrides[field], args.sequence // 2)
      candidate_cfg = dataclasses.replace(cfg, **overrides)
      row = dict(variant=name, phase=args.phase, seed=args.seed, merged_heads=args.heads,
                 seq_len=args.sequence, head_dim=args.head_dim, dtype="bfloat16",
                 config=dataclasses.asdict(candidate_cfg),
                 reference_forward=forward_timing, device=str(jax.devices()[0]))
      candidate = None
      try:
        kernel = bench._make_kernel(ids, candidate_cfg)
        # The benchmark helper sets dq_reduction_steps; record what executes.
        row["config"] = dataclasses.asdict(kernel.kwargs["config"])
        row["compile_started_ns"] = time.time_ns()
        started = time.perf_counter()
        if args.phase == "backward":
          candidate_residuals = (*residuals[:-1], kernel.dkv_mask_info)
          candidate_args = (candidate_residuals, do)
          candidate = jax.jit(
              lambda res, do: bench._backward(kernel, res, do)
          ).lower(*candidate_args).compile()
        elif args.phase == "joint":
          candidate_args = (q, k, v, ids, do)
          candidate = jax.jit(functools.partial(joint_values, kernel)).lower(
              *candidate_args
          ).compile()
          candidate_lse_check = jax.jit(
              lambda q, k, v, ids: bench._forward(kernel, q, k, v, ids)
          ).lower(q, k, v, ids).compile()
          row["joint_lse_source"] = "untimed_standalone_forward"
        else:
          candidate_args = (q, k, v, ids)
          candidate = jax.jit(
              lambda q, k, v, ids: bench._forward(kernel, q, k, v, ids)
          ).lower(*candidate_args).compile()
        row["compile_seconds"] = time.perf_counter() - started
        row["compile_finished_ns"] = time.time_ns()
        if args.phase == "backward":
          candidate_grads = bench._ready(candidate(*candidate_args))
          output_precision = {}
        else:
          if args.phase == "joint":
            candidate_output, *candidate_grads = bench._ready(candidate(*candidate_args))
            _, checked_residuals = bench._ready(candidate_lse_check(q, k, v, ids))
            candidate_lse = checked_residuals[6]
            del checked_residuals
          else:
            candidate_output, candidate_residuals = bench._ready(candidate(*candidate_args))
            candidate_lse = candidate_residuals[6]
            candidate_grads = bench._ready(reference_backward(candidate_residuals, do))
          output_precision = precision_statistics(
              (candidate_output, candidate_lse),
              (forward_output, reference_lse), names=("output", "logsumexp"),
          )
        row["precision"] = precision_statistics(candidate_grads, reference_grads)
        row["precision"].update(output_precision)
        if oracles:
          candidate_values = (
              candidate_grads if args.phase == "backward"
              else (candidate_output, candidate_lse / splash.LOG2E, *candidate_grads)
          )
          row["fp32_oracle"] = dict(
              kind="independent_full_length_fp32_highest_precision",
              heads=args.oracle_heads, block_q=min(args.oracle_block_q, args.sequence),
              reference=reference_oracle,
              candidate=oracle_statistics(candidate_values, oracles, phase=args.phase),
              decision="diagnostic_only_no_automatic_acceptance",
          )
          del candidate_values
        if args.phase != "backward":
          del candidate_output
        if args.phase == "joint":
          del candidate_lse_check
        del candidate_grads
        exact = all(x["bitwise_equal"] and x["finite"] for x in row["precision"].values())
        row["numerical_status"] = "bitwise_equal" if exact else "needs_accuracy_review"
        row[args.phase] = bench._summary(bench._measure(
            candidate, candidate_args, args.warmup, args.repeats
        ))
        row["status"] = "measured"
        if name in args.profile_variants:
          profile_dir = out / "profiling/xprof" / name
          profile_dir.mkdir(parents=True, exist_ok=True)
          with jax.profiler.trace(str(profile_dir)):
            for step in range(args.profile_repeats):
              with jax.profiler.StepTraceAnnotation("vit_splash_" + args.phase, step_num=step):
                bench._ready(candidate(*candidate_args))
        metric = dict(variant=name, phase=args.phase, latency_ms=row[args.phase]["median_ms"],
                      seq_len=args.sequence, merged_heads=args.heads, head_dim=args.head_dim,
                      dtype="bfloat16", seed=args.seed, bitwise_equal=exact)
        metrics.write(json.dumps(metric, sort_keys=True) + "\n")
        metrics.flush()
      except Exception as error:
        if "compile_started_ns" in row and "compile_finished_ns" not in row:
          row["compile_finished_ns"] = time.time_ns()
        message = str(error)
        if len(message) > 5000:
          message = message[:2500] + "\n[... omitted ...]\n" + message[-2500:]
        row.update(status="error", error_type=type(error).__name__, error=message)
      # Recheck the live reference to expose drift during a long compile sweep.
      if args.phase == "joint":
        reference_fn, reference_args = reference_joint, (q, k, v, ids, do)
      else:
        reference_fn = reference_backward if args.phase == "backward" else reference_forward
        reference_args = (residuals, do) if args.phase == "backward" else (q, k, v, ids)
      row["reference_" + args.phase] = bench._summary(bench._measure(
          reference_fn, reference_args, 1, max(3, args.repeats // 2)
      ))
      if row["status"] == "measured":
        row["speedup"] = row["reference_" + args.phase]["median_ms"] / row[args.phase]["median_ms"]
      details.write(json.dumps(row, default=str, sort_keys=True) + "\n")
      details.flush()
      print(json.dumps(row, default=str, sort_keys=True), flush=True)
      del candidate
      jax.clear_caches()
      gc.collect()


if __name__ == "__main__":
  main()
