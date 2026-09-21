# Copyright 2026 Primatrix Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Single-device PR13 ViT Splash forward/backward benchmark.

The defaults are the strongest measured PR13 tiling from the MaxText ViT
full-remat sweep.  Forward and backward are compiled and timed separately;
backward consumes already-materialized custom-VJP residuals, so it excludes
the rematerialized forward pass.
"""

import argparse
import dataclasses
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as mask_info_lib
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


def _segment_mask_info(q_ids, kv_ids, block_q, block_kv, *, is_dkv=False):
  """Matches MaxText's conservative runtime segment metadata."""
  q = q_ids.reshape(-1, block_q)
  kv = kv_ids.reshape(-1, block_kv)

  def bounds(ids):
    nonzero = ids != 0
    low = jnp.min(
        jnp.where(nonzero, ids, jnp.iinfo(jnp.int32).max), axis=-1
    )
    high = jnp.max(
        jnp.where(nonzero, ids, jnp.iinfo(jnp.int32).min), axis=-1
    )
    return low, high, jnp.any(nonzero, axis=-1), jnp.any(~nonzero, axis=-1)

  qlo, qhi, qnz, qzero = bounds(q)
  klo, khi, knz, kzero = bounds(kv)
  active = (
      (qlo[:, None] <= khi[None, :])
      & (klo[None, :] <= qhi[:, None])
      & qnz[:, None]
      & knz[None, :]
  ) | (qzero[:, None] & kzero[None, :])
  full = (
      jnp.all(q == q[:, :1], axis=1)[:, None]
      & jnp.all(kv == kv[:, :1], axis=1)[None, :]
      & (q[:, :1] == kv[:, 0][None, :])
  )
  if is_dkv:
    active = active.T
    full = full.T
    active = active.at[:, 0].set(active[:, 0] | ~active.any(axis=1))
  indices = jnp.argwhere(active, size=active.size, fill_value=-1)
  count = active.sum(dtype=jnp.int32).reshape(1)
  kind = jnp.where(full[indices[:, 0], indices[:, 1]], 2, 1)
  block_mask = jnp.where(jnp.arange(active.size) < count, kind, 0).astype(
      jnp.int8
  )
  return mask_info_lib.MaskInfo(
      mask_next=jnp.full((active.size,), -1, dtype=jnp.int8),
      active_rows=indices[:, 0].astype(jnp.int32),
      active_cols=indices[:, 1].astype(jnp.int32),
      block_mask=block_mask,
      num_active_blocks=count,
      partial_mask_blocks=None,
      q_sequence=None,
  )


def _make_kernel(ids, config):
  config = dataclasses.replace(config, dq_reduction_steps=3)
  fwd = _segment_mask_info(
      ids.q, ids.kv, config.block_q, config.block_kv
  )
  bwd = _segment_mask_info(
      ids.q,
      ids.kv,
      config.block_q_dkv,
      config.block_kv_dkv,
      is_dkv=True,
  )
  return splash.SplashAttentionKernel(
      fwd,
      bwd,
      config=config,
      is_mqa=False,
      save_residuals=False,
      mask_value=base.DEFAULT_MASK_VALUE,
      mask_function=None,
      fwd_mask_sparsity=1.0,
      dkv_mask_sparsity=1.0,
  )


def _forward(kernel, q, k, v, ids):
  kw = kernel.kwargs
  return splash._splash_attention_fwd(  # pylint: disable=protected-access
      kernel.fwd_mask_info,
      kernel.dkv_mask_info,
      q,
      k,
      v,
      ids,
      None,
      kw["save_residuals"],
      kw["mask_value"],
      kw["is_mqa"],
      kw["config"],
      kw["mask_function"],
      kw["fwd_mask_sparsity"],
      kw["dkv_mask_sparsity"],
  )


def _backward(kernel, residuals, do):
  kw = kernel.kwargs
  result = splash._splash_attention_bwd(  # pylint: disable=protected-access
      kw["save_residuals"],
      kw["mask_value"],
      kw["is_mqa"],
      kw["config"],
      kw["mask_function"],
      kw["fwd_mask_sparsity"],
      kw["dkv_mask_sparsity"],
      residuals,
      do,
  )
  return result[2:5]


def _ready(tree):
  return jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
      tree,
  )


def _measure(fn, args, warmup, repeats):
  for _ in range(warmup):
    _ready(fn(*args))
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    _ready(fn(*args))
    samples.append((time.perf_counter() - start) * 1e3)
  return samples


def _summary(samples):
  return {
      "median_ms": statistics.median(samples),
      "min_ms": min(samples),
      "max_ms": max(samples),
      "samples_ms": samples,
  }


def _parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--heads", type=int, default=16)
  parser.add_argument("--sequence", type=int, default=32768)
  parser.add_argument("--head-dim", type=int, default=72)
  parser.add_argument(
      "--segment-lengths", type=int, nargs="+", default=(16384, 16368, 8, 8)
  )
  parser.add_argument("--block-q", type=int, default=1024)
  parser.add_argument("--block-kv", type=int, default=8192)
  parser.add_argument("--block-kv-compute", type=int, default=256)
  parser.add_argument("--block-q-dkv", type=int, default=4096)
  parser.add_argument("--block-kv-dkv", type=int, default=8192)
  parser.add_argument("--block-kv-dkv-compute", type=int, default=1024)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeats", type=int, default=10)
  parser.add_argument("--seed", type=int, default=27)
  parser.add_argument("--variant", default="pr13_best_tiling")
  parser.add_argument("--output")
  parser.add_argument("--interpret", action="store_true")
  parser.add_argument("--split-major-segments", action="store_true")
  parser.add_argument("--bwd-dq-contract-ds-axis0", action="store_true")
  parser.add_argument("--bwd-keep-kv-seq-minor", action="store_true")
  parser.add_argument("--bwd-dp-before-qk", action="store_true")
  parser.add_argument("--bwd-head-group-size", type=int, default=1)
  parser.add_argument("--use-base2-exp", action="store_true")
  parser.add_argument("--bwd-reuse-bf16-probabilities", action="store_true")
  parser.add_argument("--bwd-compact-segment-ids", action="store_true")
  parser.add_argument("--bwd-dkv-scratch-seq-minor", action="store_true")
  parser.add_argument("--bwd-dkv-output-seq-minor", action="store_true")
  parser.add_argument("--bwd-fuse-segment-id-inputs", action="store_true")
  parser.add_argument(
      "--bwd-parallel-heads", action=argparse.BooleanOptionalAction, default=False
  )
  parser.add_argument(
      "--bwd-scheduler", action=argparse.BooleanOptionalAction, default=False
  )
  return parser


def main():
  args = _parser().parse_args()
  if sum(args.segment_lengths) != args.sequence:
    raise ValueError("segment lengths must sum to sequence")
  ids_np = np.concatenate(
      [
          *(np.full(n, i, np.int32) for i, n in enumerate(args.segment_lengths[:-1], 1)),
          np.zeros(args.segment_lengths[-1], np.int32),
      ]
  )
  ids = base.SegmentIds(jnp.asarray(ids_np), jnp.asarray(ids_np))
  config = splash.SplashConfig(
      block_q=args.block_q,
      block_kv=args.block_kv,
      block_kv_compute=args.block_kv_compute,
      block_q_dkv=args.block_q_dkv,
      block_kv_dkv=args.block_kv_dkv,
      block_kv_dkv_compute=args.block_kv_dkv_compute,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=args.head_dim**-0.5,
      max_logit_const=0.0,
      use_base2_exp=args.use_base2_exp,
      combine_log2_scale=args.use_base2_exp,
      bwd_kv_unroll=False,
      bwd_dq_first=True,
      bwd_cast_before_transpose=True,
      bwd_scale_after_dot=True,
      compact_stats_output=True,
      omit_unused_max_logits=True,
      segment_mask_on_partial_only=True,
      fwd_vmem_limit_bytes=60 * 1024**2,
      bwd_vmem_limit_bytes=63 * 1024**2,
      bwd_parallel_heads=args.bwd_parallel_heads,
      bwd_scheduler=args.bwd_scheduler,
      bwd_dq_contract_ds_axis0=args.bwd_dq_contract_ds_axis0,
      bwd_keep_kv_seq_minor=args.bwd_keep_kv_seq_minor,
      bwd_dp_before_qk=args.bwd_dp_before_qk,
      bwd_head_group_size=args.bwd_head_group_size,
      bwd_reuse_bf16_probabilities=args.bwd_reuse_bf16_probabilities,
      bwd_compact_segment_ids=args.bwd_compact_segment_ids,
      bwd_dkv_scratch_seq_minor=args.bwd_dkv_scratch_seq_minor,
      bwd_dkv_output_seq_minor=args.bwd_dkv_output_seq_minor,
      bwd_fuse_segment_id_inputs=args.bwd_fuse_segment_id_inputs,
      interpret=args.interpret,
  )
  kernel = _make_kernel(ids, config)
  shape = (args.batch * args.heads, args.sequence, args.head_dim)
  keys = jax.random.split(jax.random.key(args.seed), 4)
  q, k, v = [jax.random.normal(key, shape, jnp.bfloat16) for key in keys[:3]]
  do = jax.random.normal(keys[3], shape, jnp.bfloat16)
  _ready((q, k, v, do, ids))

  if args.split_major_segments:
    if tuple(args.segment_lengths[:2]) != (
        args.sequence // 2,
        args.sequence // 2 - 16,
    ):
      raise ValueError("split-major-segments requires the production segment layout")
    half = args.sequence // 2
    first_kernel = splash.make_splash_mha_single_device(
        mask_lib.FullMask((half, half)), config=config
    )
    second_ids_np = np.concatenate(
        [
            np.ones(args.segment_lengths[1], np.int32),
            np.full(args.segment_lengths[2], 2, np.int32),
            np.zeros(args.segment_lengths[3], np.int32),
        ]
    )
    second_ids = base.SegmentIds(
        jnp.asarray(second_ids_np), jnp.asarray(second_ids_np)
    )
    second_kernel = _make_kernel(second_ids, config)

    def split_forward(q, k, v, unused_ids):
      del unused_ids
      first_out, first_res = _forward(
          first_kernel, q[:, :half], k[:, :half], v[:, :half], None
      )
      second_out, second_res = _forward(
          second_kernel,
          q[:, half:],
          k[:, half:],
          v[:, half:],
          second_ids,
      )
      return jnp.concatenate((first_out, second_out), axis=1), (
          first_res,
          second_res,
      )

    def split_backward(residuals, do):
      first = _backward(first_kernel, residuals[0], do[:, :half])
      second = _backward(second_kernel, residuals[1], do[:, half:])
      return tuple(
          jnp.concatenate((a, b), axis=1) for a, b in zip(first, second)
      )

    forward_fn = split_forward
    backward_fn = split_backward
  else:
    forward_fn = lambda q, k, v, ids: _forward(kernel, q, k, v, ids)
    backward_fn = lambda residuals, do: _backward(kernel, residuals, do)

  forward = jax.jit(forward_fn)
  compile_start = time.perf_counter()
  forward = forward.lower(q, k, v, ids).compile()
  forward_compile_s = time.perf_counter() - compile_start
  output, residuals = _ready(forward(q, k, v, ids))
  del output
  backward = jax.jit(backward_fn)
  compile_start = time.perf_counter()
  backward = backward.lower(residuals, do).compile()
  backward_compile_s = time.perf_counter() - compile_start
  _ready(backward(residuals, do))

  fwd = _summary(_measure(forward, (q, k, v, ids), args.warmup, args.repeats))
  bwd = _summary(_measure(backward, (residuals, do), args.warmup, args.repeats))
  combined_ms = fwd["median_ms"] + bwd["median_ms"]
  result = {
      "benchmark": "vit_splash_pr13_single_device",
      "variant": args.variant,
      "device": str(jax.devices()[0]),
      "shape": {
          "batch": args.batch,
          "heads": args.heads,
          "sequence": args.sequence,
          "head_dim": args.head_dim,
          "segment_lengths": args.segment_lengths,
      },
      "config": dataclasses.asdict(config),
      "compile_seconds": {
          "forward": forward_compile_s,
          "backward": backward_compile_s,
      },
      "forward": fwd,
      "backward": bwd,
      "combined_median_ms": combined_ms,
  }
  if args.output:
    with open(args.output, "w", encoding="utf-8") as output_file:
      for phase, latency in (
          ("forward", fwd["median_ms"]),
          ("backward", bwd["median_ms"]),
          ("combined", combined_ms),
      ):
        output_file.write(
            json.dumps(
                {
                    "variant": args.variant,
                    "phase": phase,
                    "latency_ms": latency,
                    "batch_size": args.batch,
                    "heads": args.heads,
                    "seq_len": args.sequence,
                    "head_dim": args.head_dim,
                    "dtype": "bfloat16",
                },
                sort_keys=True,
            )
            + "\n"
        )
  print(json.dumps(result, default=str, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
