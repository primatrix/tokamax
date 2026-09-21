# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Full-shape BF16 Splash scheduling screen with explicit precision evidence.

Non-bitwise candidates are measured for diagnosis but are never marked accepted.
Backward tiling changes replace the residual MaskInfo, not just the config.
Forward activations and residual statistics are shared unchanged in this sweep.
"""

import argparse
import dataclasses
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


_EXACT_LAYOUT = dict(
    bwd_dq_scratch_seq_minor=True,
    bwd_dkv_scratch_seq_minor=True,
    bwd_dkv_output_seq_minor=True,
    bwd_fuse_segment_id_inputs=True,
    bwd_do_seq_minor=True,
)


def variants():
  return [
      ("pr13", {}),
      ("seqminor", _EXACT_LAYOUT),
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
  actual_bits = jax.lax.bitcast_convert_type(actual, jnp.uint16)
  expected_bits = jax.lax.bitcast_convert_type(expected, jnp.uint16)
  return (
      jnp.sum(actual_bits != expected_bits, dtype=jnp.int32),
      jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)),
      jnp.max(jnp.abs(difference)),
      jnp.sqrt(jnp.sum(difference * difference)
               / jnp.maximum(jnp.sum(b * b), jnp.float32(1e-30))),
  )


def precision_statistics(actual, expected):
  report = {}
  for name, value, reference in zip(("dq", "dk", "dv"), actual, expected):
    if value.dtype != jnp.bfloat16 or reference.dtype != jnp.bfloat16:
      raise ValueError("The scheduling screen expects BF16 gradient outputs")
    mismatches, finite, max_abs, rel_l2 = bench._ready(
        _error_statistics(value, reference)
    )
    report[name] = dict(
        bitwise_equal=int(mismatches) == 0,
        mismatch_count=int(mismatches), elements=value.size,
        finite=bool(finite), max_abs=float(max_abs), relative_l2=float(rel_l2),
    )
  return report


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
  parser.add_argument("--region-trace-mode", choices=("none", "coarse", "fine"), default="none")
  parser.add_argument("--interpret", action="store_true")
  args = parser.parse_args()
  if args.sequence < 256 or args.sequence & (args.sequence - 1):
    raise ValueError("sequence must be a power of two >= 256")
  available = dict(variants())
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
  del forward_output
  reference_backward = jax.jit(
      lambda res, do: bench._backward(reference_kernel, res, do)
  ).lower(residuals, do).compile()
  reference_grads = bench._ready(reference_backward(residuals, do))
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
        for field in ("block_q_dkv", "block_kv_dkv_compute"):
          if field in overrides:
            overrides[field] = min(overrides[field], args.sequence // 2)
      candidate_cfg = dataclasses.replace(cfg, **overrides)
      row = dict(variant=name, seed=args.seed, merged_heads=args.heads,
                 seq_len=args.sequence, head_dim=args.head_dim, dtype="bfloat16",
                 config=dataclasses.asdict(candidate_cfg),
                 reference_forward=forward_timing, device=str(jax.devices()[0]))
      candidate = None
      try:
        kernel = bench._make_kernel(ids, candidate_cfg)
        candidate_residuals = (*residuals[:-1], kernel.dkv_mask_info)
        started = time.perf_counter()
        candidate = jax.jit(
            lambda res, do: bench._backward(kernel, res, do)
        ).lower(candidate_residuals, do).compile()
        row["compile_seconds"] = time.perf_counter() - started
        candidate_grads = bench._ready(candidate(candidate_residuals, do))
        row["precision"] = precision_statistics(candidate_grads, reference_grads)
        del candidate_grads
        exact = all(x["bitwise_equal"] and x["finite"] for x in row["precision"].values())
        row["numerical_status"] = "bitwise_equal" if exact else "needs_accuracy_review"
        row["backward"] = bench._summary(bench._measure(
            candidate, (candidate_residuals, do), args.warmup, args.repeats
        ))
        row["status"] = "measured"
        if name in args.profile_variants:
          profile_dir = out / "profiling/xprof" / name
          profile_dir.mkdir(parents=True, exist_ok=True)
          with jax.profiler.trace(str(profile_dir)):
            for step in range(args.profile_repeats):
              with jax.profiler.StepTraceAnnotation("vit_splash_bwd", step_num=step):
                bench._ready(candidate(candidate_residuals, do))
        metric = dict(variant=name, phase="backward", latency_ms=row["backward"]["median_ms"],
                      seq_len=args.sequence, merged_heads=args.heads, head_dim=args.head_dim,
                      dtype="bfloat16", seed=args.seed, bitwise_equal=exact)
        metrics.write(json.dumps(metric, sort_keys=True) + "\n")
        metrics.flush()
      except Exception as error:
        row.update(status="error", error_type=type(error).__name__, error=str(error)[-5000:])
      # Recheck the live reference to expose drift during a long compile sweep.
      row["reference_backward"] = bench._summary(bench._measure(
          reference_backward, (residuals, do), 1, max(3, args.repeats // 2)
      ))
      if row["status"] == "measured":
        row["speedup"] = row["reference_backward"]["median_ms"] / row["backward"]["median_ms"]
      details.write(json.dumps(row, default=str, sort_keys=True) + "\n")
      details.flush()
      print(json.dumps(row, default=str, sort_keys=True), flush=True)
      del candidate
      jax.clear_caches()
      gc.collect()


if __name__ == "__main__":
  main()
