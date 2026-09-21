# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0

"""Diagnose nonfinite compiled FP32 reference values; not a kernel benchmark."""

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy


@jax.jit
def _health(value):
  bad = ~jnp.isfinite(value)
  flat = bad.reshape(-1)
  return (jnp.sum(bad), jnp.sum(jnp.isnan(value)),
          jnp.max(jnp.where(bad, 0., jnp.abs(value))),
          jnp.nonzero(flat, size=8, fill_value=-1)[0])


def health(value):
  count, nan_count, max_abs, first_bad = _health(value)
  return dict(nonfinite_count=int(count), nan_count=int(nan_count),
              finite_max_abs=float(max_abs), first_bad_flat=np.asarray(first_bad).tolist())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--sequence", type=int, default=32768)
  parser.add_argument("--heads", type=int, default=32)
  args = parser.parse_args()
  out = Path(args.output_dir) / "benchmark"
  out.mkdir(parents=True, exist_ok=True)
  shape = (args.heads, args.sequence, 72)
  q, k, v, do = [jax.random.normal(key, shape, jnp.bfloat16)[0]
                 for key in jax.random.split(jax.random.key(27), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32),
                             [args.sequence // 2, args.sequence // 2 - 16, 8, 8]))
  variants = [("original", -jnp.inf, False), ("finite_mask", -1e30, False),
              ("barriers", -jnp.inf, True), ("finite_mask_barriers", -1e30, True)]
  with (out / "oracle-diagnostic.jsonl").open("w") as stream, (out / "metrics.jsonl").open("w") as metrics:
    for name, mask_value, barrier_stages in variants:
      row = dict(variant=name, seq_len=args.sequence, head=0, seed=27,
                 mask_value=str(mask_value), barrier_stages=barrier_stages)
      try:
        fn = jax.jit(functools.partial(accuracy.fp32_attention_and_gradients,
            block_q=min(512, args.sequence), mask_value=mask_value, barrier_stages=barrier_stages))
        values = fn(q, k, v, do, ids, ids)
        row["health"] = {key: health(value) for key, value in zip(
            ("output", "logsumexp", "dq", "dk", "dv"), values)}
        row["status"] = "finite" if all(x["nonfinite_count"] == 0 for x in row["health"].values()) else "nonfinite"
      except Exception as error:
        row.update(status="error", error=str(error)[:3000])
      stream.write(json.dumps(row) + "\n")
      stream.flush()
      if "health" in row:
        metrics.write(json.dumps(dict(variant=name, phase="oracle_health", seq_len=args.sequence,
            nonfinite_count=sum(value["nonfinite_count"] for value in row["health"].values()),
            status=row["status"])) + "\n")
        metrics.flush()
      print(json.dumps(row), flush=True)


if __name__ == "__main__":
  main()
