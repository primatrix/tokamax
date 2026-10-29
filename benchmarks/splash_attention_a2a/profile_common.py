"""Shared low-volume XPlane capture helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def set_profile_flags(out: Path):
  import os
  os.environ["JAX_PLATFORMS"] = "tpu"
  os.environ["LIBTPU_INIT_ARGS"] = " ".join([
      "--xla_tpu_dvfs_p_state=7",
      "--xla_enable_custom_call_region_trace=false",
      "--xla_xprof_enable_custom_call_tracing=false",
      "--xla_xprof_register_llo_debug_info=false",
      "--xla_tpu_enable_llo_profiling=false",
  ])
  os.environ["XLA_FLAGS"] = " ".join([
      "--xla_dump_hlo_as_text",
      f"--xla_dump_to={out / 'hlo'}",
  ])
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def capture_one(fn, args, out: Path, label: str):
  import jax
  out.mkdir(parents=True, exist_ok=False)
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 0
  options.host_tracer_level = 2
  options.advanced_configuration = {
      "tpu_trace_mode": "TRACE_ONLY_XLA",
      "num_tensor_cores_to_trace_per_device": 1,
      "tpu_enable_periodic_counter_sampling": False,
  }
  with jax.profiler.trace(str(out / "trace"), profiler_options=options):
    with jax.profiler.StepTraceAnnotation(label, step_num=0):
      result = fn(*args)
      jax.block_until_ready(result)
  traces = sorted((out / "trace").rglob("*.xplane.pb"))
  if not traces:
    raise RuntimeError(f"no XPlane files in {out}")
  (out / "traces.json").write_text(json.dumps([str(x) for x in traces], indent=2) + "\n")
  return result


def array_sha(value):
  import numpy as np
  import jax
  rows = []
  for leaf in jax.tree_util.tree_leaves(value):
    for shard in getattr(leaf, "addressable_shards", ()):
      # Gradient shards can be strided views; hash the row-major bytes. This
      # is a no-op for contiguous arrays, so earlier hashes stay comparable.
      raw = np.ascontiguousarray(np.asarray(shard.data)).view(np.uint8)
      rows.append({"device": shard.device.id, "index": str(shard.index),
                   "sha256": hashlib.sha256(raw).hexdigest()})
  return rows
