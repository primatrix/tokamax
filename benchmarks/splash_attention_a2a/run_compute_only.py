"""Run the compute-only diagnostic for the fused Splash structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--global-heads", type=int, default=64)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  from config import production_config
  from inputs import make_compute_only_arrays, make_segment_ids
  from profile_common import array_sha, capture_one, set_profile_flags
  import jax
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_compute_only_kernel,
  )

  set_profile_flags(args.out)
  config = production_config(global_heads=args.global_heads)
  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)
  if len(devices) != 8:
    raise RuntimeError(f"expected 8 TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (config["mesh_axis"],))
  fn, specs, metadata = splash_attention_compute_only_kernel.build(config, mesh)
  replicated = NamedSharding(mesh, P(None, None, None))
  ids_sharding = NamedSharding(mesh, P(None, None))
  compiled = fn.lower(*specs).compile()
  q, k, v = make_compute_only_arrays(jax, specs[:3], replicated)
  ids = make_segment_ids(config["tokens"], "multimodal")
  qids = jax.device_put(np.broadcast_to(ids[:, None],
                                        (config["tokens"], 128)), ids_sharding)
  kvids = jax.device_put(np.broadcast_to(ids[None, :],
                                         (8, config["tokens"])), ids_sharding)
  qseq = jax.device_put(np.broadcast_to(
      np.arange(config["tokens"], dtype=np.int32)[:, None],
      (config["tokens"], 128)), ids_sharding)
  args.out.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")
  args.out.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
  args.out.joinpath("input_ids.json").write_text(json.dumps({
      "mode": "multimodal",
      "sha256": __import__("hashlib").sha256(ids.tobytes()).hexdigest(),
  }, indent=2) + "\n")
  args.out.joinpath("compile_memory.txt").write_text(str(compiled.memory_analysis()))
  args_list = (q, k, v, qids, kvids, qseq)
  for _ in range(3):
    jax.block_until_ready(compiled(*args_list))
  result = capture_one(compiled, args_list, args.out / "profile",
                       "SPLASH512_COMPUTE_ONLY")
  args.out.joinpath("output_sha.json").write_text(
      json.dumps(array_sha(result), indent=2) + "\n")


if __name__ == "__main__":
  main()
