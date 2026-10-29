"""Run one-head direct Splash compute with no A2A and no nested pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--global-heads", type=int, default=8)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)

  from attention_core import build as build_core
  from config import production_config
  from inputs import make_compute_only_arrays, make_segment_ids
  from profile_common import array_sha, capture_one, set_profile_flags

  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

  config = production_config(global_heads=args.global_heads)
  if config["heads"] != 1:
    raise ValueError("direct compute probe requires one local head")
  set_profile_flags(args.out)
  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)
  if len(devices) != config["cp_size"]:
    raise RuntimeError(f"expected {config['cp_size']} TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (config["mesh_axis"],))
  core, specs, metadata = build_core(config)
  in_specs = (P(None, None, None),) * 3 + (P(None),)
  out_specs = (
      P(None, None, None),
      {"logsumexp": P(None, None, None),
       "max_logits": P(None, None, None)},
  )

  @jax.shard_map(mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                 check_vma=False)
  def mapped(q, k, v, ids):
    return core(q, k, v, ids, ids)

  replicated = NamedSharding(mesh, P(None, None, None))
  ids_sharding = NamedSharding(mesh, P(None))
  # ``mapped`` accepts one runtime segment-id vector and reuses it for both
  # q/k metadata operands; lower only the four arguments that it exposes.
  compiled = jax.jit(mapped).lower(*specs[:4]).compile()
  q, k, v = make_compute_only_arrays(jax, specs[:3], replicated)
  ids = jax.device_put(
      make_segment_ids(config["tokens"], "multimodal"), ids_sharding)
  args_list = (q, k, v, ids)
  args.out.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")
  args.out.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
  args.out.joinpath("input_ids.json").write_text(json.dumps({
      "mode": "multimodal",
      "sha256": hashlib.sha256(np.asarray(ids).tobytes()).hexdigest(),
  }, indent=2) + "\n")
  args.out.joinpath("compile_memory.txt").write_text(str(compiled.memory_analysis()))
  for _ in range(3):
    jax.block_until_ready(compiled(*args_list))
  result = capture_one(compiled, args_list, args.out / "profile",
                       "SPLASH512_DIRECT_COMPUTE")
  args.out.joinpath("output_sha.json").write_text(
      json.dumps(array_sha(result), indent=2) + "\n")


if __name__ == "__main__":
  main()
