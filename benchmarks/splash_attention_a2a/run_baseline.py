"""Profile native Ulysses pre-a2a + Splash512 + post-a2a once."""

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
  parser.add_argument("--maxtext-source", type=Path, required=True)
  parser.add_argument("--mode", default="multimodal", choices=("multimodal", "single"))
  parser.add_argument("--global-heads", type=int, default=64)
  parser.add_argument("--cp", type=int, default=8)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  from config import production_config
  from inputs import make_segment_ids, make_sharded_arrays, make_replicated_ids
  from profile_common import capture_one, set_profile_flags, array_sha
  set_profile_flags(args.out)
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from ulysses_baseline import build

  config = production_config(global_heads=args.global_heads, cp_size=args.cp)
  # The first ``cp`` devices form the context mesh, matching run_a2a_cp.py.
  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)[:args.cp]
  if len(devices) != args.cp:
    raise RuntimeError(f"expected {args.cp} TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (config["mesh_axis"],))
  fn, specs, in_shardings, out_shardings, metadata = build(
      config, mesh, args.maxtext_source
  )
  compiled = jax.jit(fn, in_shardings=in_shardings,
                     out_shardings=out_shardings).lower(*specs).compile()
  input_arrays = make_sharded_arrays(jax, specs[:3], in_shardings[:3], devices)
  ids = make_segment_ids(config["tokens"], args.mode)
  replicated = NamedSharding(mesh, P(None, None))
  ids_array = jax.device_put(ids[None, :], replicated)
  args.out.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")
  args.out.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
  args.out.joinpath("input_ids.json").write_text(json.dumps({
      "mode": args.mode,
      "sha256": __import__("hashlib").sha256(ids.tobytes()).hexdigest(),
  }, indent=2) + "\n")
  args.out.joinpath("compile_memory.txt").write_text(str(compiled.memory_analysis()))
  for _ in range(3):
    jax.block_until_ready(compiled(*input_arrays, ids_array))
  result = capture_one(compiled, tuple(input_arrays + [ids_array]),
                       args.out / "profile", "SPLASH512_BASELINE")
  args.out.joinpath("output_sha.json").write_text(
      json.dumps(array_sha(result), indent=2) + "\n"
  )


if __name__ == "__main__":
  main()
