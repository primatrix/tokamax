"""Profile the fused Splash+A2A head-overlap kernel once."""

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
  parser.add_argument("--mode", default="multimodal", choices=("multimodal", "single"))
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  from config import production_config
  from inputs import make_segment_ids, make_sharded_arrays
  from profile_common import capture_one, set_profile_flags, array_sha
  set_profile_flags(args.out)
  import jax
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_a2a_kernel

  config = production_config()
  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)
  if len(devices) != 8:
    raise RuntimeError(f"expected 8 TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (config["mesh_axis"],))
  fn, specs, metadata = splash_attention_a2a_kernel.build(config, mesh)
  in_shardings = (
      NamedSharding(mesh, P(config["mesh_axis"], None, None)),
      NamedSharding(mesh, P(config["mesh_axis"], None, None)),
      NamedSharding(mesh, P(config["mesh_axis"], None, None)),
      NamedSharding(mesh, P(None, None)),
      NamedSharding(mesh, P(None, None)),
      NamedSharding(mesh, P(None, None)),
  )
  compiled = fn.lower(*specs).compile()
  input_arrays = make_sharded_arrays(jax, specs[:3], in_shardings[:3], devices)
  ids = make_segment_ids(config["tokens"], args.mode)
  replicated = NamedSharding(mesh, P(None, None))
  qids = jax.device_put(np.broadcast_to(ids[:, None], (config["tokens"], 128)), replicated)
  kvids = jax.device_put(np.broadcast_to(ids[None, :], (8, config["tokens"])), replicated)
  qseq = jax.device_put(np.broadcast_to(np.arange(config["tokens"], dtype=np.int32)[:, None],
                                        (config["tokens"], 128)), replicated)
  args.out.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")
  args.out.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
  args.out.joinpath("input_ids.json").write_text(json.dumps({
      "mode": args.mode,
      "sha256": __import__("hashlib").sha256(ids.tobytes()).hexdigest(),
  }, indent=2) + "\n")
  args.out.joinpath("compile_memory.txt").write_text(str(compiled.memory_analysis()))
  args_list = input_arrays + [qids, kvids, qseq]
  for _ in range(3):
    jax.block_until_ready(compiled(*args_list))
  result = capture_one(compiled, tuple(args_list), args.out / "profile",
                       "SPLASH512_A2A_FUSED")
  args.out.joinpath("output_sha.json").write_text(
      json.dumps(array_sha(result), indent=2) + "\n"
  )


if __name__ == "__main__":
  main()
