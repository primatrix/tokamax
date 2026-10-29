"""Profile (or AOT-compile) the fused a2a-cp Splash kernel once.

Uses the public ``make_splash_mha_a2a_cp`` factory exactly as a training
caller would: inside a ``shard_map`` over the cp axis, q/k/v sequence sharded,
segment ids replicated.  ``--cp 1`` runs the same kernel on one device without
any exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--cp", type=int, default=8)
  parser.add_argument("--mode", default="multimodal",
                      choices=("multimodal", "single"))
  parser.add_argument("--global-heads", type=int, default=64)
  parser.add_argument("--compile-only", action="store_true")
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)

  from config import production_config
  from inputs import make_segment_ids, make_sharded_arrays
  from profile_common import array_sha, capture_one, set_profile_flags

  set_profile_flags(args.out)
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_a2a_cp_kernel as a2a,
  )
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_kernel as sk,
  )
  from tokamax._src.ops.experimental.tpu.splash_attention import (
      splash_attention_mask as sm,
  )

  config = production_config(global_heads=args.global_heads, cp_size=args.cp)
  n, heads = config["tokens"], config["global_heads"]
  dq, dv = config["qk_head_dim"], config["v_head_dim"]
  axis = config["mesh_axis"]
  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)[:args.cp]
  if len(devices) != args.cp:
    raise RuntimeError(f"expected {args.cp} TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (axis,))

  splash_values = dict(config["splash"])
  for name in ("q_layout", "k_layout", "v_layout"):
    splash_values[name] = sk.QKVLayout(splash_values[name])
  splash_config = sk.SplashConfig(**splash_values)
  kernel = a2a.make_splash_mha_a2a_cp(
      sm.CausalMask((n, n)), config=splash_config, save_residuals=True,
      cp_size=args.cp, cp_axis=axis,
  )

  # Inputs use the baseline's [N, H, D] sequence-sharded construction so the
  # o / residual hashes are directly comparable with run_baseline.py.
  seq_in = P(axis, None, None)

  @jax.jit
  @jax.shard_map(mesh=mesh, in_specs=(seq_in, seq_in, seq_in, P(None)),
                 out_specs=(seq_in, {"logsumexp": P(axis, None),
                                     "max_logits": P(axis, None)}),
                 check_vma=False)
  def fused(q, k, v, ids):
    q, k, v = (x.transpose(1, 0, 2) for x in (q, k, v))
    out, stats = kernel(q, k, v, sk.SegmentIds(ids, ids), save_residuals=True)
    return out.transpose(1, 0, 2), stats

  specs = (
      jax.ShapeDtypeStruct((n, heads, dq), jnp.bfloat16),
      jax.ShapeDtypeStruct((n, heads, dq), jnp.bfloat16),
      jax.ShapeDtypeStruct((n, heads, dv), jnp.bfloat16),
      jax.ShapeDtypeStruct((n,), jnp.int32),
  )
  compiled = fused.lower(*specs).compile()
  args.out.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")
  args.out.joinpath("compile_memory.txt").write_text(str(compiled.memory_analysis()))
  if args.compile_only:
    args.out.joinpath("status.json").write_text(
        json.dumps({"status": "compiled", "executed": False}, indent=2) + "\n")
    return

  ids = make_segment_ids(n, args.mode)
  args.out.joinpath("input_ids.json").write_text(json.dumps({
      "mode": args.mode,
      "sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
  }, indent=2) + "\n")
  sharding = NamedSharding(mesh, seq_in)
  inputs = make_sharded_arrays(jax, specs[:3], (sharding,) * 3, devices)
  inputs.append(jax.device_put(ids, NamedSharding(mesh, P(None))))
  for _ in range(3):
    jax.block_until_ready(compiled(*inputs))
  result = capture_one(compiled, tuple(inputs), args.out / "profile",
                       "SPLASH_A2A_CP")
  args.out.joinpath("output_sha.json").write_text(
      json.dumps(array_sha(result), indent=2) + "\n")


if __name__ == "__main__":
  main()
