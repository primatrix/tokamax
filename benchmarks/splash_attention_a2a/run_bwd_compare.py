"""Compare dq/dk/dv of the fused a2a-cp kernel against the native baseline.

Baseline: frozen MaxText Ulysses pre-a2a + native Splash + post-a2a, the
gradient comes from autodiff through the sharding constraints (dO a2a, native
bwd, dq/dk/dv a2a).  Candidate: ``make_splash_mha_a2a_cp`` inside a
shard_map, whose custom_vjp runs the same native bwd on head-sharded buffers.

Both sides use the same q/k/v/dO values (same ``make_sharded_arrays`` seeds as
the forward benchmarks), so per-shard SHA equality means bitwise equality.
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
  parser.add_argument("--maxtext-source", type=Path, required=True)
  parser.add_argument("--cp", type=int, default=8)
  parser.add_argument("--mode", default="multimodal",
                      choices=("multimodal", "single"))
  parser.add_argument("--global-heads", type=int, default=64)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)

  from config import production_config
  from inputs import make_segment_ids, make_sharded_arrays
  from profile_common import array_sha, set_profile_flags

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
  from ulysses_baseline import build

  config = production_config(global_heads=args.global_heads, cp_size=args.cp)
  n, heads = config["tokens"], config["global_heads"]
  dqk, dv = config["qk_head_dim"], config["v_head_dim"]
  axis = config["mesh_axis"]
  devices = sorted(jax.devices("tpu"), key=lambda d: d.id)[:args.cp]
  if len(devices) != args.cp:
    raise RuntimeError(f"expected {args.cp} TPU devices, got {len(devices)}")
  mesh = Mesh(np.asarray(devices), (axis,))

  # Baseline: [1, N, H, D] sequence sharded.
  base_fn, base_specs, base_in, _, _ = build(config, mesh, args.maxtext_source)
  seq4 = base_in[0]
  ids_sharding = base_in[3]

  @jax.jit
  def base_grad(q, k, v, ids, do):
    _, vjp = jax.vjp(lambda q, k, v: base_fn(q, k, v, ids)[0], q, k, v)
    return vjp(do)

  # Candidate: [N, H, D] sequence sharded, same bytes per shard.
  splash_values = dict(config["splash"])
  for name in ("q_layout", "k_layout", "v_layout"):
    splash_values[name] = sk.QKVLayout(splash_values[name])
  kernel = a2a.make_splash_mha_a2a_cp(
      sm.CausalMask((n, n)), config=sk.SplashConfig(**splash_values),
      save_residuals=True, cp_size=args.cp, cp_axis=axis,
  )
  seq3 = P(axis, None, None)

  @jax.shard_map(mesh=mesh, in_specs=(seq3, seq3, seq3, P(None)),
                 out_specs=seq3, check_vma=False)
  def fused(q, k, v, ids):
    q, k, v = (x.transpose(1, 0, 2) for x in (q, k, v))
    out, _ = kernel(q, k, v, sk.SegmentIds(ids, ids), save_residuals=True)
    return out.transpose(1, 0, 2)

  @jax.jit
  def fused_grad(q, k, v, ids, do):
    _, vjp = jax.vjp(lambda q, k, v: fused(q, k, v, ids), q, k, v)
    return vjp(do)

  # Inputs: operands 0..2 are q/k/v with the forward benchmark seeds; dO is
  # operand 3 of the same generator.
  do_spec4 = jax.ShapeDtypeStruct((1, n, heads, dv), jnp.bfloat16)
  base_inputs = make_sharded_arrays(
      jax, list(base_specs[:3]) + [do_spec4], (seq4,) * 4, devices)
  ids = make_segment_ids(n, args.mode)
  ids_base = jax.device_put(ids[None, :], ids_sharding)
  ids_fused = jax.device_put(ids, NamedSharding(mesh, P(None)))
  # Same shard bytes, leading batch axis dropped.
  fused_inputs = [x[0] for x in base_inputs]
  fused_inputs = [jax.device_put(x, NamedSharding(mesh, seq3))
                  for x in fused_inputs]

  base_out = jax.block_until_ready(
      base_grad(base_inputs[0], base_inputs[1], base_inputs[2], ids_base,
                base_inputs[3]))
  fused_out = jax.block_until_ready(
      fused_grad(fused_inputs[0], fused_inputs[1], fused_inputs[2], ids_fused,
                 fused_inputs[3]))

  report = {"cp": args.cp, "mode": args.mode,
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
  all_equal = True
  for name, gb, gf in zip(("dq", "dk", "dv"), base_out, fused_out):
    sb = sorted(array_sha(gb), key=lambda r: r["device"])
    sf = sorted(array_sha(gf), key=lambda r: r["device"])
    equal = sum(a["sha256"] == b["sha256"] for a, b in zip(sb, sf))
    diff = jnp.abs(gb[0].astype(jnp.float32) - gf.astype(jnp.float32))
    entry = {
        "shards_equal": f"{equal}/{len(sb)}",
        "max_abs_diff": float(jnp.max(diff)),
        "num_diff": int(jnp.sum(diff > 0)),
        "max_abs_ref": float(jnp.max(jnp.abs(gb.astype(jnp.float32)))),
    }
    all_equal &= equal == len(sb)
    report[name] = entry
  report["bitwise_equal"] = bool(all_equal)
  args.out.joinpath("grad_compare.json").write_text(
      json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
