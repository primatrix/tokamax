"""Probe whether block-diagonal packing can fill v7x BF16 MXU lanes."""

import argparse
import json
import time

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _measure(fn, args, warmup, repeats):
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    samples.append((time.perf_counter() - start) * 1e3)
  return sorted(samples)[len(samples) // 2]


def _separate_kernel(a0_ref, b0_ref, a1_ref, b1_ref, o0_ref, o1_ref):
  o0_ref[...] = lax.dot(
      a0_ref[...], b0_ref[...], preferred_element_type=jnp.float32
  )
  o1_ref[...] = lax.dot(
      a1_ref[...], b1_ref[...], preferred_element_type=jnp.float32
  )


def _packed_kernel(a_ref, b_ref, o0_ref, o1_ref):
  packed = lax.dot(
      a_ref[...], b_ref[...], preferred_element_type=jnp.float32
  )
  o0_ref[...] = packed[:128, :128]
  o1_ref[...] = packed[128:, 128:]


def _single_kernel(a_ref, b_ref, o_ref):
  o_ref[...] = lax.dot(
      a_ref[...], b_ref[...], preferred_element_type=jnp.float32
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--grid", type=int, default=256)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--output")
  args = parser.parse_args()

  key = jax.random.key(27)
  keys = jax.random.split(key, 4)
  a0, a1 = [jax.random.normal(k, (128, 72), jnp.bfloat16) for k in keys[:2]]
  b0, b1 = [jax.random.normal(k, (72, 128), jnp.bfloat16) for k in keys[2:]]
  zero_a = jnp.zeros_like(a0)
  zero_b = jnp.zeros_like(b0)
  a_packed = jnp.concatenate(
      (
          jnp.concatenate((a0, zero_a), axis=1),
          jnp.concatenate((zero_a, a1), axis=1),
      ),
      axis=0,
  )
  b_packed = jnp.concatenate(
      (
          jnp.concatenate((b0, zero_b), axis=1),
          jnp.concatenate((zero_b, b1), axis=1),
      ),
      axis=0,
  )

  def same_block(*_):
    return 0, 0

  def output_block(i):
    return i, 0, 0

  separate = pl.pallas_call(
      _separate_kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(args.grid,),
          in_specs=[
              pl.BlockSpec((128, 72), same_block),
              pl.BlockSpec((72, 128), same_block),
              pl.BlockSpec((128, 72), same_block),
              pl.BlockSpec((72, 128), same_block),
          ],
          out_specs=[
              pl.BlockSpec((None, 128, 128), output_block),
              pl.BlockSpec((None, 128, 128), output_block),
          ],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((args.grid, 128, 128), jnp.float32),
          jax.ShapeDtypeStruct((args.grid, 128, 128), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",), vmem_limit_bytes=63 * 1024**2
      ),
  )
  packed = pl.pallas_call(
      _packed_kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(args.grid,),
          in_specs=[
              pl.BlockSpec((256, 144), same_block),
              pl.BlockSpec((144, 256), same_block),
          ],
          out_specs=[
              pl.BlockSpec((None, 128, 128), output_block),
              pl.BlockSpec((None, 128, 128), output_block),
          ],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((args.grid, 128, 128), jnp.float32),
          jax.ShapeDtypeStruct((args.grid, 128, 128), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",), vmem_limit_bytes=63 * 1024**2
      ),
  )
  separate = jax.jit(separate).lower(a0, b0, a1, b1).compile()
  packed = jax.jit(packed).lower(a_packed, b_packed).compile()
  separate_ms = _measure(
      separate, (a0, b0, a1, b1), args.warmup, args.repeats
  )
  packed_ms = _measure(packed, (a_packed, b_packed), args.warmup, args.repeats)
  k_sweep_ms = {}
  for k_dim in (64, 72, 128, 144, 256):
    a = jax.random.normal(jax.random.fold_in(key, k_dim), (128, k_dim), jnp.bfloat16)
    b = jax.random.normal(
        jax.random.fold_in(key, k_dim + 1000), (k_dim, 128), jnp.bfloat16
    )
    single = pl.pallas_call(
        _single_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(args.grid,),
            in_specs=[
                pl.BlockSpec((128, k_dim), same_block),
                pl.BlockSpec((k_dim, 128), same_block),
            ],
            out_specs=pl.BlockSpec((None, 128, 128), output_block),
        ),
        out_shape=jax.ShapeDtypeStruct((args.grid, 128, 128), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=63 * 1024**2
        ),
    )
    single = jax.jit(single).lower(a, b).compile()
    k_sweep_ms[str(k_dim)] = _measure(
        single, (a, b), args.warmup, args.repeats
    )
  result = {
      "variant": "v7x_bf16_block_diagonal_dot",
      "separate_ms": separate_ms,
      "packed_ms": packed_ms,
      "packed_speedup": separate_ms / packed_ms,
      "grid": args.grid,
      "k_sweep_ms": k_sweep_ms,
  }
  if args.output:
    with open(args.output, "w", encoding="utf-8") as output_file:
      for phase, latency in (
          ("separate", separate_ms),
          ("packed", packed_ms),
          *((f"k_{k_dim}", latency) for k_dim, latency in k_sweep_ms.items()),
      ):
        output_file.write(
            json.dumps(
                {
                    "variant": result["variant"],
                    "phase": phase,
                    "latency_ms": latency,
                    "dtype": "bfloat16",
                },
                sort_keys=True,
            )
            + "\n"
        )
  print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
