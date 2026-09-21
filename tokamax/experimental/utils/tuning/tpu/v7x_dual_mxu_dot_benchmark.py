"""Compare compiler-scheduled dots with explicit dual-MXU issue on TPU7x."""

import argparse
import json
import statistics
import time

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


_INNER_REPEATS = 16
_TILE = 256


def _reference_kernel(a0_ref, b0_ref, a1_ref, b1_ref, o0_ref, o1_ref):
  acc0 = jnp.zeros((_TILE, _TILE), jnp.float32)
  acc1 = jnp.zeros((_TILE, _TILE), jnp.float32)
  for i in range(_INNER_REPEATS):
    acc0 += lax.dot(
        a0_ref[i, ...], b0_ref[i, ...], preferred_element_type=jnp.float32
    )
    acc1 += lax.dot(
        a1_ref[i, ...], b1_ref[i, ...], preferred_element_type=jnp.float32
    )
  o0_ref[...] = acc0
  o1_ref[...] = acc1


def _dual_mxu_kernel(a0_ref, b0_ref, a1_ref, b1_ref, o0_ref, o1_ref):
  for i in range(_INNER_REPEATS):
    pltpu.matmul_push_rhs(b0_ref[i, ...], staging_register=0, mxu_index=0)
    pltpu.matmul_push_rhs(b1_ref[i, ...], staging_register=0, mxu_index=1)
    pltpu.matmul_acc_lhs(
        0, a0_ref[i, ...], mxu_index=0, load_staged_rhs=0
    )
    pltpu.matmul_acc_lhs(
        0, a1_ref[i, ...], mxu_index=1, load_staged_rhs=0
    )
  o0_ref[...] = pltpu.matmul_pop(0, (_TILE, _TILE), jnp.float32, 0)
  o1_ref[...] = pltpu.matmul_pop(0, (_TILE, _TILE), jnp.float32, 1)


def _measure(fn, args, warmup, repeats):
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    samples.append((time.perf_counter() - start) * 1e3)
  return statistics.median(samples)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--grid", type=int, default=256)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--output")
  args = parser.parse_args()

  key = jax.random.key(27)
  tensors = []
  for operand in range(4):
    value = jax.random.normal(
        jax.random.fold_in(key, operand),
        (_INNER_REPEATS, _TILE, _TILE),
        jnp.bfloat16,
    )
    # Match the ViT head dimension while giving the explicit MXU primitive its
    # required 256-wide physical operands.
    if operand in (0, 2):
      value = value.at[:, :, 72:].set(0)
    else:
      value = value.at[:, 72:, :].set(0)
    tensors.append(value)
  a0, b0, a1, b1 = tensors

  def same_input(*_):
    return 0, 0, 0

  def output_block(i):
    return i, 0, 0

  def make_call(kernel):
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(args.grid,),
            in_specs=[
                pl.BlockSpec((_INNER_REPEATS, _TILE, _TILE), same_input)
                for _ in range(4)
            ],
            out_specs=[
                pl.BlockSpec((None, _TILE, _TILE), output_block),
                pl.BlockSpec((None, _TILE, _TILE), output_block),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((args.grid, _TILE, _TILE), jnp.float32),
            jax.ShapeDtypeStruct((args.grid, _TILE, _TILE), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=63 * 1024**2,
        ),
    )

  reference = jax.jit(make_call(_reference_kernel)).lower(*tensors).compile()
  dual_mxu = jax.jit(make_call(_dual_mxu_kernel)).lower(*tensors).compile()
  reference_ms = _measure(reference, tensors, args.warmup, args.repeats)
  dual_mxu_ms = _measure(dual_mxu, tensors, args.warmup, args.repeats)
  expected = reference(*tensors)
  actual = dual_mxu(*tensors)
  relative_l2 = max(
      float(
          jnp.linalg.norm(x - y)
          / jnp.maximum(jnp.linalg.norm(y), jnp.float32(1e-30))
      )
      for x, y in zip(actual, expected)
  )
  result = {
      "variant": "v7x_bf16_dual_mxu_dot",
      "reference_ms": reference_ms,
      "dual_mxu_ms": dual_mxu_ms,
      "speedup": reference_ms / dual_mxu_ms,
      "relative_l2": relative_l2,
      "grid": args.grid,
      "inner_repeats": _INNER_REPEATS,
  }
  if args.output:
    with open(args.output, "w", encoding="utf-8") as output_file:
      for phase, latency in (
          ("reference", reference_ms),
          ("dual_mxu", dual_mxu_ms),
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
