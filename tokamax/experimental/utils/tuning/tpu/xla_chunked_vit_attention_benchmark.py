"""Benchmark native-XLA batched BF16 attention on one dense ViT segment."""

import argparse
import json
import time

import jax
import jax.numpy as jnp

from tokamax._src.ops.attention import base
from tokamax._src.ops.attention import xla_chunked


def _ready(tree):
  return jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
      tree,
  )


def _median_ms(fn, args, warmup, repeats):
  for _ in range(warmup):
    _ready(fn(*args))
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    _ready(fn(*args))
    samples.append((time.perf_counter() - start) * 1e3)
  return sorted(samples)[len(samples) // 2]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--heads", type=int, default=16)
  parser.add_argument("--sequence", type=int, default=16384)
  parser.add_argument("--head-dim", type=int, default=72)
  parser.add_argument("--block-q", type=int, default=256)
  parser.add_argument("--block-kv", type=int, default=1024)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeats", type=int, default=10)
  parser.add_argument("--output")
  args = parser.parse_args()

  shape = (args.batch, args.sequence, args.heads, args.head_dim)
  keys = jax.random.split(jax.random.key(27), 4)
  q, k, v, do = [
      jax.random.normal(key, shape, dtype=jnp.bfloat16) for key in keys
  ]
  precision = (jax.lax.DotAlgorithmPreset.BF16_BF16_F32,) * 2

  def forward(q, k, v):
    out, _ = xla_chunked._attend_chunked(  # pylint: disable=protected-access
        q,
        k,
        v,
        precision=precision,
        logits_dtype=jnp.float32,
        logits_scale=args.head_dim**-0.5,
        bias=None,
        logits_soft_cap=None,
        mask=base.Mask(),
        dropout_mask=None,
        dropout_rate=0.0,
        paging_info=None,
        q_indices=None,
        k_indices=None,
        normalize_output=True,
        chunk_size=(args.block_q, args.block_kv),
    )
    return out

  def value_and_grad(q, k, v, do):
    def loss_with_output(q, k, v):
      out = forward(q, k, v)
      loss = jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32))
      return loss, out

    return jax.value_and_grad(loss_with_output, argnums=(0, 1, 2), has_aux=True)(
        q, k, v
    )

  forward_compiled = jax.jit(forward).lower(q, k, v).compile()
  train_compiled = jax.jit(value_and_grad).lower(q, k, v, do).compile()
  forward_ms = _median_ms(
      forward_compiled, (q, k, v), args.warmup, args.repeats
  )
  train_ms = _median_ms(
      train_compiled, (q, k, v, do), args.warmup, args.repeats
  )
  result = {
      "variant": "xla_chunked_dense_half",
      "forward_ms": forward_ms,
      "train_ms": train_ms,
      "inferred_backward_ms": train_ms - forward_ms,
      "estimated_two_segment_forward_ms": 2 * forward_ms,
      "estimated_two_segment_combined_ms": 2 * train_ms,
      "shape": shape,
      "block_q": args.block_q,
      "block_kv": args.block_kv,
  }
  if args.output:
    with open(args.output, "w", encoding="utf-8") as output_file:
      for phase, latency in (
          ("forward", 2 * forward_ms),
          ("combined", 2 * train_ms),
      ):
        output_file.write(
            json.dumps(
                {
                    "variant": result["variant"],
                    "phase": phase,
                    "latency_ms": latency,
                    "batch_size": args.batch,
                    "heads": args.heads,
                    "seq_len": 2 * args.sequence,
                    "head_dim": args.head_dim,
                    "dtype": "bfloat16",
                },
                sort_keys=True,
            )
            + "\n"
        )
  print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
