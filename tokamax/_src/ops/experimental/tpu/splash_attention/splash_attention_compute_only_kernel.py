"""Compute-only diagnostic for the CP8 fused Splash candidate.

This keeps the same per-head loop and nested Splash emit_pipeline as the
head-overlap candidate, but removes all remote/local DMA and semaphore work.
Each device receives complete local-head Q/K/V regions as inputs.  It is a
diagnostic timing path, not a production ABI or correctness implementation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_a2a_kernel import (
    BQ,
    BKV,
    BKV_COMPUTE,
    H_LOCAL,
    N,
    NUM_LANES,
    NUM_SUBLANES,
    QK,
    V,
    _build_mask,
    _pipeline_for_head,
    _splash_config,
    _whole_spec,
)


def build(config: dict[str, Any], mesh):
  """Build the current fused structure with DMA removed for timing only."""
  if config.get("cp_size") != 8 or config.get("heads") != H_LOCAL:
    raise ValueError("compute-only diagnostic requires CP8 and 8 local heads")
  if config.get("tokens") != N or config.get("global_heads") != 64:
    raise ValueError("compute-only diagnostic requires N=262144 and H=64")
  splash_values = config.get("splash", {})
  if tuple(splash_values.get(x) for x in
           ("block_q", "block_kv", "block_kv_compute")) != (
               BQ, BKV, BKV_COMPUTE):
    raise ValueError("compute-only diagnostic requires 2048/2048/512 tiling")

  mask_info, splash_kwargs = _build_mask(config)
  active_rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
  active_cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
  block_mask = jnp.asarray(mask_info.block_mask)
  row_break = active_rows[1:] != active_rows[:-1]
  row_break_i32 = row_break.astype(jnp.int32)
  bounds_start = jnp.concatenate((jnp.asarray([1], dtype=jnp.int32),
                                  row_break_i32))
  bounds_end = jnp.concatenate((row_break_i32,
                                jnp.asarray([1], dtype=jnp.int32)))

  def kernel_body(active_rows_ref, active_cols_ref, mask_next_ref,
                  bounds_start_ref, bounds_end_ref, block_mask_ref,
                  qref, kref, vref, qid_ref, kvid_ref, qseq_ref,
                  ofull, lse, max_logits,
                  m_scratch, l_scratch, o_scratch):
    # The Python-unrolled head loop is intentionally identical to the
    # candidate.  Only the A2A start/wait calls are absent.
    for h in range(H_LOCAL):
      _pipeline_for_head(
          h, mask_info=mask_info, splash_kwargs=splash_kwargs,
          qref=qref.at[pl.ds(h * QK, QK), :],
          kref=kref.at[pl.ds(h * QK, QK), :],
          vref=vref.at[pl.ds(h * N, N), :],
          qids=qid_ref, kvids=kvid_ref, qseq=qseq_ref,
          out=ofull.at[pl.ds(h * N, N), :],
          lse=lse.at[pl.ds(h * N, N), :],
          max_logits=max_logits.at[pl.ds(h * N, N), :],
          metadata_refs=(active_rows_ref, active_cols_ref, mask_next_ref,
                         bounds_start_ref, bounds_end_ref, block_mask_ref),
          m_scratch=m_scratch, l_scratch=l_scratch, o_scratch=o_scratch)

  call = pl.pallas_call(
      kernel_body,
      out_shape=(
          jax.ShapeDtypeStruct((H_LOCAL * N, V), jnp.bfloat16),
          jax.ShapeDtypeStruct((H_LOCAL * N, NUM_LANES), jnp.float32),
          jax.ShapeDtypeStruct((H_LOCAL * N, NUM_LANES), jnp.float32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=6,
          grid=(),
          in_specs=(
              _whole_spec((H_LOCAL * QK, N)),
              _whole_spec((H_LOCAL * QK, N)),
              _whole_spec((H_LOCAL * N, V)),
              _whole_spec((N, NUM_LANES)),
              _whole_spec((NUM_SUBLANES, N)),
              _whole_spec((N, NUM_LANES)),
          ),
          out_specs=(
              _whole_spec((H_LOCAL * N, V)),
              _whole_spec((H_LOCAL * N, NUM_LANES)),
              _whole_spec((H_LOCAL * N, NUM_LANES)),
          ),
          scratch_shapes=[
              pltpu.VMEM((BQ, NUM_LANES), jnp.float32),
              pltpu.VMEM((BQ, NUM_LANES), jnp.float32),
              pltpu.VMEM((BQ, V), jnp.float32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(),
      ),
      name="splash_mha_compute_only",
  )

  axis = config["mesh_axis"]

  def wrapped(q, k, v, qids_in, kvids_in, qseq_in):
    qref = q.transpose(0, 2, 1).reshape(H_LOCAL * QK, N)
    kref = k.transpose(0, 2, 1).reshape(H_LOCAL * QK, N)
    vref = v.reshape(H_LOCAL * N, V)
    outputs = call(active_rows, active_cols, jnp.zeros_like(active_rows),
                   bounds_start, bounds_end, block_mask,
                   qref, kref, vref, qids_in, kvids_in, qseq_in)
    return (outputs[0].reshape(H_LOCAL, N, V),
            outputs[1].reshape(H_LOCAL, N, NUM_LANES),
            outputs[2].reshape(H_LOCAL, N, NUM_LANES))

  def local(q, k, v, qids_in, kvids_in, qseq_in):
    return wrapped(q, k, v, qids_in, kvids_in, qseq_in)

  mapped = jax.shard_map(
      local,
      mesh=mesh,
      in_specs=(P(None, None, None),) * 3 + (P(None, None),) * 3,
      out_specs=(P(None, None, None),) * 3,
      check_vma=False,
  )
  specs = (
      jax.ShapeDtypeStruct((H_LOCAL, N, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((H_LOCAL, N, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((H_LOCAL, N, V), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, N), jnp.int32),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
  )
  metadata = {
      "expected_kernel_name": "splash_mha_compute_only",
      "custom_call_count_expected": 1,
      "diagnostic": "same head loop and nested Splash emit_pipeline; no DMA/semaphore",
      "shape": {"H_local": H_LOCAL, "N": N, "qk": QK, "v": V},
      "splash": {"block_q": BQ, "block_kv": BKV,
                  "block_kv_compute": BKV_COMPUTE},
  }
  return jax.jit(mapped), specs, metadata
