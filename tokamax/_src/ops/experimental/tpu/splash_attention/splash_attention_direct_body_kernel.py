"""Single-head Splash body in one pallas_call, without nested emit_pipeline.

This is an isolation probe.  It keeps the production ``flash_attention_kernel``
body and metadata ABI, but puts the active-block grid on the outer pallas call
instead of wrapping it in ``pltpu.emit_pipeline``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as sk,
)
from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_a2a_kernel import (
    BQ,
    BKV,
    BKV_COMPUTE,
    N,
    NUM_LANES,
    NUM_SUBLANES,
    QK,
    V,
    _build_mask,
    _whole_spec,
)


def build(config: dict[str, Any], mesh):
  """Build a one-local-head, DMA-free direct Splash body probe."""
  if config.get("cp_size") != 8 or config.get("global_heads") != 8:
    raise ValueError("direct body probe requires CP8/global_heads=8")
  if config.get("heads") != 1 or config.get("tokens") != N:
    raise ValueError("direct body probe requires one local head and N=262144")
  splash_values = config.get("splash", {})
  if tuple(splash_values.get(x) for x in
           ("block_q", "block_kv", "block_kv_compute")) != (
               BQ, BKV, BKV_COMPUTE):
    raise ValueError("direct body probe requires 2048/2048/512 tiling")

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

  def kernel_body(active_rows_ref_in, active_cols_ref_in, mask_next_ref,
                  bounds_start_ref, bounds_end_ref, block_mask_ref,
                  q_ref, k_ref, v_ref, q_segment_ids_ref,
                  kv_segment_ids_ref, q_sequence_ref,
                  o_ref, logsumexp_ref, max_logits_ref,
                  m_scratch_ref, l_scratch_ref, o_scratch_ref):
    sk.flash_attention_kernel(
        active_rows_ref_in, active_cols_ref_in, mask_next_ref,
        bounds_start_ref, bounds_end_ref, block_mask_ref,
        q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref,
        None, None, q_sequence_ref, None,
        o_ref, logsumexp_ref, None, max_logits_ref,
        m_scratch_ref, l_scratch_ref, o_scratch_ref,
        mask_value=splash_kwargs["mask_value"], kv_steps=N // BKV,
        bq=BQ, bkv=BKV, bkv_compute=BKV_COMPUTE, head_dim_v=V,
        fuse_reciprocal=splash_kwargs["config"].fuse_reciprocal,
        config=splash_kwargs["config"],
        mask_function=splash_kwargs["mask_function"],
        native_layout=False,
    )

  # Index maps return BlockSpec block indices, not element offsets.  The
  # scalar-prefetch refs are passed by Pallas after the program ids, matching
  # Splash's own ``unravel`` helper; no JAX metadata is captured in a closure.
  def q_map(_h, grid_idx, rows_ref, cols_ref, *_):
    return (0, rows_ref[grid_idx])
  def k_map(_h, grid_idx, rows_ref, cols_ref, *_):
    return (0, cols_ref[grid_idx])
  def v_map(_h, grid_idx, rows_ref, cols_ref, *_):
    return (cols_ref[grid_idx], 0)
  def qid_map(_h, grid_idx, rows_ref, cols_ref, *_):
    return (rows_ref[grid_idx], 0)
  def kid_map(_h, grid_idx, rows_ref, cols_ref, *_):
    return (0, cols_ref[grid_idx])

  call = pl.pallas_call(
        kernel_body,
        out_shape=(
            jax.ShapeDtypeStruct((N, V), jnp.bfloat16),
            jax.ShapeDtypeStruct((N, NUM_LANES), jnp.float32),
            jax.ShapeDtypeStruct((N, NUM_LANES), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1, int(active_rows.shape[0])),
            in_specs=(
                pl.BlockSpec((QK, BQ), q_map),
                pl.BlockSpec((QK, BKV), k_map),
                pl.BlockSpec((BKV, V), v_map),
                pl.BlockSpec((BQ, NUM_LANES), qid_map),
                pl.BlockSpec((NUM_SUBLANES, BKV), kid_map),
                pl.BlockSpec((BQ, NUM_LANES), qid_map),
            ),
            out_specs=(
                pl.BlockSpec((BQ, V), qid_map),
                pl.BlockSpec((BQ, NUM_LANES), qid_map),
                pl.BlockSpec((BQ, NUM_LANES), qid_map),
            ),
            dimension_semantics=("parallel", "arbitrary"),
            scratch_shapes=[
                pltpu.VMEM((BQ, NUM_LANES), jnp.float32),
                pltpu.VMEM((BQ, NUM_LANES), jnp.float32),
                pltpu.VMEM((BQ, V), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
        ),
        name="splash_mha_direct_body_compute_only",
    )

  def wrapped(q, k, v, qids_in, kvids_in, qseq_in):
    qref = q[0].transpose(1, 0)
    kref = k[0].transpose(1, 0)
    vref = v[0]
    outputs = call(active_rows, active_cols, jnp.zeros_like(active_rows),
                   bounds_start, bounds_end, block_mask,
                   qref, kref, vref, qids_in, kvids_in, qseq_in)
    return outputs[0][None, ...], outputs[1][None, ...], outputs[2][None, ...]

  mapped = jax.shard_map(
      wrapped,
      mesh=mesh,
      in_specs=(P(None, None, None),) * 3 + (P(None, None),) * 3,
      out_specs=(P(None, None, None),) * 3,
      check_vma=False,
  )
  specs = (
      jax.ShapeDtypeStruct((1, N, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((1, N, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((1, N, V), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, N), jnp.int32),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
  )
  return jax.jit(mapped), specs, {
      "expected_kernel_name": "splash_mha_direct_body_compute_only",
      "custom_call_count_expected": 1,
      "diagnostic": "one-head outer pallas grid; no nested emit_pipeline; no DMA",
      "shape": {"H_local": 1, "N": N, "qk": QK, "v": V},
      "splash": {"block_q": BQ, "block_kv": BKV,
                  "block_kv_compute": BKV_COMPUTE},
  }
