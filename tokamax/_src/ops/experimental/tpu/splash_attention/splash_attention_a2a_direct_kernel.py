"""Single-call Splash+A2A candidate without nested ``emit_pipeline``.

The head-level A2A schedule is the accepted Step-C schedule.  Unlike the
older candidate, the active-block loop is explicit inside this pallas call;
each block is loaded into VMEM and invokes the pinned Splash body with an
explicit grid index.  This is a structural probe, not yet a performance
candidate: tile DMA is intentionally synchronous until the no-emit shape is
validated by AOT.
"""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
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
    CP_SIZE,
    H_GLOBAL,
    H_LOCAL,
    N,
    NUM_LANES,
    NUM_SUBLANES,
    QK,
    T,
    V,
    _build_mask,
    _copy,
    _remote,
    _start_a2a_head,
    _start_o_head,
    _wait_a2a_head,
    _wait_o_head,
    _whole_spec,
)


def _direct_head_loop(
    h, *, rank, qref, kref, vref, qid_ref, kvid_ref, qseq_ref,
    ofull, lse, max_logits, active_rows_ref, active_cols_ref,
    mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref,
    m_scratch, l_scratch, o_scratch,
    q_tile, k_tile, v_tile, qid_tile, kvid_tile, qseq_tile, tile_sem,
    splash_kwargs,
):
  """Run one head's active blocks without a nested pipeline."""
  active_count = active_rows_ref.shape[0]

  def body(grid_idx, _):
    q_block = active_rows_ref[grid_idx] * BQ
    k_block = active_cols_ref[grid_idx] * BKV
    # Load the same physical tiles that the old BlockSpec/emit_pipeline path
    # supplied.  A single slot is deliberate for this structural probe.
    copies = (
        pltpu.make_async_copy(
            qref.at[pl.ds(h * QK, QK), pl.ds(q_block, BQ)],
            q_tile, tile_sem.at[0]),
        pltpu.make_async_copy(
            kref.at[pl.ds(h * QK, QK), pl.ds(k_block, BKV)],
            k_tile, tile_sem.at[1]),
        pltpu.make_async_copy(
            vref.at[pl.ds(h * N + k_block, BKV), :],
            v_tile, tile_sem.at[2]),
        pltpu.make_async_copy(
            qid_ref.at[pl.ds(q_block, BQ), :], qid_tile, tile_sem.at[3]),
        pltpu.make_async_copy(
            kvid_ref.at[:, pl.ds(k_block, BKV)], kvid_tile, tile_sem.at[4]),
        pltpu.make_async_copy(
            qseq_ref.at[pl.ds(q_block, BQ), :], qseq_tile, tile_sem.at[5]),
    )
    for copy in copies:
      copy.start()
    for copy in copies:
      copy.wait()

    sk.flash_attention_kernel(
        active_rows_ref, active_cols_ref, mask_next_ref,
        bounds_start_ref, bounds_end_ref, block_mask_ref,
        q_tile, k_tile, v_tile, qid_tile, kvid_tile, None, None,
        qseq_tile, None,
        ofull.at[pl.ds(h * N, N), :],
        lse.at[pl.ds(h * N, N), :], None,
        max_logits.at[pl.ds(h * N, N), :],
        m_scratch, l_scratch, o_scratch,
        mask_value=splash_kwargs["mask_value"], kv_steps=N // BKV,
        bq=BQ, bkv=BKV, bkv_compute=BKV_COMPUTE, head_dim_v=V,
        fuse_reciprocal=splash_kwargs["config"].fuse_reciprocal,
        config=splash_kwargs["config"],
        mask_function=splash_kwargs["mask_function"],
        native_layout=False,
        grid_idx_override=grid_idx,
        head_idx_override=jnp.asarray(0, dtype=jnp.int32),
        num_heads_override=1,
    )
    return ()

  lax.fori_loop(0, active_count, body, ())


def build(config: dict[str, Any], mesh):
  """Build the direct-body A2A candidate with one custom call."""
  if config.get("cp_size") != CP_SIZE or config.get("global_heads") != H_GLOBAL:
    raise ValueError("direct A2A currently supports CP8/H=64 only")
  if config.get("tokens") != N or config.get("heads") != H_LOCAL:
    raise ValueError("direct A2A currently supports N=262144/H_local=8 only")
  splash_values = config.get("splash", {})
  if tuple(splash_values.get(x) for x in
           ("block_q", "block_kv", "block_kv_compute")) != (
               BQ, BKV, BKV_COMPUTE):
    raise ValueError("direct A2A requires 2048/2048/512 tiling")

  mask_info, splash_kwargs = _build_mask(config)
  active_rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
  active_cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
  block_mask = jnp.asarray(mask_info.block_mask)
  row_break_i32 = (active_rows[1:] != active_rows[:-1]).astype(jnp.int32)
  bounds_start = jnp.concatenate((jnp.asarray([1], dtype=jnp.int32),
                                  row_break_i32))
  bounds_end = jnp.concatenate((row_break_i32,
                                jnp.asarray([1], dtype=jnp.int32)))

  def kernel_body(rank_ref, active_rows_ref, active_cols_ref, mask_next_ref,
                  bounds_start_ref, bounds_end_ref, block_mask_ref,
                  qref, kref, vref, qid_ref, kvid_ref, qseq_ref, *outs):
    qrecv, krecv, vrecv, ofull, oseq, lse, max_logits = outs[:7]
    m_scratch, l_scratch, o_scratch = outs[7:10]
    q_tile, k_tile, v_tile = outs[10:13]
    qid_tile, kvid_tile, qseq_tile = outs[13:16]

    barrier_sem = pltpu.get_barrier_semaphore()
    for step in range(1, CP_SIZE):
      dst = (rank_ref[0] + step) % CP_SIZE
      pl.semaphore_signal(
          barrier_sem, inc=1, device_id={"context": dst},
          device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, CP_SIZE - 1)

    def scope(q_send, q_recv_sem, o_send, o_recv_sem, local_sem,
              tile_sem):
      pending_in = _start_a2a_head(
          rank_ref[0], 0, qref, kref, vref, qrecv, krecv, vrecv,
          q_send, q_recv_sem, local_sem, mesh_axis="context")
      _wait_a2a_head(pending_in)
      pending_in = _start_a2a_head(
          rank_ref[0], 1, qref, kref, vref, qrecv, krecv, vrecv,
          q_send, q_recv_sem, local_sem, mesh_axis="context")
      pending_o = None
      for h in range(H_LOCAL):
        if h > 0:
          _wait_a2a_head(pending_in)
          if h + 1 < H_LOCAL:
            pending_in = _start_a2a_head(
                rank_ref[0], h + 1, qref, kref, vref, qrecv, krecv,
                vrecv, q_send, q_recv_sem, local_sem, mesh_axis="context")
          else:
            pending_in = None
        _direct_head_loop(
            h, rank=rank_ref[0], qref=qrecv, kref=krecv, vref=vrecv,
            qid_ref=qid_ref, kvid_ref=kvid_ref, qseq_ref=qseq_ref,
            ofull=ofull, lse=lse, max_logits=max_logits,
            active_rows_ref=active_rows_ref, active_cols_ref=active_cols_ref,
            mask_next_ref=mask_next_ref, bounds_start_ref=bounds_start_ref,
            bounds_end_ref=bounds_end_ref, block_mask_ref=block_mask_ref,
            m_scratch=m_scratch, l_scratch=l_scratch, o_scratch=o_scratch,
            q_tile=q_tile, k_tile=k_tile, v_tile=v_tile,
            qid_tile=qid_tile, kvid_tile=kvid_tile, qseq_tile=qseq_tile,
            tile_sem=tile_sem, splash_kwargs=splash_kwargs)
        if pending_o is not None:
          _wait_o_head(pending_o)
        pending_o = _start_o_head(
            rank_ref[0], h, ofull, oseq, o_send, o_recv_sem, local_sem,
            mesh_axis="context")
      _wait_o_head(pending_o)

    pl.run_scoped(
        scope,
        pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE, 3)),
        pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE, 3)),
        pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE)),
        pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE)),
        pltpu.SemaphoreType.DMA((H_LOCAL, 4)),
        pltpu.SemaphoreType.DMA((6,)),
    )

  out_shapes = (
      jax.ShapeDtypeStruct((H_LOCAL * QK, N), jnp.bfloat16),
      jax.ShapeDtypeStruct((H_LOCAL * QK, N), jnp.bfloat16),
      jax.ShapeDtypeStruct((H_LOCAL * N, V), jnp.bfloat16),
      jax.ShapeDtypeStruct((H_LOCAL * N, V), jnp.bfloat16),
      jax.ShapeDtypeStruct((T, H_GLOBAL * V), jnp.bfloat16),
      jax.ShapeDtypeStruct((H_LOCAL * N, NUM_LANES), jnp.float32),
      jax.ShapeDtypeStruct((H_LOCAL * N, NUM_LANES), jnp.float32),
  )
  call = pl.pallas_call(
      kernel_body,
      out_shape=out_shapes,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=7,
          grid=(),
          in_specs=(
              _whole_spec((H_GLOBAL * QK, T)),
              _whole_spec((H_GLOBAL * QK, T)),
              _whole_spec((H_GLOBAL * T, V)),
              _whole_spec((N, NUM_LANES)),
              _whole_spec((NUM_SUBLANES, N)),
              _whole_spec((N, NUM_LANES)),
          ),
          out_specs=tuple(_whole_spec(s.shape) for s in out_shapes),
          scratch_shapes=[
              pltpu.VMEM((BQ, NUM_LANES), jnp.float32),
              pltpu.VMEM((BQ, NUM_LANES), jnp.float32),
              pltpu.VMEM((BQ, V), jnp.float32),
              pltpu.VMEM((QK, BQ), jnp.bfloat16),
              pltpu.VMEM((QK, BKV), jnp.bfloat16),
              pltpu.VMEM((BKV, V), jnp.bfloat16),
              pltpu.VMEM((BQ, NUM_LANES), jnp.int32),
              pltpu.VMEM((NUM_SUBLANES, BKV), jnp.int32),
              pltpu.VMEM((BQ, NUM_LANES), jnp.int32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(), collective_id=0,
          flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
      ),
      name="splash_mha_direct_body_a2a_overlap",
  )

  def wrapped(rank_value, q, k, v, qids_in, kvids_in, qseq_in):
    qsrc = q.transpose(1, 2, 0).reshape(H_GLOBAL * QK, T)
    ksrc = k.transpose(1, 2, 0).reshape(H_GLOBAL * QK, T)
    vsrc = v.transpose(1, 0, 2).reshape(H_GLOBAL * T, V)
    outputs = call(rank_value, active_rows, active_cols,
                   jnp.zeros_like(active_rows), bounds_start, bounds_end,
                   block_mask, qsrc, ksrc, vsrc, qids_in, kvids_in, qseq_in)
    return outputs[4].reshape(T, H_GLOBAL, V), outputs[5].reshape(
        H_LOCAL, N, NUM_LANES), outputs[6].reshape(H_LOCAL, N, NUM_LANES)

  def local(q, k, v, qids_in, kvids_in, qseq_in):
    rank_value = lax.axis_index("context").reshape((1,))
    return wrapped(rank_value, q, k, v, qids_in, kvids_in, qseq_in)

  mapped = jax.shard_map(
      local, mesh=mesh,
      in_specs=(P("context", None, None),) * 3 + (P(None, None),) * 3,
      out_specs=(P("context", None, None),) * 3,
      check_vma=False,
  )
  specs = (
      jax.ShapeDtypeStruct((N, H_GLOBAL, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, H_GLOBAL, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, H_GLOBAL, V), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, N), jnp.int32),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
  )
  return jax.jit(mapped), specs, {
      "expected_kernel_name": "splash_mha_direct_body_a2a_overlap",
      "custom_call_count_expected": 1,
      "diagnostic": "head-level A2A overlap with manual active-block loop; no emit_pipeline",
      "shape": {"H_local": H_LOCAL, "H_global": H_GLOBAL, "N": N,
                "T": T, "qk": QK, "v": V},
      "splash": {"block_q": BQ, "block_kv": BKV,
                  "block_kv_compute": BKV_COMPUTE},
  }


make_splash_mha_a2a = build
