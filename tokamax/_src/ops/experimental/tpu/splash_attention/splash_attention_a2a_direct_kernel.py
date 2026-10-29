"""Single-call Splash+A2A candidate without nested ``emit_pipeline``.

The head-level A2A schedule is the accepted Step-C schedule.  Unlike the
older candidate, the active-block loop is explicit inside this pallas call;
each block is loaded into VMEM and invokes the pinned Splash body with an
explicit grid index.  Input tiles are double-buffered (block g+1 is prefetched
while block g computes) and output tiles are double-buffered (a finished row is
written back to HBM while the next row computes).
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


def _in_copies(h, g, slot, *, qref, kref, vref, qid_ref, kvid_ref, qseq_ref,
               active_rows_ref, active_cols_ref, q_tile, k_tile, v_tile,
               qid_tile, kvid_tile, qseq_tile, in_sem):
  """DMA descriptors that load active block ``g`` of head ``h`` into ``slot``."""
  q_block = active_rows_ref[g] * BQ
  k_block = active_cols_ref[g] * BKV
  return (
      pltpu.make_async_copy(
          qref.at[pl.ds(h * QK, QK), pl.ds(q_block, BQ)],
          q_tile.at[slot], in_sem.at[slot, 0]),
      pltpu.make_async_copy(
          kref.at[pl.ds(h * QK, QK), pl.ds(k_block, BKV)],
          k_tile.at[slot], in_sem.at[slot, 1]),
      pltpu.make_async_copy(
          vref.at[pl.ds(h * N + k_block, BKV), :],
          v_tile.at[slot], in_sem.at[slot, 2]),
      pltpu.make_async_copy(
          qid_ref.at[pl.ds(q_block, BQ), :], qid_tile.at[slot],
          in_sem.at[slot, 3]),
      pltpu.make_async_copy(
          kvid_ref.at[:, pl.ds(k_block, BKV)], kvid_tile.at[slot],
          in_sem.at[slot, 4]),
      pltpu.make_async_copy(
          qseq_ref.at[pl.ds(q_block, BQ), :], qseq_tile.at[slot],
          in_sem.at[slot, 5]),
  )


def _out_copies(h, q_block, slot, *, ofull, lse, max_logits,
                o_tile, lse_tile, max_tile, out_sem):
  """DMA descriptors that write one finished q row from ``slot`` to HBM."""
  return (
      pltpu.make_async_copy(
          o_tile.at[slot], ofull.at[pl.ds(h * N + q_block, BQ), :],
          out_sem.at[slot, 0]),
      pltpu.make_async_copy(
          lse_tile.at[slot], lse.at[pl.ds(h * N + q_block, BQ), :],
          out_sem.at[slot, 1]),
      pltpu.make_async_copy(
          max_tile.at[slot], max_logits.at[pl.ds(h * N + q_block, BQ), :],
          out_sem.at[slot, 2]),
  )


def _direct_head_loop(
    h, *, rank, qref, kref, vref, qid_ref, kvid_ref, qseq_ref,
    ofull, lse, max_logits, active_rows_ref, active_cols_ref,
    mask_next_ref, bounds_start_ref, bounds_end_ref, block_mask_ref,
    m_scratch, l_scratch, o_scratch,
    q_tile, k_tile, v_tile, qid_tile, kvid_tile, qseq_tile, in_sem,
    o_tile, lse_tile, max_tile, out_sem,
    splash_kwargs,
):
  """Run one head's active blocks with double-buffered input/output DMA."""
  del rank
  active_count = active_rows_ref.shape[0]
  in_kw = dict(qref=qref, kref=kref, vref=vref, qid_ref=qid_ref,
               kvid_ref=kvid_ref, qseq_ref=qseq_ref,
               active_rows_ref=active_rows_ref,
               active_cols_ref=active_cols_ref, q_tile=q_tile, k_tile=k_tile,
               v_tile=v_tile, qid_tile=qid_tile, kvid_tile=kvid_tile,
               qseq_tile=qseq_tile, in_sem=in_sem)
  out_kw = dict(ofull=ofull, lse=lse, max_logits=max_logits, o_tile=o_tile,
                lse_tile=lse_tile, max_tile=max_tile, out_sem=out_sem)

  # Prologue: block 0 goes to input slot 0.
  for copy in _in_copies(h, 0, 0, **in_kw):
    copy.start()

  def body(grid_idx, out_count):
    slot = grid_idx % 2

    # Prefetch block g+1 into the other slot; that slot's previous block
    # (g-1) has already been consumed by the sequential compute.
    @pl.when(grid_idx + 1 < active_count)
    def prefetch():
      for copy in _in_copies(h, grid_idx + 1, 1 - slot, **in_kw):
        copy.start()

    for copy in _in_copies(h, grid_idx, slot, **in_kw):
      copy.wait()

    write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    oslot = out_count % 2

    # Before the body stores into an output slot, drain the writeback that
    # used the same slot two rows ago.  Byte counts match any q_block.
    @pl.when(jnp.logical_and(write, out_count >= 2))
    def drain_slot():
      for copy in _out_copies(h, 0, oslot, **out_kw):
        copy.wait()

    sk.flash_attention_kernel(
        active_rows_ref, active_cols_ref, mask_next_ref,
        bounds_start_ref, bounds_end_ref, block_mask_ref,
        q_tile.at[slot], k_tile.at[slot], v_tile.at[slot],
        qid_tile.at[slot], kvid_tile.at[slot], None, None,
        qseq_tile.at[slot], None,
        o_tile.at[oslot], lse_tile.at[oslot], None, max_tile.at[oslot],
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

    # Start the row writeback and do not wait; the next row computes in the
    # other output slot meanwhile.
    @pl.when(write)
    def store_output():
      for copy in _out_copies(h, active_rows_ref[grid_idx] * BQ, oslot,
                              **out_kw):
        copy.start()

    return out_count + write.astype(jnp.int32)

  out_count = lax.fori_loop(0, active_count, body, jnp.int32(0))

  # Epilogue: drain the last two writebacks so ``ofull`` is complete before
  # the caller starts this head's O a2a.
  @pl.when(out_count >= 1)
  def drain_last():
    for copy in _out_copies(h, 0, (out_count - 1) % 2, **out_kw):
      copy.wait()

  @pl.when(out_count >= 2)
  def drain_second_last():
    for copy in _out_copies(h, 0, out_count % 2, **out_kw):
      copy.wait()


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
    o_tile, lse_tile, max_tile = outs[16:19]

    barrier_sem = pltpu.get_barrier_semaphore()
    for step in range(1, CP_SIZE):
      dst = (rank_ref[0] + step) % CP_SIZE
      pl.semaphore_signal(
          barrier_sem, inc=1, device_id={"context": dst},
          device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier_sem, CP_SIZE - 1)

    def scope(q_send, q_recv_sem, o_send, o_recv_sem, local_sem,
              in_sem, out_sem):
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
            in_sem=in_sem, o_tile=o_tile, lse_tile=lse_tile,
            max_tile=max_tile, out_sem=out_sem, splash_kwargs=splash_kwargs)
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
        pltpu.SemaphoreType.DMA((2, 6)),
        pltpu.SemaphoreType.DMA((2, 3)),
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
              pltpu.VMEM((2, QK, BQ), jnp.bfloat16),
              pltpu.VMEM((2, QK, BKV), jnp.bfloat16),
              pltpu.VMEM((2, BKV, V), jnp.bfloat16),
              pltpu.VMEM((2, BQ, NUM_LANES), jnp.int32),
              pltpu.VMEM((2, NUM_SUBLANES, BKV), jnp.int32),
              pltpu.VMEM((2, BQ, NUM_LANES), jnp.int32),
              pltpu.VMEM((2, BQ, V), jnp.bfloat16),
              pltpu.VMEM((2, BQ, NUM_LANES), jnp.float32),
              pltpu.VMEM((2, BQ, NUM_LANES), jnp.float32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(), collective_id=0,
          vmem_limit_bytes=56 * 1024 * 1024,
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
