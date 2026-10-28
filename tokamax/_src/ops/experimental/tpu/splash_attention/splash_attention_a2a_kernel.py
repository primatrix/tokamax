"""CP8 Splash forward with head-overlap Ulysses A2A.

This is deliberately a new candidate.  It does not import any of the V6j
compute kernels.  The only attention implementation used here is the pinned
Tokamax ``flash_attention_kernel`` body; the outer pallas call owns the
sequence-shard A2A and invokes the body through the nested emit_pipeline API.

Each local head owns a complete receive area. Incoming(h+1) is started
before the Splash pipeline for head h, and outgoing(h) is started as soon as
head h finishes. Independent per-head DMA semaphores make those transfers
safe without a bank/credit protocol.
"""
from __future__ import annotations

import dataclasses
import functools
import json
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
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as sm


NUM_LANES = 128
NUM_SUBLANES = 8
H_LOCAL = 8
H_GLOBAL = 64
CP_SIZE = 8
N = 262144
T = N // CP_SIZE
QK = 192
V = 128
BQ = 2048
BKV = 2048
BKV_COMPUTE = 512
MASK_VALUE = -0.7 * jnp.finfo(jnp.float32).max


def _splash_config(config: dict[str, Any]):
  """Construct the exact production SplashConfig from the resolved manifest."""
  values = dict(config["splash"])
  qkv = sk.QKVLayout
  for name in ("q_layout", "k_layout", "v_layout"):
    value = values.get(name)
    if isinstance(value, str):
      values[name] = qkv[value]
  values.update(
      block_q=BQ,
      block_kv=BKV,
      block_kv_compute=BKV_COMPUTE,
      softmax_scale=0.07216878364870322,
      use_base2_exp=False,
      fuse_reciprocal=True,
      compact_stats_output=False,
      segment_mask_on_partial_only=False,
      max_logit_const=None,
      attn_logits_soft_cap=None,
      use_experimental_scheduler=True,
  )
  return sk.SplashConfig(**values)


def _whole_spec(shape, *, memory_space=pltpu.HBM):
  rank = len(shape)
  return pl.BlockSpec(shape, lambda *idx: (0,) * rank, memory_space=memory_space)


def _copy(src, dst, sem):
  return pltpu.make_async_copy(src, dst, sem)


def _remote(src, dst, send_sem, recv_sem, destination, *, mesh_axis):
  # A mesh-index destination is required by the production CP path.  The AOT
  # wrapper uses the same eight-device ``context`` mesh as the CP contract.
  return pltpu.make_async_remote_copy(
      src, dst, send_sem, recv_sem,
      device_id={mesh_axis: destination},
      device_id_type=pl.DeviceIdType.MESH,
  )


def _build_mask(config):
  mask = sm.CausalMask((N, N))
  kernel = sk.make_splash_mha(
      mask,
      config=_splash_config(config),
      save_residuals=True,
      q_seq_shards=1,
  )
  return kernel.fwd_mask_info, kernel.kwargs


def _pipeline_for_head(h: int, *, mask_info, splash_kwargs, qref, kref, vref,
                       qids, kvids, qseq, out, lse, max_logits,
                       metadata_refs, m_scratch, l_scratch, o_scratch):
  """Emit the unmodified production Splash body for one static local head."""
  bq = BQ
  bkv = BKV
  bkv_compute = BKV_COMPUTE
  cfg = splash_kwargs["config"]
  (active_rows_ref, active_cols_ref, mask_next_ref, bounds_start_ref,
   bounds_end_ref, block_mask_ref) = metadata_refs

  # BlockSpec index maps return block indices, not element offsets.  The
  # active-row/column tables already contain Q/K block numbers, so do not
  # multiply them by BQ/BKV.  Tuple order follows each physical ref layout:
  # Q/K are [D, sequence], V/ids/outputs/stats are [sequence, feature], and
  # kv ids are [lane, sequence].
  def q_start(_h, grid_idx, *_):
    return (0, active_rows_ref[grid_idx])

  def k_start(_h, grid_idx, *_):
    return (0, active_cols_ref[grid_idx])

  def v_start(_h, grid_idx, *_):
    return (active_cols_ref[grid_idx], 0)

  def qid_start(_h, grid_idx, *_):
    return (active_rows_ref[grid_idx], 0)

  def kid_start(_h, grid_idx, *_):
    return (0, active_cols_ref[grid_idx])

  def body(q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref,
           q_sequence_ref, o_ref, logsumexp_ref, max_logits_ref,
           m_scratch_ref, l_scratch_ref, o_scratch_ref):
    # Keep optional Splash operands as Python None.  emit_pipeline does not
    # accept None BlockSpecs, so this adapter keeps the exact ordinary body
    # while exposing only the live causal/segmented ABI to the nested grid.
    sk.flash_attention_kernel(
        active_rows_ref, active_cols_ref, mask_next_ref,
        bounds_start_ref, bounds_end_ref, block_mask_ref,
        q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref,
        None, None, q_sequence_ref, None,
        o_ref, logsumexp_ref, None, max_logits_ref,
        m_scratch_ref, l_scratch_ref, o_scratch_ref,
        mask_value=splash_kwargs["mask_value"], kv_steps=N // BKV,
        bq=bq, bkv=bkv, bkv_compute=bkv_compute, head_dim_v=V,
        fuse_reciprocal=splash_kwargs["config"].fuse_reciprocal,
        config=cfg, mask_function=splash_kwargs["mask_function"],
        native_layout=False,
    )
  # The refs are full receive/output arrays.  The static h is captured in the
  # surrounding Python loop, while the nested program's head id is ignored.
  in_specs = (
      pl.BlockSpec((QK, BQ), q_start),
      pl.BlockSpec((QK, BKV), k_start),
      pl.BlockSpec((BKV, V), v_start),
      pl.BlockSpec((BQ, NUM_LANES), qid_start),
      pl.BlockSpec((NUM_SUBLANES, BKV), kid_start),
      pl.BlockSpec((BQ, NUM_LANES), qid_start),
  )
  out_specs = (
      pl.BlockSpec((BQ, V), qid_start),
      pl.BlockSpec((BQ, NUM_LANES), qid_start),
      pl.BlockSpec((BQ, NUM_LANES), qid_start),
  )
  pipeline = pltpu.emit_pipeline(
      body,
      grid=(1, active_rows_ref.shape[0]),
      in_specs=in_specs,
      out_specs=out_specs,
      dimension_semantics=("parallel", "arbitrary"),
      trace_scopes=False,
  )
  pipeline(
      qref, kref, vref, qids, kvids, qseq,
      out, lse, max_logits,
      scratches=(m_scratch, l_scratch, o_scratch),
  )


def _start_a2a_head(rank, h, qsrc, ksrc, vsrc, qrecv, krecv, vrecv,
                    q_send, q_recv, local_sem, *, mesh_axis):
  """Start Q/K/V transfers for one head and return their live descriptors.

  The descriptors remain live while the preceding head's Splash pipeline runs.
  Receive descriptors are constructed here but only wait in
  ``_wait_a2a_head``; this preserves the semaphore key and lets the next
  head's incoming traffic overlap compute.
  """
  # All peers receive a full sequence tile.  The target is a complete per-head
  # [D,N] / [N,V] region, so no bank is reused in this serial Step-B probe.
  local_q = _copy(
      qsrc.at[pl.ds((rank * H_LOCAL + h) * QK, QK), :],
      qrecv.at[pl.ds(h * QK, QK), pl.ds(rank * T, T)],
      local_sem.at[h, 0])
  local_k = _copy(
      ksrc.at[pl.ds((rank * H_LOCAL + h) * QK, QK), :],
      krecv.at[pl.ds(h * QK, QK), pl.ds(rank * T, T)],
      local_sem.at[h, 1])
  local_v = _copy(
      vsrc.at[pl.ds((rank * H_LOCAL + h) * T, T), :],
      vrecv.at[pl.ds(h * N + rank * T, T), :], local_sem.at[h, 2])
  for d in (local_q, local_k, local_v):
    d.start()
  sends = []
  for step in range(1, CP_SIZE):
    dst = (rank + step) % CP_SIZE
    sends.extend([
        _remote(qsrc.at[pl.ds((dst * H_LOCAL + h) * QK, QK), :],
                qrecv.at[pl.ds(h * QK, QK), pl.ds(rank * T, T)],
                q_send.at[h, step, 0], q_recv.at[h, step, 0], dst,
                mesh_axis=mesh_axis),
        _remote(ksrc.at[pl.ds((dst * H_LOCAL + h) * QK, QK), :],
                krecv.at[pl.ds(h * QK, QK), pl.ds(rank * T, T)],
                q_send.at[h, step, 1], q_recv.at[h, step, 1], dst,
                mesh_axis=mesh_axis),
        _remote(vsrc.at[pl.ds((dst * H_LOCAL + h) * T, T), :],
                vrecv.at[pl.ds(h * N + rank * T, T), :],
                q_send.at[h, step, 2], q_recv.at[h, step, 2], dst,
                mesh_axis=mesh_axis),
    ])
  for d in sends:
    d.start()

  # The receive descriptor mirrors hbm_transport: step indexes the sender's
  # semaphore slot, src selects the sequence region on this rank, and the
  # source ref is shape-only for wait_recv.
  receives = []
  for step in range(1, CP_SIZE):
    dst = (rank + step) % CP_SIZE
    src = (rank - step + CP_SIZE) % CP_SIZE
    for stream, width in enumerate((QK, QK, V)):
      dummy = qsrc.at[0:width, :] if stream < 2 else vsrc.at[0:T, :]
      target = (
          qrecv.at[pl.ds(h * QK, QK), pl.ds(src * T, T)] if stream == 0
          else krecv.at[pl.ds(h * QK, QK), pl.ds(src * T, T)] if stream == 1
          else vrecv.at[pl.ds(h * N + src * T, T), :]
      )
      receives.append(_remote(
          dummy, target, q_send.at[h, step, stream],
          q_recv.at[h, step, stream], dst, mesh_axis=mesh_axis))
  return (local_q, local_k, local_v), tuple(sends), tuple(receives)


def _wait_a2a_head(pending):
  """Wait a previously-started Q/K/V transfer set."""
  locals_, sends, receives = pending
  for d in locals_:
    d.wait()
  for d in sends:
    d.wait_send()
  for d in receives:
    d.wait_recv()


def _start_o_head(rank, h, ofull, oseq, o_send, o_recv, local_sem,
                  *, mesh_axis):
  """Start O writeback for one head and return live send/receive descriptors."""
  # oseq is the post-A2A sequence-shard output with all global heads.  The
  # source rank's computed head occupies one global head column on every
  # destination rank; only the sequence slice varies per transfer.
  head_col = pl.ds((rank * H_LOCAL + h) * V, V)
  local = _copy(ofull.at[pl.ds(h * N + rank * T, T), :],
                oseq.at[:, head_col], local_sem.at[h, 3])
  local.start()
  sends = []
  for step in range(1, CP_SIZE):
    dst = (rank + step) % CP_SIZE
    sends.append(_remote(
        ofull.at[pl.ds(h * N + dst * T, T), :],
        oseq.at[:, head_col], o_send.at[h, step], o_recv.at[h, step], dst,
        mesh_axis=mesh_axis))
  for d in sends:
    d.start()
  receives = []
  for step in range(1, CP_SIZE):
    dst = (rank + step) % CP_SIZE
    src = (rank - step + CP_SIZE) % CP_SIZE
    recv_col = pl.ds((src * H_LOCAL + h) * V, V)
    receives.append(_remote(
        ofull.at[0:T, :], oseq.at[:, recv_col],
        o_send.at[h, step], o_recv.at[h, step], dst,
        mesh_axis=mesh_axis))
  return local, tuple(sends), tuple(receives)


def _wait_o_head(pending):
  """Wait a previously-started O writeback set."""
  local, sends, receives = pending
  local.wait()
  for d in sends:
    d.wait_send()
  for d in receives:
    d.wait_recv()


def build(config: dict[str, Any], mesh):
  """Return the eight-device shard-mapped Step-C module and global specs.

  The outer ``shard_map`` supplies each rank's sequence shard and makes
  ``lax.axis_index(mesh_axis)`` available to the single inner pallas call.
  Rank is therefore not a scalar-prefetch operand and is never hard-coded to
  zero.  Q/K/V inputs retain all global head columns on every sequence shard;
  the serial descriptors select the destination-owned head for each peer.
  """
  if config.get("cp_size") != CP_SIZE or config.get("global_heads") != H_GLOBAL:
    raise ValueError("the fused Splash transport currently supports CP8/H=64 only")
  if config.get("tokens") != N or config.get("heads") != H_LOCAL:
    raise ValueError("the fused Splash transport currently supports N=262144/H_local=8 only")
  splash_values = config.get("splash", {})
  if tuple(splash_values.get(x) for x in ("block_q", "block_kv", "block_kv_compute")) != (
      BQ, BKV, BKV_COMPUTE
  ):
    raise ValueError("fused Splash requires block_q/block_kv/block_kv_compute=2048/2048/512")
  mask_info, splash_kwargs = _build_mask(config)
  # Mosaic memref_slice dynamic offsets must be signless i32.  Tokamax's
  # compact block-list arrays can be i8 in this path, so widen only the scalar
  # metadata refs; the Splash mask/data dtypes remain unchanged.
  active_rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
  active_cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
  block_mask = jnp.asarray(mask_info.block_mask)
  row_break = active_rows[1:] != active_rows[:-1]
  # Splash's find_bounds emits int32 metadata.  The body converts these
  # refs to bool at the use site; preserve that ABI instead of passing bool
  # arrays through the scalar-prefetch path.
  row_break_i32 = row_break.astype(jnp.int32)
  bounds_start = jnp.concatenate((jnp.asarray([1], dtype=jnp.int32),
                                  row_break_i32))
  bounds_end = jnp.concatenate((row_break_i32,
                                jnp.asarray([1], dtype=jnp.int32)))
  def _make_candidate():

    def kernel_body(rank_ref, active_rows_ref, active_cols_ref, mask_next_ref,
                    bounds_start_ref, bounds_end_ref, block_mask_ref,
                    qref, kref, vref, qid_ref, kvid_ref, qseq_ref, *outs):
      qrecv, krecv, vrecv, ofull, oseq, lse, max_logits = outs[:7]
      m_scratch, l_scratch, o_scratch = outs[7:10]
      # Local DMA/regular semaphores are allocated by run_scoped only after
      # all peers have entered this kernel.  Use the globally allocated
      # barrier semaphore once before the first peer write.  The compiler
      # collective_id selects the barrier pool for this pallas_call.
      barrier_sem = pltpu.get_barrier_semaphore()
      for step in range(1, CP_SIZE):
        dst = (rank_ref[0] + step) % CP_SIZE
        pl.semaphore_signal(
            barrier_sem, inc=1, device_id={axis: dst},
            device_id_type=pl.DeviceIdType.MESH)
      pl.semaphore_wait(barrier_sem, CP_SIZE - 1)

      # Metadata is immutable and replicated in this first probe.  Incoming
      # head 0 is the only prologue transfer: it is started and waited after
      # the entry barrier.  Every later incoming head is started before the
      # preceding head's pipeline and waited immediately before its own
      # pipeline.  O(h) is started after compute(h), then drained after the
      # next head's pipeline, so both directions overlap the body without
      # banks or credits.
      def scope(q_send, q_recv, o_send, o_recv, local_sem):
        pending_in = _start_a2a_head(
            rank_ref[0], 0, qref, kref, vref, qrecv, krecv, vrecv,
            q_send, q_recv, local_sem, mesh_axis=axis)
        _wait_a2a_head(pending_in)
        pending_in = _start_a2a_head(
            rank_ref[0], 1, qref, kref, vref, qrecv, krecv, vrecv,
            q_send, q_recv, local_sem, mesh_axis=axis)
        pending_o = None
        for h in range(H_LOCAL):
          if h > 0:
            _wait_a2a_head(pending_in)
            if h + 1 < H_LOCAL:
              pending_in = _start_a2a_head(
                  rank_ref[0], h + 1, qref, kref, vref, qrecv, krecv,
                  vrecv, q_send, q_recv, local_sem, mesh_axis=axis)
            else:
              pending_in = None
          _pipeline_for_head(
              h, mask_info=mask_info, splash_kwargs=splash_kwargs,
              qref=qrecv.at[pl.ds(h * QK, QK), :],
              kref=krecv.at[pl.ds(h * QK, QK), :],
              vref=vrecv.at[pl.ds(h * N, N), :], qids=qid_ref,
              kvids=kvid_ref, qseq=qseq_ref,
              out=ofull.at[pl.ds(h * N, N), :],
              lse=lse.at[pl.ds(h * N, N), :],
              max_logits=max_logits.at[pl.ds(h * N, N), :],
              metadata_refs=(active_rows_ref, active_cols_ref, mask_next_ref,
                             bounds_start_ref, bounds_end_ref, block_mask_ref),
              m_scratch=m_scratch, l_scratch=l_scratch, o_scratch=o_scratch)
          if pending_o is not None:
            _wait_o_head(pending_o)
          pending_o = _start_o_head(
              rank_ref[0], h, ofull, oseq, o_send, o_recv, local_sem,
              mesh_axis=axis)
        _wait_o_head(pending_o)

      # Semaphores are separate for every head and stream.  There are no
      # credits/banks: the independent receive areas make the delayed waits
      # safe while incoming(h+1) and outgoing(h-1) are in flight together.
      pl.run_scoped(
          scope,
          pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE, 3)),
          pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE, 3)),
          pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE)),
          pltpu.SemaphoreType.DMA((H_LOCAL, CP_SIZE)),
          pltpu.SemaphoreType.DMA((H_LOCAL, 4)),
      )

    # Rank is produced by lax.axis_index in the enclosing shard_map and passed
    # as the first one-element scalar-prefetch ref, exactly as the accepted
    # hbm_forward AOT path does.  It is not hard-coded to rank zero.
    out_shapes = (
        # Receive buffers are flattened so dynamic rank/head offsets are
        # expressed as byte-aligned column/row ds slices.
        jax.ShapeDtypeStruct((H_LOCAL * QK, N), jnp.bfloat16),
        jax.ShapeDtypeStruct((H_LOCAL * QK, N), jnp.bfloat16),
        jax.ShapeDtypeStruct((H_LOCAL * N, V), jnp.bfloat16),
        jax.ShapeDtypeStruct((H_LOCAL * N, V), jnp.bfloat16),
        # The post-A2A sequence shard carries all global head columns.
        jax.ShapeDtypeStruct((T, H_GLOBAL * V), jnp.bfloat16),
        jax.ShapeDtypeStruct((H_LOCAL * N, NUM_LANES), jnp.float32),
        jax.ShapeDtypeStruct((H_LOCAL * N, NUM_LANES), jnp.float32),
    )
    # q/k/v follow six scalar-prefetched Splash metadata refs.  Rank is the
    # first scalar-prefetched input produced by axis_index inside shard_map;
    # it is not captured by the kernel closure.
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
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(),
            vmem_limit_bytes=32 * 1024 * 1024,
            collective_id=0,
        ),
        name="v7a_step_c_head_overlap_a2a_splash512",
    )
    # Return only the public O/stats ABI.  The receive/output arrays are kept
    # as pallas outputs so XLA cannot DCE the DMA targets; the wrapper selects
    # O/lse/max after the single custom call.
    def wrapped(rank_value, q, k, v, qids_in, kvids_in, qseq_in):
      # Sequence-shard ABI is [T,H_global,D] / [T,H_global,V].  Flattening
      # the head and feature axes is a bitcast-style layout view for DMA;
      # it does not change the element order or introduce a head copy.
      qsrc = q.transpose(1, 2, 0).reshape(H_GLOBAL * QK, T)
      ksrc = k.transpose(1, 2, 0).reshape(H_GLOBAL * QK, T)
      vsrc = v.transpose(1, 0, 2).reshape(H_GLOBAL * T, V)
      outputs = call(rank_value, active_rows, active_cols,
                     jnp.zeros_like(active_rows),
                     bounds_start, bounds_end,
                     block_mask, qsrc, ksrc, vsrc, qids_in,
                     kvids_in, qseq_in)
      return (outputs[4].reshape(T, H_GLOBAL, V),
              outputs[5].reshape(H_LOCAL, N, NUM_LANES),
              outputs[6].reshape(H_LOCAL, N, NUM_LANES))

    return wrapped

  axis = config["mesh_axis"]
  if axis != "context":
    raise ValueError(f"V7a Step-C requires the production mesh axis 'context', got {axis!r}")

  def local(q, k, v, qids_in, kvids_in, qseq_in):
    rank_value = lax.axis_index(axis).reshape((1,))
    wrapped = _make_candidate()
    return wrapped(rank_value, q, k, v, qids_in, kvids_in, qseq_in)

  mapped = jax.shard_map(
      local, mesh=mesh,
      in_specs=(P(axis, None, None),) * 3 + (P(None, None),) * 3,
      out_specs=(P(axis, None, None),) * 3,
      check_vma=False)
  specs = (
      jax.ShapeDtypeStruct((N, H_GLOBAL, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, H_GLOBAL, QK), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, H_GLOBAL, V), jnp.bfloat16),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
      jax.ShapeDtypeStruct((NUM_SUBLANES, N), jnp.int32),
      jax.ShapeDtypeStruct((N, NUM_LANES), jnp.int32),
  )
  metadata = {
      "expected_kernel_name": "v7a_step_c_head_overlap_a2a_splash512",
      "custom_call_count_expected": 1,
      "a2a": "head-overlap full-head Q/K/V and O; independent per-head semaphores; no bank/credit",
      "rank_source": "lax.axis_index(mesh_axis) inside shard_map",
      "shape": {"H_local": H_LOCAL, "H_global": H_GLOBAL, "N": N,
                "T": T, "qk": QK, "v": V},
      "layout": {"q": "[H_global*192,T] flattened SEQ_MINOR",
                  "k": "[H_global*192,T] flattened SEQ_MINOR",
                  "v": "[H_global*T,128] flattened HEAD_DIM_MINOR",
                  "o": "[T,H_global*128] sequence-sharded"},
      "splash": {"block_q": BQ, "block_kv": BKV,
                  "block_kv_compute": BKV_COMPUTE,
                  "use_base2_exp": False, "save_residuals": True},
      "note": "candidate is Step-C head-overlap; not device-run and not yet review-approved",
  }
  return jax.jit(mapped), specs, metadata


make_splash_mha_a2a = build
