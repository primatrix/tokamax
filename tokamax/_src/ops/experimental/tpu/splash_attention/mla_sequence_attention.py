"""Sequence-layout custom VJP for the qualified MLA dKV export kernel.

The factory accepts the caller's complete Splash kernel/configuration; it does
not import benchmark state, change its tile/cost configuration, or install a
remat policy. The caller remains responsible for choosing checkpoint policies.
The fused path is restricted to the validated static causal geometry; callers
retain their existing implementation for other masks and shapes.
"""

from __future__ import annotations

import functools
import math
import numpy as np

import jax
import jax.numpy as jnp
from jax import ad_checkpoint, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.layout import Layout, with_layout_constraint
from jax.sharding import NamedSharding, PartitionSpec as P

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk

from tokamax._src.ops.experimental.tpu.splash_attention.mla_fused_backward import make_backward, validate_source


def _forward_output_kernel(
    source_ref,
    output_ref,
    stage_ref,
    send_ref,
    recv_ref,
    *,
    cp,
    axis_name,
    tile,
    tiles_per_peer,
):
  """Redistribute completed head-layout O tiles with remote HBM DMA.

  A source rank owns four heads and sends each global sequence tile to the
  one context peer that owns that sequence range. The output buffer is full
  on every rank because remote writes address all global heads. The final
  program drains local sends and receives before the call returns; callers
  can therefore consume the sequence-layout output without an external
  all-to-all. This is deliberately separate from Splash arithmetic until the
  device protocol and timing are qualified.
  """
  head, global_tile = pl.program_id(0), pl.program_id(1)
  sequence_tiles = pl.num_programs(1)
  sender = lax.axis_index(axis_name)
  peer = lax.div(global_tile, tiles_per_peer)
  local_tile = lax.rem(global_tile, tiles_per_peer)
  transfer = pltpu.make_async_remote_copy(
      source_ref.at[head, pl.ds(global_tile * tile, tile), :],
      # Integer indexing removes the head axis.  A size-one ``pl.ds`` slice
      # leaves a rank-3 view and is rejected by the CPU Pallas interpreter for
      # a dynamic remote destination.
      output_ref.at[sender * 4 + head, pl.ds(local_tile * tile, tile), :],
      send_ref.at[global_tile],
      recv_ref.at[sender, head, local_tile],
      device_id={axis_name: peer},
  )
  transfer.start()

  @pl.when(jnp.logical_and(head == 3, global_tile == sequence_tiles - 1))
  def drain():
    for source_head in range(4):
      for index in range(sequence_tiles):
        peer_index = lax.div(index, tiles_per_peer)
        local_index = lax.rem(index, tiles_per_peer)
        pltpu.make_async_remote_copy(
            source_ref.at[source_head, pl.ds(index * tile, tile), :],
            output_ref.at[sender * 4 + source_head, pl.ds(local_index * tile, tile), :],
            send_ref.at[index],
            recv_ref.at[sender, source_head, local_index],
            device_id={axis_name: peer_index},
        ).wait_send()
    for source in range(cp):
      for source_head in range(4):
        for local in range(tiles_per_peer):
          pltpu.make_async_copy(
              stage_ref,
              output_ref.at[source * 4 + source_head, pl.ds(local * tile, tile), :],
              recv_ref.at[source, source_head, local],
          ).wait()


def _forward_output_exchange(x, *, cp, axis_name):
  """Move `[4, S, D]` head-layout output to `[CP*4, S/CP, D]`."""
  if cp not in (8, 16):
    raise ValueError("Forward output exchange is qualified only for CP8/CP16")
  if x.ndim != 3 or x.shape[0] != 4 or x.shape[-1] != 128 or x.shape[1] % cp or x.dtype != jnp.bfloat16:
    raise ValueError("Forward output exchange requires BF16 [4, S, 128] with S divisible by CP")
  tile = 4096
  if x.shape[1] % tile or (x.shape[1] // tile) % cp:
    raise ValueError("Forward output exchange requires 4096-aligned complete CP tiles")
  sequence_tiles = x.shape[1] // tile
  tiles_per_peer = sequence_tiles // cp
  output_shape = (cp * 4, x.shape[1] // cp, x.shape[2])
  return pl.pallas_call(
      functools.partial(
          _forward_output_kernel,
          cp=cp,
          axis_name=axis_name,
          tile=tile,
          tiles_per_peer=tiles_per_peer,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          grid=(4, sequence_tiles),
          scratch_shapes=[
              pltpu.VMEM((tile, x.shape[-1]), x.dtype),
              pltpu.SemaphoreType.DMA((sequence_tiles,)),
              pltpu.SemaphoreType.DMA((cp, 4, tiles_per_peer)),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          skip_device_barrier=False,
      ),
      name="mla_fwd_output_remote_a2a",
  )(x)


def _remote_input_exchange(x, *, cp, axis_name, name):
  """Redistribute sequence-sharded heads with direct remote HBM writes.

  ``x`` is the sequence-layout source on one context rank:
  ``[CP * 4, S / CP, width]``.  Each rank sends its four-head slice for every
  destination head group to that group's owner.  The destination therefore
  receives ``[4, S, width]`` without an XLA all-to-all.  The call is kept
  phase-local: forward and backward use separate wrappers and selectors even
  though the byte-level protocol is identical.

  The remote copies are drained before returning.  This makes the output safe
  for the following Splash call while keeping all communication ownership in
  the phase that requested it.
  """
  if cp not in (8, 16):
    raise ValueError("Remote input exchange is qualified only for CP8/CP16")
  if x.ndim != 3 or x.shape[0] != cp * 4 or x.shape[1] % 4096 or x.dtype != jnp.bfloat16 or x.shape[2] <= 0:
    raise ValueError("Remote input exchange requires BF16 [CP*4,S/CP,width] with 4096-aligned length")
  tile = 4096
  length = x.shape[1]
  tiles = length // tile

  def kernel(source_ref, output_ref, send_ref, recv_ref):
    head, local_tile = pl.program_id(0), pl.program_id(1)
    sender = lax.axis_index(axis_name)
    source_offset = local_tile * tile
    target_offset = sender * length + source_offset

    # @reference.begin mla_input_a2a_mapping
    # Reference: ``to_head`` below, i.e. transpose BSHD -> BHSD followed by
    # ``lax.all_to_all(split_axis=1, concat_axis=2, tiled=True)``.  For every
    # source rank ``sender``, destination head group ``peer`` and local tile
    # ``local_tile``, source[peer*4+head, local_tile*4096:,...] maps to the
    # destination's output[head, sender*length+local_tile*4096:,...].
    def copy_for_peer(peer):
      return pltpu.make_async_remote_copy(
          source_ref.at[peer * 4 + head, pl.ds(source_offset, tile), :],
          output_ref.at[head, pl.ds(target_offset, tile), :],
          send_ref.at[head, local_tile, peer],
          recv_ref.at[sender, head, local_tile],
          device_id={axis_name: peer},
      )

    # @reference.end mla_input_a2a_mapping

    def start_peer(peer, _):
      copy_for_peer(peer).start()

    def wait_peer(peer, _):
      copy_for_peer(peer).wait_send()

    lax.fori_loop(0, cp, start_peer, None, unroll=False)
    lax.fori_loop(0, cp, wait_peer, None, unroll=False)

    # One invocation drains every receive on this destination rank.  The
    # descriptor is wait-only; the remote DMA already wrote output_ref.
    @pl.when(jnp.logical_and(head == 3, local_tile == tiles - 1))
    def drain():
      for source in range(cp):
        for source_head in range(4):
          for source_tile in range(tiles):
            pltpu.make_async_copy(
                source_ref.at[source_head, pl.ds(0, tile), :],
                output_ref.at[source_head, pl.ds(source * length + source_tile * tile, tile), :],
                recv_ref.at[source, source_head, source_tile],
            ).wait()

  return pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          grid=(4, tiles),
          scratch_shapes=[
              pltpu.SemaphoreType.DMA((4, tiles, cp)),
              pltpu.SemaphoreType.DMA((cp, 4, tiles)),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct((4, cp * length, x.shape[-1]), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          skip_device_barrier=False,
      ),
      name=name,
  )(x)


def _forward_input_exchange(x, *, cp, axis_name):
  """Move packed forward Q/K/V from sequence to head layout."""
  return _remote_input_exchange(x, cp=cp, axis_name=axis_name, name="mla_fwd_input_remote_a2a")


def _backward_input_exchange(x, *, cp, axis_name):
  """Move packed backward Q/K/V/dO from sequence to head layout."""
  return _remote_input_exchange(x, cp=cp, axis_name=axis_name, name="mla_bwd_input_remote_a2a")


def _exchange_head(x, head, cp, axis_name):
  """Exchange one head from a local [CP*4,L,D] source into [1,S,D]."""
  if x.ndim != 3 or x.shape[0] != cp * 4 or not 0 <= head < 4:
    raise ValueError("Per-head exchange requires [CP*4,L,D] and a head in [0,4)")
  return lax.all_to_all(
      x.reshape(cp, 4, x.shape[1], x.shape[2])[:, head],
      axis_name,
      split_axis=0,
      concat_axis=1,
      tiled=True,
  )


def _merge_head_gradients(parts):
  """Restore sender-major, head-minor rows from four fresh [CP,L,D] outputs."""
  if len(parts) != 4 or any(x.shape != parts[0].shape for x in parts) or parts[0].ndim != 3:
    raise ValueError("Gradient assembly requires four matching [CP,L,D] outputs")
  cp, length, width = parts[0].shape
  return jnp.stack(parts, axis=1).reshape(cp * 4, length, width)


def make_static_splash_kernel(mask, config):
  """Keep pinned Tokamax mask metadata as NumPy constants for topology AOT."""
  validate_source()
  # Same processor/defaults as Tokamax's factory, without its eager jnp.array
  # conversion. That conversion executes on an enclosing topology-only mesh
  # even under default_device(cpu). NumPy leaves become constants when the
  # attention call is traced, and remain concrete for ownership validation.
  common = dict(downcast_smem_data=True, partial_mask_blocks_dtype=np.int8, q_seq_shards=1)
  fwd, mask_function = sk.mask_info_lib.process_mask(mask, (config.block_q, config.block_kv), **common)
  dkv, backward_mask_function = sk.mask_info_lib.process_mask(
      mask,
      (config.block_q_dkv, config.block_kv_dkv),
      is_dkv=True,
      return_dynamic_grid=config.dq_reduction_steps == 3,
      **common,
  )
  assert (mask_function is None) == (backward_mask_function is None)
  return sk.SplashAttentionKernel(
      fwd,
      dkv,
      config=config,
      is_mqa=False,
      save_residuals=False,
      mask_value=sk.base.DEFAULT_MASK_VALUE,
      mask_function=mask_function,
      fwd_mask_sparsity=float(np.mean(fwd.block_mask != 0)),
      dkv_mask_sparsity=float(np.mean(dkv.block_mask != 0)),
  )


def _validate_kernel(kernel, cp):
  cfg = kernel.kwargs["config"]
  if cp not in (8, 16):
    raise ValueError("Qualified CP sizes are 8 and 16")
  if kernel.kwargs["is_mqa"]:
    raise ValueError("MLA fused export requires MHA")
  # The public forward converts base2 LSE to natural log, while native bwd
  # consumes the configured base. The qualified production config is base e.
  if cfg.use_base2_exp:
    raise ValueError("Base2 LSE conversion has not been qualified")
  if cfg.attn_logits_soft_cap is not None:
    raise ValueError("Soft-cap support has not been qualified")
  if not cfg.bwd_cast_before_transpose:
    raise ValueError("Pass the explicitly selected cast-before-transpose kernel")
  if cfg.bwd_scale_after_dot or cfg.bwd_dq_first:
    raise ValueError("Only the confirmed backward arithmetic configuration is qualified")
  return cfg


def make_local_attention(
    kernel,
    *,
    cp,
    axis_name="context",
    fused=True,
    record_max_logits=False,
    evidence_dir=None,
    cp_barrier=False,
    bwd_input_exchange="remat",
    fwd_input_exchange="external",
    fwd_output_exchange="external",
    _return_pullback=False,
):
  """Map local BSHD sequence shards to the same layout and gradient ownership.

  Global H is CP*4; a rank owns all heads for S/CP tokens at the boundary.
  Kernel inputs are H4/Sfull. Runtime segment IDs enter as sequence shards.
  The local batch is exactly one, as in the production FSDP32 configuration.
  Experimental backward input exchanges require the original remat policy:
  head Q/K/V are recomputed, while Splash output/LSE may be retained. Head
  checkpoint tags do not retain experimental backward inputs.
  """
  if bwd_input_exchange not in (
      "remat",
      "packed",
      "separate",
      "remote_dma",
      "per_head_split",
      "per_head_a2a",
      "per_head_import",
      "kv_credit_serial",
      "kv_credit_prefetch",
      "kv_credit_deferred",
      "kv_credit_ahead",
      "kv_hbm_serial",
      "kv_hbm_prefetch",
      "kv_hbm_traced",
      "do_hbm_serial",
      "do_hbm_prefetch",
      "do_hbm_traced",
      "do_vmem_di_prefetch",
      "kv_shadow_serial",
      "kv_shadow_prefetch",
  ):
    raise ValueError("Unsupported MLA backward input exchange")
  if fwd_input_exchange not in ("external", "remote_dma"):
    raise ValueError("Unsupported MLA forward input exchange")
  if fwd_output_exchange not in ("external", "remote_dma"):
    raise ValueError("Unsupported MLA forward output exchange")
  if bwd_input_exchange != "remat" and not fused:
    raise ValueError("Experimental backward input exchanges require the fused VJP")
  per_head = bwd_input_exchange in ("per_head_split", "per_head_a2a", "per_head_import")
  do_hbm = bwd_input_exchange in ("do_hbm_serial", "do_hbm_prefetch", "do_hbm_traced", "do_vmem_di_prefetch")
  kv_hbm = do_hbm or bwd_input_exchange in ("kv_hbm_serial", "kv_hbm_prefetch", "kv_hbm_traced")
  kv_import = kv_hbm or bwd_input_exchange in (
      "kv_credit_serial",
      "kv_credit_prefetch",
      "kv_credit_deferred",
      "kv_credit_ahead",
  )
  kv_shadow = bwd_input_exchange in ("kv_shadow_serial", "kv_shadow_prefetch")
  if per_head and cp_barrier:
    raise ValueError("Per-head MLA backward requires the compiler global barrier")
  if kv_import and cp_barrier:
    raise ValueError("Experimental K/V row import requires the compiler global barrier")
  if kv_shadow and cp_barrier:
    raise ValueError("Shadow K/V traffic requires the compiler global barrier")
  sequence_residuals = bwd_input_exchange in (
      "packed",
      "separate",
      "remote_dma",
      "per_head_a2a",
      "per_head_import",
      "kv_credit_serial",
      "kv_credit_prefetch",
      "kv_credit_deferred",
      "kv_credit_ahead",
      "kv_hbm_serial",
      "kv_hbm_prefetch",
      "kv_hbm_traced",
      "do_hbm_serial",
      "do_hbm_prefetch",
      "do_hbm_traced",
      "do_vmem_di_prefetch",
      "kv_shadow_serial",
      "kv_shadow_prefetch",
  )
  if fused:
    cfg = _validate_kernel(kernel, cp)
    backward = make_backward(kernel, cp, axis_name=axis_name, cp_barrier=cp_barrier)
  else:
    cfg = kernel.kwargs["config"]

  def to_head(x):
    # BSHD -> BHSD, then split all heads / concatenate sequence pieces.
    x = jnp.transpose(x, (0, 2, 1, 3))
    return lax.all_to_all(x, axis_name, split_axis=1, concat_axis=2, tiled=True)

  def to_sequence(x):
    if fwd_output_exchange == "remote_dma":
      x = _forward_output_exchange(x[0], cp=cp, axis_name=axis_name)[None]
    else:
      x = lax.all_to_all(x, axis_name, split_axis=2, concat_axis=1, tiled=True)
    return jnp.transpose(x, (0, 2, 1, 3))

  def inputs(q, k, v, ids):
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
      raise ValueError("MHA Q/K/V leading shapes must match")
    if q.shape[0] != 1 or q.shape[2:] != (cp * 4, 192) or v.shape[-1] != 128:
      raise ValueError("Qualified local boundary is B1, H=CP*4, DQK192/DV128")
    if q.shape[1] % 4096 or ids.shape != q.shape[:2]:
      raise ValueError("Sequence shard must be divisible by 4096 and match segment IDs")
    if not all(x.dtype == jnp.bfloat16 for x in (q, k, v)) or ids.dtype != jnp.int32:
      raise ValueError("Qualified data types are BF16 and int32 segment IDs")
    # Preserve the production input Q/K/V collective fusion.
    packed = jnp.concatenate((q, k, v), axis=-1)
    if fwd_input_exchange == "remote_dma":
      packed = _forward_input_exchange(jnp.transpose(packed, (0, 2, 1, 3))[0], cp=cp, axis_name=axis_name)[None]
    else:
      packed = to_head(packed)
    qh, kh, vh = (
        ad_checkpoint.checkpoint_name(x, tag)
        for x, tag in zip(jnp.split(packed, (192, 384), axis=-1), ("mla_q_head", "mla_k_head", "mla_v_head"))
    )
    full_ids = lax.all_gather(ids, axis_name, axis=1, tiled=True)
    return qh, kh, vh, full_ids

  if not fused:

    def reference(q, k, v, ids):
      qh, kh, vh, full_ids = inputs(q, k, v, ids)
      # Differentiate the actual original Tokamax custom VJP through the
      # ordinary collectives; do not hand-write the reference gradient.
      result = kernel(qh[0], kh[0], vh[0], sk.SegmentIds(full_ids[0], full_ids[0]), save_residuals=record_max_logits)
      if record_max_logits:
        oh, stats = result
        # Match AttentionOp: reduce full local Q length; keep heads
        # sharded by context. Tokamax already stops stats gradients.
        return to_sequence(oh[None]), lax.stop_gradient(jnp.max(stats["max_logits"], axis=-1)[None])
      return to_sequence(result[None])

    return reference

  def forward(q, k, v, ids):
    qh, kh, vh, full_ids = inputs(q, k, v, ids)
    oh, stats = kernel(qh[0], kh[0], vh[0], sk.SegmentIds(full_ids[0], full_ids[0]), save_residuals=True)
    oh, lse = oh[None], stats["logsumexp"][None]
    if cfg.residual_checkpoint_name:
      oh = ad_checkpoint.checkpoint_name(oh, cfg.residual_checkpoint_name)
      lse = ad_checkpoint.checkpoint_name(lse, cfg.residual_checkpoint_name)
    output = ad_checkpoint.checkpoint_name(to_sequence(oh), "mla_attention_out")
    if record_max_logits:
      output = (output, lax.stop_gradient(jnp.max(stats["max_logits"], axis=-1)[None]))
    # The original policy already saves the sequence-layout input arguments.
    # Return those arguments so pullback owns the selected Q/K/V/dO exchange,
    # instead of remat separately rebuilding the head-layout Q/K/V residuals.
    qkv_residuals = (q, k, v) if sequence_residuals else (qh, kh, vh)
    return output, (*qkv_residuals, full_ids, oh, lse)

  @jax.custom_vjp
  def attention(q, k, v, ids):
    return forward(q, k, v, ids)[0]

  def pullback(residuals, cotangent):
    # Max-logit statistics are stop-gradient in the original Tokamax API.
    # Their cotangent must not enter attention's Q/K/V gradients.
    do_sequence = cotangent[0] if record_max_logits else cotangent
    qkv_a, qkv_b, qkv_c, ids, oh, lse = residuals
    if sequence_residuals:
      q_seq, k_seq, v_seq = qkv_a, qkv_b, qkv_c
      if do_sequence.dtype != jnp.bfloat16 or do_sequence.shape != q_seq.shape[:3] + (128,):
        raise ValueError("Backward input exchanges require BF16 dO with Q/K/V leading shape")
    if bwd_input_exchange == "packed":
      packed = to_head(jnp.concatenate((q_seq, k_seq, v_seq, do_sequence), axis=-1))
      qh, kh, vh, do_head = jnp.split(packed, (192, 384, 512), axis=-1)
      if qh.shape[-1] != 192 or do_head.shape[-1] != 128:
        raise ValueError("Packed backward input split has an invalid head dimension")
    elif bwd_input_exchange == "remote_dma":
      packed = _backward_input_exchange(
          jnp.transpose(jnp.concatenate((q_seq, k_seq, v_seq, do_sequence), axis=-1), (0, 2, 1, 3))[0],
          cp=cp,
          axis_name=axis_name,
      )[None]
      qh, kh, vh, do_head = jnp.split(packed, (192, 384, 512), axis=-1)
    elif bwd_input_exchange == "separate" or kv_shadow:
      qh, kh, vh, do_head = (to_head(x) for x in (q_seq, k_seq, v_seq, do_sequence))
    elif kv_import:
      qh = to_head(q_seq)
      # dO is sequence-owned only for the experimental dO-import selector.
      # Q and every forward exchange retain the existing path.
      do_head = jnp.transpose(do_sequence, (0, 2, 1, 3)) if do_hbm else to_head(do_sequence)
      kh, vh = (jnp.transpose(x, (0, 2, 1, 3)) for x in (k_seq, v_seq))
    elif bwd_input_exchange in ("per_head_a2a", "per_head_import"):
      # Keep each tensor separate so no 640-wide unpacking is needed. The
      # head slice is gathered before XLA's exchange; measure that cost too.
      sources = tuple(jnp.transpose(x, (0, 2, 1, 3))[0] for x in (q_seq, k_seq, v_seq, do_sequence))
      gradients = []
      for h in range(4):
        with jax.named_scope(f"mla_bwd_head_{h}"):
          if h == 0 or bwd_input_exchange == "per_head_a2a":
            head_inputs = tuple(_exchange_head(x, h, cp, axis_name) for x in sources)
          # all_to_all preserves a singleton head: [CP,L,D] -> [1,S,D].
          q_h, k_h, v_h, do_h = head_inputs
          importing = bwd_input_exchange == "per_head_import" and h < 3
          result = backward(
              q_h,
              k_h,
              v_h,
              ids[0],
              oh[0, h : h + 1],
              lse[0, h : h + 1],
              do_h,
              import_head=h + 1 if importing else None,
              imports=sources if importing else None,
          )
          gradients.append(result[:3])
          if importing:
            head_inputs = result[3:]
      dq, dk, dv = (_merge_head_gradients(parts) for parts in zip(*gradients))
    else:
      # Both C and E0 use the original full-head remat input exchange.
      qh, kh, vh = qkv_a, qkv_b, qkv_c
      do_head = to_head(do_sequence)
    if bwd_input_exchange == "per_head_split":
      # E0 isolates the four-call cost: retain C's full-head di reduction.
      with jax.named_scope("splash_bwd_di"):
        di = jnp.einsum("hsd,hsd->hs", oh[0].astype(jnp.float32), do_head[0].astype(jnp.float32))
      gradients = []
      for h in range(4):
        with jax.named_scope(f"mla_bwd_head_{h}"):
          gradients.append(
              backward(
                  qh[0, h : h + 1],
                  kh[0, h : h + 1],
                  vh[0, h : h + 1],
                  ids[0],
                  oh[0, h : h + 1],
                  lse[0, h : h + 1],
                  do_head[0, h : h + 1],
                  di=di[h : h + 1],
              )
          )
      dq, dk, dv = (_merge_head_gradients(parts) for parts in zip(*gradients))
    elif bwd_input_exchange not in ("per_head_a2a", "per_head_import"):
      dq, dk, dv = backward(
          *(x[0] for x in (qh, kh, vh, ids, oh, lse, do_head)),
          kv_import_prefetch=(bwd_input_exchange != "kv_credit_serial") if kv_import and not kv_hbm else None,
          kv_hbm_prefetch=(bwd_input_exchange != "kv_hbm_serial") if kv_hbm else None,
          kv_hbm_trace=bwd_input_exchange in ("kv_hbm_traced", "do_hbm_traced"),
          do_hbm_prefetch=(bwd_input_exchange != "do_hbm_serial") if do_hbm else None,
          do_di_resident=bwd_input_exchange == "do_vmem_di_prefetch",
          kv_import_defer_send=bwd_input_exchange in ("kv_credit_deferred", "kv_credit_ahead"),
          kv_import_launch_after_credit=bwd_input_exchange == "kv_credit_ahead",
          # Keep the diagnostic source conversion independent of the real
          # input A2A. Sharing the transpose lets its dim-minor requirement
          # force K through a copy and a reverse conversion before the A2A.
          # This barrier is only a layout-isolation candidate until AOT
          # confirms that the real collective's producer chain is preserved.
          shadow_imports=(
              tuple(jnp.transpose(lax.optimization_barrier(x), (0, 2, 1, 3))[0] for x in (k_seq, v_seq))
              if kv_shadow
              else None
          ),
          shadow_prefetch=bwd_input_exchange == "kv_shadow_prefetch",
      )
    # All three gradients already have [CP*H, S/CP, D] ownership.
    return (
        jnp.transpose(dq[None], (0, 2, 1, 3)),
        jnp.transpose(dk[None], (0, 2, 1, 3)),
        jnp.transpose(dv[None], (0, 2, 1, 3)),
        None,
    )

  attention.defvjp(forward, pullback)
  if _return_pullback:
    return pullback, sequence_residuals
  return attention


def _make_attention_with_external_inputs(
    kernel, *, mesh, batch_axes, axis_name, record_max_logits, pullback, sequence_residuals, fwd_output_exchange
):
  """Keep the original global forward boundary and own backward exchanges.

  The original MLA packs BSHD Q/K/V and constrains the packed tensor to head
  sharding before entering Splash's shard_map. Packing after entering a
  sequence-sharded shard_map changes the compiler boundary of the projection
  producers. Preserve that boundary here; only the backward rule uses the
  sequence-layout residuals and fused gradient exports.
  """
  sequence = P(batch_axes, axis_name, None, None)
  head = P(batch_axes, axis_name, None, None)  # BHSD, rather than BSHD.
  head_stats = P(batch_axes, axis_name, None)
  full_segments = P(batch_axes, None)
  cfg = kernel.kwargs["config"]

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(head, head, head, full_segments),
      out_specs=(head, head_stats, head_stats if record_max_logits else None),
      check_vma=False,
  )
  def head_forward(qh, kh, vh, ids):
    def one(q, k, v, segments):
      return kernel(q, k, v, sk.SegmentIds(segments, segments), save_residuals=True)

    oh, stats = jax.vmap(one)(qh, kh, vh, ids)
    lse = stats["logsumexp"]
    if cfg.residual_checkpoint_name:
      oh = ad_checkpoint.checkpoint_name(oh, cfg.residual_checkpoint_name)
      lse = ad_checkpoint.checkpoint_name(lse, cfg.residual_checkpoint_name)
    return oh, lse, stats["max_logits"] if record_max_logits else None

  if fwd_output_exchange == "remote_dma":
    remote_output = jax.shard_map(
        lambda oh: _forward_output_exchange(oh[0], cp=mesh.shape[axis_name], axis_name=axis_name)[None],
        mesh=mesh,
        in_specs=head,
        out_specs=P(batch_axes, None, axis_name, None),
        check_vma=False,
    )

  residual_qkv = sequence if sequence_residuals else head

  def layout_preserving_pullback(residuals, cotangent):
    dq, dk, dv, dids = pullback(residuals, cotangent)
    # The original backward A2A leaves BSHD dQ sequence-minor. Preserve that
    # physical boundary: head-dimension-minor remote output otherwise changes
    # wq_b's downstream FP8 GEMM layout/emitter and BF16 rounding. This only
    # constrains layout; the fused kernel still owns every gradient exchange.
    dq = with_layout_constraint(dq, Layout((0, 2, 3, 1)))
    return dq, dk, dv, dids

  mapped_pullback = jax.shard_map(
      layout_preserving_pullback,
      mesh=mesh,
      in_specs=(
          (residual_qkv, residual_qkv, residual_qkv, full_segments, head, head_stats),
          (sequence, P(batch_axes, axis_name)) if record_max_logits else sequence,
      ),
      out_specs=(sequence, sequence, sequence, None),
      check_vma=False,
  )

  def forward(q, k, v, ids):
    packed = lax.with_sharding_constraint(
        jnp.concatenate((q, k, v), axis=-1), NamedSharding(mesh, P(batch_axes, None, axis_name, None))
    )
    qh, kh, vh = (
        jnp.transpose(ad_checkpoint.checkpoint_name(x, name), (0, 2, 1, 3))
        for x, name in zip(jnp.split(packed, (192, 384), axis=-1), ("mla_q_head", "mla_k_head", "mla_v_head"))
    )
    full_ids = lax.with_sharding_constraint(ids, NamedSharding(mesh, full_segments))
    oh, lse, max_logits = head_forward(qh, kh, vh, full_ids)
    if fwd_output_exchange == "remote_dma":
      output = jnp.transpose(remote_output(oh), (0, 2, 1, 3))
    else:
      output = lax.with_sharding_constraint(jnp.transpose(oh, (0, 2, 1, 3)), NamedSharding(mesh, sequence))
    output = ad_checkpoint.checkpoint_name(output, "mla_attention_out")
    if record_max_logits:
      output = (output, lax.stop_gradient(jnp.max(max_logits, axis=2)))
    qkv = (q, k, v) if sequence_residuals else (qh, kh, vh)
    return output, (*qkv, full_ids, oh, lse)

  @jax.custom_vjp
  def attention(q, k, v, ids):
    return forward(q, k, v, ids)[0]

  attention.defvjp(forward, mapped_pullback)
  return attention


def make_attention(
    kernel,
    *,
    mesh,
    batch_axes,
    axis_name="context",
    fused=True,
    record_max_logits=False,
    evidence_dir=None,
    cp_barrier=False,
    bwd_input_exchange="remat",
    fwd_input_exchange="external",
    fwd_output_exchange="external",
):
  """Global BSHD boundary, preserving FSDP batch groups and CP peer groups."""
  cp = mesh.shape[axis_name]
  axes = (batch_axes,) if isinstance(batch_axes, str) else tuple(batch_axes or ())
  if axis_name in axes or len(set(axes)) != len(axes):
    raise ValueError("Batch and context axes must be disjoint")
  batch_size = math.prod(mesh.shape[a] for a in axes)
  sequence = P(batch_axes, axis_name, None, None)
  segments = P(batch_axes, axis_name)
  external_inputs = fused and fwd_input_exchange == "external"
  local = make_local_attention(
      kernel,
      cp=cp,
      axis_name=axis_name,
      fused=fused,
      record_max_logits=record_max_logits,
      evidence_dir=evidence_dir,
      cp_barrier=cp_barrier,
      bwd_input_exchange=bwd_input_exchange,
      fwd_input_exchange=fwd_input_exchange,
      fwd_output_exchange=fwd_output_exchange,
      _return_pullback=external_inputs,
  )
  if external_inputs:
    pullback, sequence_residuals = local
    mapped = _make_attention_with_external_inputs(
        kernel,
        mesh=mesh,
        batch_axes=batch_axes,
        axis_name=axis_name,
        record_max_logits=record_max_logits,
        pullback=pullback,
        sequence_residuals=sequence_residuals,
        fwd_output_exchange=fwd_output_exchange,
    )
  else:
    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(sequence, sequence, sequence, segments),
        out_specs=(sequence, P(batch_axes, axis_name)) if record_max_logits else sequence,
        check_vma=False,
    )

  def call(q, k, v, ids):
    if q.shape[0] != batch_size:
      raise ValueError("Qualified local batch is exactly one per FSDP group")
    if external_inputs:
      if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[2:] != (cp * 4, 192) or v.shape[-1] != 128:
        raise ValueError("Qualified global boundary requires matching Q/K/V with H=CP*4, DQK192/DV128")
      if q.shape[1] % (cp * 4096) or ids.shape != q.shape[:2]:
        raise ValueError("Sequence shards must be divisible by 4096 and match segment IDs")
      if not all(x.dtype == jnp.bfloat16 for x in (q, k, v)) or ids.dtype != jnp.int32:
        raise ValueError("Qualified data types are BF16 and int32 segment IDs")
    return mapped(q, k, v, ids)

  return call
