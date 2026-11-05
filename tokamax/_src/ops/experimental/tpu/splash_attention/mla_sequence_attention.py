"""Sequence-layout custom VJP for the fused MLA backward kernel.

The factory accepts the caller's complete Splash kernel/configuration; it does
not change its tile/cost configuration or install a remat policy. The fused
backward supports any context parallelism CP that divides the global head
count and the number of K/V tiles (``block_kv_dkv`` tokens each). Head widths
come from the Q/K/V inputs. Callers keep their existing implementation for
other masks and shapes.
"""

from __future__ import annotations

import functools
import math
import numpy as np

import jax
import jax.numpy as jnp
from jax import ad_checkpoint, lax
from jax.sharding import NamedSharding, PartitionSpec as P

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk

from tokamax._src.ops.experimental.tpu.splash_attention.mla_fused_backward import make_backward, validate_source


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
  if type(cp) is not int or cp < 1:
    raise ValueError("Context parallelism must be a positive integer")
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
    _return_pullback=False,
):
  """Map local BSHD sequence shards to the same layout and gradient ownership.

  Global H must be a multiple of CP. At the boundary a rank owns all heads
  for S/CP tokens; inside the kernel it owns H/CP heads for the full
  sequence. Runtime segment IDs enter as sequence shards and the local batch
  is exactly one. ``kv_hbm_*`` imports K/V inside the backward kernel from
  the sequence-layout arguments that the original remat policy already
  saves; ``remat`` uses the rematerialized head-layout Q/K/V.
  """
  del evidence_dir  # Accepted for API compatibility.
  if bwd_input_exchange not in ("remat", "kv_hbm_serial", "kv_hbm_prefetch"):
    raise ValueError("Unsupported MLA backward input exchange")
  kv_import = bwd_input_exchange != "remat"
  if kv_import and not fused:
    raise ValueError("Backward K/V import requires the fused VJP")
  if kv_import and cp_barrier:
    raise ValueError("Backward K/V import requires the compiler global barrier")
  sequence_residuals = kv_import
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
    x = lax.all_to_all(x, axis_name, split_axis=2, concat_axis=1, tiled=True)
    return jnp.transpose(x, (0, 2, 1, 3))

  def inputs(q, k, v, ids):
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
      raise ValueError("MHA Q/K/V leading shapes must match")
    if q.shape[0] != 1 or q.shape[2] % cp:
      raise ValueError("Local boundary requires batch one and heads divisible by CP")
    if q.shape[1] % cfg.block_kv_dkv or ids.shape != q.shape[:2]:
      raise ValueError("Sequence shard must be divisible by block_kv_dkv and match segment IDs")
    if not all(x.dtype == jnp.bfloat16 for x in (q, k, v)) or ids.dtype != jnp.int32:
      raise ValueError("Supported data types are BF16 and int32 segment IDs")
    # Preserve the production input Q/K/V collective fusion.
    packed = to_head(jnp.concatenate((q, k, v), axis=-1))
    dqk = q.shape[-1]
    qh, kh, vh = (
        ad_checkpoint.checkpoint_name(x, tag)
        for x, tag in zip(jnp.split(packed, (dqk, 2 * dqk), axis=-1), ("mla_q_head", "mla_k_head", "mla_v_head"))
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
    # Return those arguments so pullback owns the K/V exchange, instead of
    # remat separately rebuilding the head-layout Q/K/V residuals.
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
    if kv_import:
      q_seq, k_seq, v_seq = qkv_a, qkv_b, qkv_c
      if do_sequence.dtype != jnp.bfloat16 or do_sequence.shape[:3] != q_seq.shape[:3]:
        raise ValueError("Backward K/V import requires BF16 dO with the Q/K/V leading shape")
      # Q/dO use external exchanges. The kernel imports each K/V head.
      qh, do_head = (to_head(x) for x in (q_seq, do_sequence))
      kh, vh = (jnp.transpose(x, (0, 2, 1, 3)) for x in (k_seq, v_seq))
    else:
      qh, kh, vh = qkv_a, qkv_b, qkv_c
      do_head = to_head(do_sequence)
    dq, dk, dv = backward(
        *(x[0] for x in (qh, kh, vh, ids, oh, lse, do_head)),
        kv_hbm_prefetch=(bwd_input_exchange == "kv_hbm_prefetch") if kv_import else None,
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
    kernel, *, mesh, batch_axes, axis_name, record_max_logits, pullback, sequence_residuals
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

  residual_qkv = sequence if sequence_residuals else head

  # dQ keeps the kernel's head-dimension-minor layout; XLA chooses the
  # downstream projection layout.
  mapped_pullback = jax.shard_map(
      pullback,
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
        for x, name in zip(
            jnp.split(packed, (q.shape[-1], 2 * q.shape[-1]), axis=-1), ("mla_q_head", "mla_k_head", "mla_v_head")
        )
    )
    full_ids = lax.with_sharding_constraint(ids, NamedSharding(mesh, full_segments))
    oh, lse, max_logits = head_forward(qh, kh, vh, full_ids)
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
):
  """Global BSHD boundary, preserving FSDP batch groups and CP peer groups."""
  cp = mesh.shape[axis_name]
  axes = (batch_axes,) if isinstance(batch_axes, str) else tuple(batch_axes or ())
  if axis_name in axes or len(set(axes)) != len(axes):
    raise ValueError("Batch and context axes must be disjoint")
  batch_size = math.prod(mesh.shape[a] for a in axes)
  sequence = P(batch_axes, axis_name, None, None)
  segments = P(batch_axes, axis_name)
  external_inputs = fused
  local = make_local_attention(
      kernel,
      cp=cp,
      axis_name=axis_name,
      fused=fused,
      record_max_logits=record_max_logits,
      evidence_dir=evidence_dir,
      cp_barrier=cp_barrier,
      bwd_input_exchange=bwd_input_exchange,
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
      raise ValueError("Local batch must be exactly one per FSDP group")
    if external_inputs:
      if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[2] % cp:
        raise ValueError("Global boundary requires matching Q/K/V with heads divisible by CP")
      if q.shape[1] % (cp * kernel.kwargs["config"].block_kv_dkv) or ids.shape != q.shape[:2]:
        raise ValueError("Sequence shards must be divisible by block_kv_dkv and match segment IDs")
      if not all(x.dtype == jnp.bfloat16 for x in (q, k, v)) or ids.dtype != jnp.int32:
        raise ValueError("Supported data types are BF16 and int32 segment IDs")
    return mapped(q, k, v, ids)

  return call
