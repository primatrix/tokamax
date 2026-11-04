"""Abstract regression of the fused MLA backward caller.

Runs on CPU without allocating the large attention tensors. Covers the
production CP16/four-head geometry and other CP sizes and local head counts,
including metadata that is traced (as in topology AOT). Device numerics are
validated separately by full-model training.
"""


import dataclasses
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.ad_checkpoint import saved_residuals
from jax.sharding import Mesh
from tokamax._src.ops.experimental.tpu.splash_attention import mla_fused_backward
from tokamax._src.ops.experimental.tpu.splash_attention import mla_sequence_attention
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as sm
from tokamax.experimental import mla_attention


@pytest.mark.skipif(len(jax.devices()) < 4, reason="Requires four logical devices")
@pytest.mark.parametrize("record_max_logits", [False, True])
def test_global_forward_boundary_preserves_values_and_vjp_ownership(record_max_logits):
  """Check the global VJP boundary against an independent elementwise function.

  A small analytic kernel makes every batch/head/token observable on CPU.
  Actual Splash arithmetic and remote DMA still require device qualification.
  """
  mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("fsdp", "context"))

  class AnalyticKernel:
    kwargs = {"config": SimpleNamespace(residual_checkpoint_name="mla_splash_residuals")}

    def __call__(self, q, k, v, ids, *, save_residuals):
      del ids
      assert save_residuals
      return q[..., :128] * k[..., :128] + v, {
          "logsumexp": jnp.zeros(q.shape[:2], jnp.float32),
          "max_logits": jnp.max(q.astype(jnp.float32), axis=-1),
      }

  def pullback(residuals, cotangent):
    q, k, _, _, _, _ = residuals
    do = cotangent[0] if record_max_logits else cotangent
    pad = ((0, 0), (0, 0), (0, 0), (0, 64))
    return jnp.pad(do * k[..., :128], pad), jnp.pad(do * q[..., :128], pad), do, None

  attention = mla_sequence_attention._make_attention_with_external_inputs(
      AnalyticKernel(),
      mesh=mesh,
      batch_axes="fsdp",
      axis_name="context",
      record_max_logits=record_max_logits,
      pullback=pullback,
      sequence_residuals=True,
      fwd_output_exchange="external",
  )
  shape = (2, 16, 8, 192)
  q = (jnp.arange(math.prod(shape)).reshape(shape) % 7).astype(jnp.bfloat16)
  k = ((jnp.arange(math.prod(shape)).reshape(shape) // 7) % 5).astype(jnp.bfloat16)
  v = (q[..., :128] % 3).astype(jnp.bfloat16)
  do = (k[..., :128] % 3).astype(jnp.bfloat16)
  ids = jnp.ones(shape[:2], jnp.int32)

  def reference(q, k, v):
    return q[..., :128] * k[..., :128] + v

  expected, reference_vjp = jax.vjp(reference, q, k, v)

  @jax.jit
  def cycle(q, k, v, ids, do):
    output, vjp = jax.vjp(lambda q, k, v: attention(q, k, v, ids), q, k, v)
    cotangent = (do, jnp.ones_like(output[1])) if record_max_logits else do
    return output, vjp(cotangent)

  with jax.set_mesh(mesh):
    actual, gradients = cycle(q, k, v, ids, do)
  if record_max_logits:
    np.testing.assert_array_equal(actual[1], jnp.max(q.astype(jnp.float32), axis=(1, 3)))
    actual = actual[0]
  np.testing.assert_array_equal(actual, expected)
  for gradient, reference_gradient in zip(gradients, reference_vjp(do), strict=True):
    np.testing.assert_array_equal(gradient, reference_gradient)


def config():
  return sk.SplashConfig(
      block_q=4096,
      block_kv=2048,
      block_kv_compute=512,
      block_q_dkv=512,
      block_kv_dkv=4096,
      block_kv_dkv_compute=512,
      use_fused_bwd_kernel=True,
      dq_reduction_steps=3,
      q_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      k_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      bwd_cast_before_transpose=True,
      use_base2_exp=False,
      softmax_scale=0.07216878364870322,
      residual_checkpoint_name="mla_splash_residuals",
      use_experimental_scheduler=True,
  )


def test_pinned_source_and_bad_config_rejection():
  mla_fused_backward.validate_source()
  cfg = config()
  kernel = sk.make_splash_mha(sm.CausalMask((32768, 32768)), config=cfg, q_seq_shards=1)
  bad = sk.SplashAttentionKernel(
      kernel.fwd_mask_info,
      kernel.dkv_mask_info,
      **dict(kernel.kwargs, config=dataclasses.replace(cfg, bwd_dq_first=True)),
  )
  with pytest.raises(ValueError, match="Unsupported MLA"):
    mla_fused_backward.make_backward(bad, 8)


def test_static_mask_metadata_stays_numpy_with_outer_mesh():
  mesh = Mesh(np.asarray(jax.devices()), ("outer",))
  observed = []
  reference = sk.make_splash_mha(sm.CausalMask((32768, 32768)), config=config(), q_seq_shards=1)
  expected = jax.tree.leaves((reference.fwd_mask_info, reference.dkv_mask_info))

  def traced(x):
    kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((32768, 32768)), config())
    leaves = jax.tree.leaves((kernel.fwd_mask_info, kernel.dkv_mask_info))
    assert leaves
    for leaf, original in zip(leaves, expected, strict=True):
      assert isinstance(leaf, np.ndarray)
      np.testing.assert_array_equal(leaf, original)
    with jax.ensure_compile_time_eval():
      observed.append(jax.sharding.get_abstract_mesh())
    return x

  with jax.set_mesh(mesh):
    jax.eval_shape(traced, jax.ShapeDtypeStruct((), jnp.float32))
    assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh
  assert observed == [mesh.abstract_mesh]


@pytest.fixture(scope="module")
def export_metadata_kernel():
  return mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((65536, 65536)), config())


@pytest.mark.parametrize("cp", [1, 2, 4, 8, 16])
def test_export_writes_match_actual_bounds(export_metadata_kernel, cp):
  kernel = export_metadata_kernel
  proof = mla_fused_backward.validate_metadata(kernel, cp)
  info = kernel.dkv_mask_info
  n = proof["programs_per_head"]
  _, actual_end = sk.mask_info_lib.find_bounds(info.active_rows)
  writes = np.asarray(actual_end)[:n].astype(bool)
  np.testing.assert_array_equal(np.asarray(info.active_rows)[:n][writes], np.arange(proof["kv_steps"]))
  assert proof["export_writes_per_head"] == proof["kv_steps"]
  assert np.count_nonzero(writes) < n  # Interior Q accumulation does not send.


def test_export_rejects_padding_that_suppresses_last_send(export_metadata_kernel):
  info = export_metadata_kernel.dkv_mask_info
  n = int(np.asarray(info.num_active_blocks)[0])
  # The dynamic grid ends before this repeated final row. find_bounds would
  # move its final write into padding, leaving the final send/drain unmatched.
  padded = info._replace(
      active_rows=np.insert(info.active_rows, n, info.active_rows[n - 1]),
      active_cols=np.insert(info.active_cols, n, info.active_cols[n - 1]),
  )
  with pytest.raises(ValueError, match="exactly one final write"):
    mla_fused_backward.validate_metadata(SimpleNamespace(dkv_mask_info=padded), 16)


@pytest.mark.parametrize("count", [0, -1, 1000000])
def test_export_rejects_invalid_active_count(export_metadata_kernel, count):
  info = export_metadata_kernel.dkv_mask_info._replace(num_active_blocks=np.asarray([count], dtype=np.int32))
  with pytest.raises(ValueError, match="active program count"):
    mla_fused_backward.validate_metadata(SimpleNamespace(dkv_mask_info=info), 16)


def test_export_rejects_reordered_accumulation(export_metadata_kernel):
  info = export_metadata_kernel.dkv_mask_info
  cols = np.array(info.active_cols, copy=True)
  cols[[0, 1]] = cols[[1, 0]]
  with pytest.raises(ValueError, match="ordered causal tile schedule"):
    mla_fused_backward.validate_metadata(SimpleNamespace(dkv_mask_info=info._replace(active_cols=cols)), 16)



def test_export_rejects_cp_that_does_not_partition_kv_tiles(export_metadata_kernel):
  # 64K has sixteen 4096-token K/V tiles; three ranks cannot own whole tiles.
  with pytest.raises(ValueError, match="complete CP partitions"):
    mla_fused_backward.validate_metadata(export_metadata_kernel, 3)


@pytest.mark.parametrize("cp", [2, 4, 16])
def test_traced_metadata_uses_static_causal_proof(export_metadata_kernel, cp):
  """Topology AOT can trace the mask metadata; it is never read on the host."""
  info = export_metadata_kernel.dkv_mask_info
  concrete = mla_fused_backward.validate_metadata(export_metadata_kernel, cp, 65536)
  observed = []

  def traced(count, rows, cols):
    kernel = SimpleNamespace(
        dkv_mask_info=info._replace(num_active_blocks=count, active_rows=rows, active_cols=cols),
        kwargs=export_metadata_kernel.kwargs,
    )
    assert mla_fused_backward.validate_metadata(kernel, cp) is None
    observed.append(mla_fused_backward.validate_metadata(kernel, cp, 65536))
    with pytest.raises(ValueError, match="complete CP partitions"):
      mla_fused_backward.validate_metadata(kernel, 3, 65536)
    with pytest.raises(ValueError, match="schedule length"):
      mla_fused_backward.validate_metadata(kernel, cp, 131072)
    return count

  leaves = (info.num_active_blocks, info.active_rows, info.active_cols)
  jax.eval_shape(traced, *(jax.ShapeDtypeStruct(np.shape(x), x.dtype) for x in leaves))
  (proof,) = observed
  assert proof.pop("concrete") is False and concrete.pop("concrete") is True
  assert proof == concrete


def _mesh(cp):
  devices = jax.devices()
  fsdp = len(devices) // cp
  return Mesh(np.asarray(devices[: fsdp * cp]).reshape(fsdp, cp), ("fsdp", "context")), fsdp


# (CP, local heads): production CP16/H64 is (16, 4); Plus-VL H64 at CP8 is
# (8, 8); CP3 checks a non-power-of-two context group.
GEOMETRIES = [(16, 4), (8, 8), (16, 2), (4, 4), (3, 4), (2, 8)]


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Run with 16 logical CPU devices or on a 16-device slice")
@pytest.mark.parametrize("cp,local_heads", GEOMETRIES)
@pytest.mark.parametrize("bwd_input_exchange", ["remat", "kv_hbm_serial", "kv_hbm_prefetch"])
def test_full_sequence_vjp_output_ownership(cp, local_heads, bwd_input_exchange):
  mesh, fsdp = _mesh(cp)
  heads, seq = cp * local_heads, cp * 4096 * 2
  with jax.ensure_compile_time_eval():
    kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((seq, seq)), config())
    attention = mla_attention.make_attention(
        kernel,
        mesh=mesh,
        batch_axes="fsdp",
        record_max_logits=True,
        bwd_input_exchange=bwd_input_exchange,
    )

  def cycle(q, k, v, ids, do):
    (out, stats), pullback = jax.vjp(lambda q, k, v: attention(q, k, v, ids), q, k, v)
    return out, stats, *pullback((do, jnp.ones_like(stats)))

  args = [jax.ShapeDtypeStruct((fsdp, seq, heads, d), jnp.bfloat16) for d in (192, 192, 128)]
  args += [jax.ShapeDtypeStruct((fsdp, seq), jnp.int32), jax.ShapeDtypeStruct((fsdp, seq, heads, 128), jnp.bfloat16)]
  with jax.set_mesh(mesh):
    program = jax.make_jaxpr(cycle)(*args)
  assert [v.aval.shape for v in program.jaxpr.outvars] == [
      (fsdp, seq, heads, 128),
      (fsdp, heads),
      (fsdp, seq, heads, 192),
      (fsdp, seq, heads, 192),
      (fsdp, seq, heads, 128),
  ]
  assert str(program).count("name=mla_fused_dq_dkv_a2a") == 1


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Run with 16 logical CPU devices or on a 16-device slice")
@pytest.mark.parametrize("cp,local_heads", [(16, 4), (8, 8)])
def test_traced_kernel_metadata_end_to_end(cp, local_heads):
  """The whole fused VJP traces when every mask-metadata leaf is a tracer."""
  mesh, fsdp = _mesh(cp)
  heads, seq = cp * local_heads, cp * 4096 * 2
  kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((seq, seq)), config())
  leaves, treedef = jax.tree.flatten(kernel)
  assert leaves and all(isinstance(x, np.ndarray) for x in leaves)

  def cycle(leaves, q, k, v, ids, do):
    traced = jax.tree.unflatten(treedef, leaves)
    assert not isinstance(traced.dkv_mask_info.active_rows, np.ndarray)
    attention = mla_attention.make_attention(
        traced, mesh=mesh, batch_axes="fsdp", record_max_logits=True, bwd_input_exchange="kv_hbm_prefetch"
    )
    (out, stats), pullback = jax.vjp(lambda q, k, v: attention(q, k, v, ids), q, k, v)
    return out, stats, *pullback((do, jnp.ones_like(stats)))

  args = [jax.ShapeDtypeStruct((fsdp, seq, heads, d), jnp.bfloat16) for d in (192, 192, 128)]
  args += [jax.ShapeDtypeStruct((fsdp, seq), jnp.int32), jax.ShapeDtypeStruct((fsdp, seq, heads, 128), jnp.bfloat16)]
  with jax.set_mesh(mesh):
    program = jax.make_jaxpr(cycle)(leaves, *args)
  assert str(program).count("name=mla_fused_dq_dkv_a2a") == 1


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Requires 16 logical devices")
@pytest.mark.parametrize("cp,local_heads", [(16, 4), (8, 8), (4, 2)])
def test_input_exchanges_preserve_original_policy_saved_tensors(cp, local_heads):
  mesh, fsdp = _mesh(cp)
  heads, seq = cp * local_heads, 65536
  kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((seq, seq)), config())
  args = [jax.ShapeDtypeStruct((fsdp, seq, heads, d), jnp.bfloat16) for d in (192, 192, 128)]
  args.append(jax.ShapeDtypeStruct((fsdp, seq), jnp.int32))
  saved_sets = []
  for exchange in ("remat", "kv_hbm_serial", "kv_hbm_prefetch"):
    attention = mla_attention.make_attention(
        kernel, mesh=mesh, batch_axes="fsdp", record_max_logits=True, bwd_input_exchange=exchange
    )
    checkpointed = jax.checkpoint(
        attention,
        prevent_cse=True,
        policy=jax.checkpoint_policies.save_only_these_names("mla_splash_residuals"),
    )
    with jax.set_mesh(mesh):
      residuals = saved_residuals(checkpointed, *args)
    saved_sets.append(
        sorted(
            (str(aval.dtype), math.prod(aval.shape), origin if origin.startswith("from the argument") else "activation")
            for aval, origin in residuals
            if origin != "from a constant"
        )
    )
  expected = sorted(
      [
          ("bfloat16", fsdp * seq * heads * 192, "from the argument q"),
          ("bfloat16", fsdp * seq * heads * 192, "from the argument k"),
          ("bfloat16", fsdp * seq * heads * 128, "from the argument v"),
          ("int32", fsdp * seq, "from the argument ids"),
          ("bfloat16", fsdp * heads * seq * 128, "activation"),
          ("float32", fsdp * heads * seq, "activation"),
      ]
  )
  assert all(saved == expected for saved in saved_sets)


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Requires 16 logical devices")
@pytest.mark.parametrize("case", ["heads", "sequence"])
def test_global_boundary_rejects_incomplete_partitions(case):
  cp = 16
  mesh, fsdp = _mesh(cp)
  seq = 65536
  kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((seq, seq)), config())
  attention = mla_attention.make_attention(kernel, mesh=mesh, batch_axes="fsdp", bwd_input_exchange="kv_hbm_prefetch")
  heads = cp * 4 + 1 if case == "heads" else cp * 4
  length = seq if case == "heads" else seq + cp * 2048
  q = jax.ShapeDtypeStruct((fsdp, length, heads, 192), jnp.bfloat16)
  v = jax.ShapeDtypeStruct((fsdp, length, heads, 128), jnp.bfloat16)
  with jax.set_mesh(mesh), pytest.raises(ValueError, match="divisible"):
    jax.eval_shape(attention, q, q, v, jax.ShapeDtypeStruct((fsdp, length), jnp.int32))


def test_removed_input_exchanges_rejected(export_metadata_kernel):
  for exchange in ("separate", "remote_dma", "per_head_import", "kv_credit_prefetch", "kv_shadow_prefetch", "x"):
    with pytest.raises(ValueError, match="Unsupported MLA backward input exchange"):
      mla_sequence_attention.make_local_attention(export_metadata_kernel, cp=8, bwd_input_exchange=exchange)


def test_kv_import_requires_fused_vjp_and_global_barrier(export_metadata_kernel):
  with pytest.raises(ValueError, match="fused VJP"):
    mla_sequence_attention.make_local_attention(
        export_metadata_kernel, cp=8, fused=False, bwd_input_exchange="kv_hbm_prefetch"
    )
  with pytest.raises(ValueError, match="global barrier"):
    mla_sequence_attention.make_local_attention(
        export_metadata_kernel, cp=8, cp_barrier=True, bwd_input_exchange="kv_hbm_prefetch"
    )


def test_unknown_forward_input_exchange_rejected(export_metadata_kernel):
  with pytest.raises(ValueError, match="Unsupported MLA forward input exchange"):
    mla_sequence_attention.make_local_attention(export_metadata_kernel, cp=8, fwd_input_exchange="invalid")
