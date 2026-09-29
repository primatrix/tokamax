"""Independent output/VJP coverage for grouped segment IDs and union grids."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_layout_test import _oracle


def _mask_info(ids, block_q, block_kv, backward=False):
  allowed = ids[:, :, None] == ids[:, None, :]
  tiles = allowed.reshape(len(ids), 512 // block_q, block_q, 512 // block_kv, block_kv)
  active = tiles.any(axis=(2, 4))
  full = tiles.all(axis=(2, 4))
  if backward:
    active, full = active.swapaxes(-1, -2), full.swapaxes(-1, -2)
  union = active.any(axis=0)
  indices = np.argwhere(union)
  rows, cols = np.full((2, union.size), -1, np.int32)
  rows[:len(indices)], cols[:len(indices)] = indices.T
  kinds = np.zeros((len(ids), union.size), np.int8)
  kinds[:, :len(indices)] = np.where(
      active[:, indices[:, 0], indices[:, 1]],
      np.where(full[:, indices[:, 0], indices[:, 1]], 2, 1), 0,
  )
  return splash.MaskInfo(
      mask_next=jnp.full((union.size,), -1, jnp.int8),
      active_rows=jnp.asarray(rows), active_cols=jnp.asarray(cols),
      block_mask=jnp.asarray(kinds), num_active_blocks=jnp.array([len(indices)], jnp.int32),
      partial_mask_blocks=None, q_sequence=None,
  )


def _fixture(shift=0.0, width=72, heads_per_group=2, kv_heads_per_group=2):
  ids = np.array([[1] * 256 + [2] * 256, [1] * 384 + [2] * 96 + [0] * 32], np.int32)
  config = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR, v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=width**-0.5, max_logit_const=shift, combine_log2_scale=True,
      segment_mask_on_partial_only=True, bwd_scale_after_dot=True,
      dq_reduction_steps=3, interpret=jax.default_backend() == "cpu",
  )
  rng = np.random.default_rng(52)
  arrays = [
      jnp.asarray(rng.normal(size=(2 * heads, 512, width)), jnp.bfloat16)
      for heads in (heads_per_group, kv_heads_per_group, kv_heads_per_group, heads_per_group)
  ]
  kernel = splash.SplashAttentionKernel(
      _mask_info(ids, 128, 256), _mask_info(ids, 128, 256, True),
      config=config, is_mqa=False, save_residuals=True,
      mask_value=splash.base.DEFAULT_MASK_VALUE, mask_function=None,
      fwd_mask_sparsity=1.0, dkv_mask_sparsity=1.0,
  )
  return config, arrays, ids, kernel


@pytest.mark.parametrize("shift", [0.0, None])
@pytest.mark.parametrize("width,heads_per_group,kv_heads_per_group", [
    (72, 2, 2), (128, 1, 1),
    pytest.param(72, 2, 1, marks=pytest.mark.skipif(
        jax.default_backend() == "cpu",
        reason="Dynamic-grid GQA requires cross-program in/out alias updates; validate on TPU.",
    )),
])
def test_grouped_segments_outputs_and_gradients(shift, width, heads_per_group, kv_heads_per_group):
  config, (q, k, v, do), ids, kernel = _fixture(shift, width, heads_per_group, kv_heads_per_group)
  segments = splash.SegmentIds(jnp.asarray(ids), jnp.asarray(ids))
  (output, stats), vjp = jax.vjp(lambda q, k, v: kernel(q, k, v, segments), q, k, v)
  gradients = vjp((do, jax.tree.map(jnp.zeros_like, stats)))
  head_ids = np.repeat(ids, heads_per_group, axis=0)
  allowed = head_ids[:, :, None] == head_ids[:, None, :]
  expected, lse, *expected_grads = _oracle(q, k, v, do, None, allowed, config)
  for actual, wanted in zip((output, *gradients), (expected, *expected_grads)):
    actual = np.asarray(actual, np.float64)
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - wanted) / np.linalg.norm(wanted) < 0.01
  np.testing.assert_allclose(stats["logsumexp"], lse, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("invalid", ["head_count", "mixed_rank", "different_groups", "unfused", "mask_groups"])
def test_invalid_grouped_segment_shapes_fail_early(invalid):
  config, (q, k, v, _), ids, kernel = _fixture()
  qids = kids = jnp.asarray(ids)
  if invalid == "head_count":
    qids = kids = jnp.ones((3, 512), jnp.int32)
  elif invalid == "mixed_rank":
    kids = kids[0]
  elif invalid == "different_groups":
    kids = kids[:1]
  elif invalid == "unfused":
    with pytest.raises(ValueError, match="Only the fused bwd kernel"):
      dataclasses.replace(config, use_fused_bwd_kernel=False)
    return
  else:
    kernel.fwd_mask_info = kernel.fwd_mask_info._replace(block_mask=jnp.ones((3, 16), jnp.int8))
  with pytest.raises(ValueError, match="groups|Grouped|Block-mask"):
    kernel(q, k, v, splash.SegmentIds(qids, kids))


@pytest.mark.parametrize("groups", [2, 3])
@pytest.mark.parametrize("grouped_backward_mask", [False, True])
def test_grouped_mask_sequence_sharding_outputs_and_gradients(
    groups, grouped_backward_mask
):
  if len(jax.devices()) < 2:
    pytest.skip("Requires two devices; on CPU set JAX_NUM_CPU_DEVICES=2.")
  partition = jax.sharding.PartitionSpec
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("seq",))
  config = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, max_logit_const=0.0,
      combine_log2_scale=True, segment_mask_on_partial_only=True,
      interpret=jax.default_backend() == "cpu",
  )
  # Concatenate two sequence shards' grids. Each shard has two local Q tiles
  # and visits both KV tiles; head groups attend alternating KV halves.
  rows = np.array([0, 0, 1, 1] * 2, np.int32)
  cols = np.array([0, 1, 0, 1] * 2, np.int32)

  def mask_info(backward=False):
    kv_tiles = rows if backward else cols
    kinds = np.where(np.arange(groups)[:, None] % 2 == kv_tiles, 2, 0)
    if backward and not grouped_backward_mask:
      # A shared partial mask checks runtime IDs for every group. Forward
      # and backward metadata therefore deliberately have different ranks.
      kinds = np.ones(len(rows), np.int8)
    return splash.MaskInfo(
        active_rows=jnp.asarray(rows), active_cols=jnp.asarray(cols),
        mask_next=jnp.full((len(rows),), -1, jnp.int32),
        num_active_blocks=jnp.array([4, 4], jnp.int32),
        block_mask=jnp.asarray(kinds, jnp.int8),
        partial_mask_blocks=None, q_sequence=None,
    )

  kernel = splash.SplashAttentionKernel(
      mask_info(), mask_info(backward=True),
      config=config, is_mqa=False, save_residuals=False,
      mask_value=splash.base.DEFAULT_MASK_VALUE, mask_function=None,
      fwd_mask_sparsity=1.0, dkv_mask_sparsity=1.0,
  )
  kernel_spec = kernel.manual_sharding_spec(
      jax.sharding.NamedSharding(mesh, partition("seq"))
  )
  q_ids = np.zeros((groups, 512), np.int32)
  kv_ids = (np.arange(512)[None, :] // 256 != np.arange(groups)[:, None] % 2)
  segments = splash.SegmentIds(jnp.asarray(q_ids), jnp.asarray(kv_ids, jnp.int32))
  rng = np.random.default_rng(93)
  q, k, v, do = [
      jnp.asarray(rng.normal(size=(groups, 512, width)), jnp.bfloat16)
      for width in (72, 72, 96, 96)
  ]

  @functools.partial(
      jax.shard_map, mesh=mesh,
      in_specs=(
          kernel_spec, partition(None, "seq"), partition(), partition(),
          splash.SegmentIds(partition(None, "seq"), partition()),
      ),
      out_specs=partition(None, "seq"), check_vma=False,
  )
  def sharded(kernel, q, k, v, segments):
    return kernel(q, k, v, segments)

  output, vjp = jax.vjp(lambda q, k, v: sharded(kernel, q, k, v, segments), q, k, v)
  gradients = vjp(do)
  allowed = q_ids[:, :, None] == kv_ids[:, None, :]
  expected, _, *expected_grads = _oracle(q, k, v, do, None, allowed, config)
  for actual, wanted in zip((output, *gradients), (expected, *expected_grads)):
    actual = np.asarray(actual, np.float64)
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - wanted) / np.linalg.norm(wanted) < 0.01


@pytest.mark.parametrize("backward", [False, True])
def test_grouped_mask_sharding_rejects_indivisible_grid(backward):
  if len(jax.devices()) < 2:
    pytest.skip("Requires two devices; on CPU set JAX_NUM_CPU_DEVICES=2.")
  _, _, _, kernel = _fixture()
  attr = "dkv_mask_info" if backward else "fwd_mask_info"
  info = getattr(kernel, attr)
  setattr(kernel, attr, info._replace(block_mask=info.block_mask[:, :-1]))
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("seq",))
  with pytest.raises(ValueError, match="divide the mask blocks evenly"):
    kernel.manual_sharding_spec(
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("seq"))
    )
