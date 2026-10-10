"""Masked rows have zero mass, including before the first nonempty KV tile."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as masks


@pytest.mark.parametrize('blocks', [127, 128, 129])
def test_sparse_grid_index_dtype_boundary(blocks):
  """Trace actual forward/backward with compact indices around int8 capacity."""
  n = blocks * 128
  cfg = dataclasses.replace(
      splash.SplashConfig.get_default(),
      block_q=128, block_kv=128, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128,
      dq_reduction_steps=3,
  )
  fn = splash.make_splash_mha_single_device(
      masks.CausalMask((n, n)), config=cfg,
  )
  shape = jax.ShapeDtypeStruct((1, n, 128), jnp.bfloat16)

  def forward_and_backward(q, k, v):
    out, pullback = jax.vjp(fn, q, k, v)
    return out, pullback(jnp.ones_like(out))

  out, grads = jax.eval_shape(forward_and_backward, shape, shape, shape)
  assert out.shape == shape.shape
  assert all(g.shape == shape.shape for g in grads)


@pytest.mark.parametrize('interpret', [True, False], ids=['interpret', 'native'])
@pytest.mark.parametrize('layout', list(splash.QKVLayout))
@pytest.mark.parametrize('mask_value', [base.DEFAULT_MASK_VALUE, -np.inf])
@pytest.mark.parametrize('case', ['segments', 'static_mask', 'skipped_q_block', 'sparse_backward', 'all_empty', 'fixed_shift', 'external_reciprocal', 'natural_exp'])
def test_empty_tiles_and_rows(interpret, layout, mask_value, case):
  if not interpret and jax.default_backend() != 'tpu':
    pytest.skip('Native Splash requires TPU')
  n, width = 256, 72
  cfg = dataclasses.replace(
      splash.SplashConfig.get_default(), interpret=interpret,
      q_layout=layout, k_layout=layout, v_layout=layout,
      softmax_scale=width**-0.5, combine_log2_scale=True,
      use_base2_exp=case != 'natural_exp',
      dq_reduction_steps=3 if case == 'sparse_backward' else None,
      max_logit_const=0.0 if case == 'fixed_shift' else None,
      fuse_reciprocal=case != 'external_reciprocal',
  )
  # Reverse segments so some queries see an entirely masked first KV tile,
  # others an entirely masked last tile. Empty rows share a Q tile with valid
  # rows; all_empty also exercises completely skipped sparse Q blocks.
  q_ids = np.arange(n, dtype=np.int32) // 128
  kv_ids = 1 - q_ids
  q_ids[-16:] = 99
  if case == 'all_empty':
    q_ids[:] = 99
  if case in ('skipped_q_block', 'sparse_backward'):
    q_ids[:128] = 99
  allowed = q_ids[:, None] == kv_ids[None, :]
  if case in ('static_mask', 'skipped_q_block', 'sparse_backward', 'all_empty'):
    mask = masks.NumpyMask(allowed)
    segments = None
  else:
    mask = masks.FullMask((n, n))
    segments = splash.SegmentIds(jnp.asarray(q_ids), jnp.asarray(kv_ids))
  fn = splash.make_splash_mha_single_device(
      mask, config=cfg, mask_value=mask_value, save_residuals=True,
  )
  rng = np.random.default_rng(195)
  q, k, v, do = [jnp.asarray(rng.normal(0, .25, (1, n, width)), jnp.bfloat16)
                 for _ in range(4)]

  def forward(q, k, v):
    return fn(q, k, v, segment_ids=segments)

  def with_grad(q, k, v, do):
    (out, stats), pullback = jax.vjp(forward, q, k, v)
    gradients = pullback((do, jax.tree.map(jnp.zeros_like, stats)))
    return out, stats['logsumexp'], gradients

  out, lse, gradients = jax.jit(with_grad)(q, k, v, do)
  valid = allowed.any(-1)

  def dense(q, k, v):
    scores = jnp.einsum('hqd,hkd->hqk', q.astype(jnp.float32), k.astype(jnp.float32)) * width**-.5
    scores = jnp.where(allowed, scores, -jnp.inf)
    # Define zero output/derivative for rows without keys, without introducing
    # NaNs into either the forward computation or its autodiff transpose.
    safe_scores = jnp.where(valid[None, :, None], scores, 0.0)
    p = jnp.where(allowed, jax.nn.softmax(safe_scores, axis=-1), 0.0)
    return jnp.einsum('hqk,hkd->hqd', p, v.astype(jnp.float32))

  expected, reference_pullback = jax.vjp(dense, q, k, v)
  reference_gradients = reference_pullback(do.astype(jnp.float32))
  f32 = lambda x: np.asarray(x, dtype=np.float32)
  assert np.all(np.isfinite(f32(out)))
  np.testing.assert_array_equal(f32(out)[:, ~valid], 0)
  np.testing.assert_array_equal(f32(gradients[0])[:, ~valid], 0)
  assert np.all(np.isneginf(np.asarray(lse)[:, ~valid]))
  assert np.all(np.isfinite(np.asarray(lse)[:, valid]))
  np.testing.assert_allclose(f32(out), f32(expected), atol=2e-3, rtol=3e-2)
  for actual, expected_grad in zip(gradients, reference_gradients):
    assert np.all(np.isfinite(f32(actual)))
    np.testing.assert_allclose(f32(actual), f32(expected_grad), atol=2e-3, rtol=3e-2)


@pytest.mark.parametrize('interpret', [True, False], ids=['interpret', 'native'])
@pytest.mark.parametrize('all_empty', [False, True])
def test_skipped_rows_keep_sink_mass_without_sink_gradient(interpret, all_empty):
  if not interpret and jax.default_backend() != 'tpu':
    pytest.skip('Native Splash requires TPU')
  allowed = np.ones((256, 256), dtype=bool)
  allowed[:256 if all_empty else 128] = False
  cfg = dataclasses.replace(splash.SplashConfig.get_default(), interpret=interpret)
  fn = splash.make_splash_mha_single_device(
      masks.NumpyMask(allowed), config=cfg, save_residuals=True)
  x = jnp.ones((1, 256, 72), dtype=jnp.bfloat16) * .1
  sink = jnp.array([.7], dtype=jnp.float32)
  def forward(q, k, v, sink):
    return fn(q, k, v, sinks=sink)
  def run(x, sink):
    (out, stats), pb = jax.vjp(forward, x, x, x, sink)
    do = jnp.broadcast_to(jnp.asarray(~allowed.any(-1))[None, :, None], out.shape).astype(out.dtype)
    return out, stats, pb((do, jax.tree.map(jnp.zeros_like, stats)))
  out, stats, grads = jax.jit(run)(x, sink)
  empty = ~allowed.any(-1)
  np.testing.assert_array_equal(np.asarray(out, dtype=np.float32)[:, empty], 0)
  np.testing.assert_allclose(stats['logsumexp'][:, empty], .7, atol=1e-6)
  for grad in grads:
    np.testing.assert_array_equal(np.asarray(grad, dtype=np.float32), 0)


@pytest.mark.parametrize('mask_value', [base.DEFAULT_MASK_VALUE, -np.inf])
def test_native_ring_empty_shards_and_rows(mask_value):
  if jax.default_backend() != 'tpu' or len(jax.devices()) < 2:
    pytest.skip('Native ring regression requires at least two TPU devices')
  from tokamax._src.ops.experimental.tpu.splash_attention import ring_attention_kernel as ring
  devices = jax.devices()[:8]
  size = len(devices)
  n, width = size * 128, 72
  mesh = jax.sharding.Mesh(np.asarray(devices), ('ring',))
  spec = jax.sharding.PartitionSpec(None, 'ring', None)
  segment_spec = base.SegmentIds(jax.sharding.PartitionSpec('ring'), jax.sharding.PartitionSpec('ring'))
  cfg = dataclasses.replace(splash.SplashConfig.get_default(), use_base2_exp=False, softmax_scale=width**-.5)
  kernel = ring.make_ring_attention(
      masks.FullMask((n, n)), is_mqa=False, ring_axis='ring',
      config=cfg, q_seq_shards=size, kv_seq_shards=size, mask_value=mask_value)
  ids = jnp.arange(n) // 128
  segments = base.SegmentIds(ids.at[-16:].set(99), ids)
  allowed = np.asarray(segments.q)[:, None] == np.asarray(segments.kv)[None, :]
  valid = allowed.any(-1)
  rng = np.random.default_rng(291)
  q, k, v, do = [jnp.asarray(rng.normal(0, .25, (1, n, width)), jnp.bfloat16) for _ in range(4)]
  sharded = jax.shard_map(
      lambda kernel, q, k, v, segments: kernel(q, k, v, segments),
      mesh=mesh, in_specs=(kernel.manual_sharding_spec(), spec, spec, spec, segment_spec),
      out_specs=spec, check_vma=False)
  def forward(q, k, v):
    return sharded(kernel, q, k, v, segments)
  def both(q, k, v, do):
    out, pb = jax.vjp(forward, q, k, v)
    return out, pb(do)
  out, grads = jax.jit(both)(q, k, v, do)
  def dense(q, k, v):
    scores = jnp.einsum('hqd,hkd->hqk', q.astype(jnp.float32), k.astype(jnp.float32)) * width**-.5
    scores = jnp.where(allowed, scores, -jnp.inf)
    safe = jnp.where(valid[None, :, None], scores, 0.0)
    p = jnp.where(allowed, jax.nn.softmax(safe, axis=-1), 0.0)
    return jnp.einsum('hqk,hkd->hqd', p, v.astype(jnp.float32))
  expected, pb = jax.vjp(dense, q, k, v)
  ref_grads = pb(do.astype(jnp.float32))
  f32 = lambda x: np.asarray(x, dtype=np.float32)
  np.testing.assert_array_equal(f32(out)[:, ~valid], 0)
  np.testing.assert_array_equal(f32(grads[0])[:, ~valid], 0)
  for actual, ref in zip((out, *grads), (expected, *ref_grads)):
    assert np.all(np.isfinite(f32(actual)))
    np.testing.assert_allclose(f32(actual), f32(ref), atol=2e-3, rtol=3e-2)
