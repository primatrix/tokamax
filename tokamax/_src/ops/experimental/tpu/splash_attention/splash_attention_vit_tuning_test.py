# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CPU-interpreted forward/backward regressions for opt-in ViT tuning."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib


_TUNING = {
    "combine_log2_scale": True,
    "bwd_kv_unroll": False,
    "bwd_dq_first": True,
    "bwd_cast_before_transpose": True,
    "bwd_scale_after_dot": True,
    "compact_stats_output": True,
    "omit_unused_max_logits": True,
    "segment_mask_on_partial_only": True,
}


def _relative_l2(actual, expected):
  actual = np.asarray(actual, np.float64)
  expected = np.asarray(expected, np.float64)
  return np.linalg.norm(actual - expected) / max(
      np.linalg.norm(expected), 1e-30
  )


@pytest.mark.parametrize("mode", ["none", "coarse", "fine"])
@pytest.mark.parametrize("do_seq_minor", [False, True])
@pytest.mark.parametrize("dq_transposed", [False, True])
def test_trace_scopes_and_seqminor_scratch_preserve_head_dim_72(mode, do_seq_minor, dq_transposed):
  """Layout/diagnostic changes must be bitwise exact at the production width."""
  arrays = [
      jax.random.normal(key, (1, 256, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(72), 4)
  ]
  q, k, v, do = arrays
  ids = np.repeat(np.array([1, 2], np.int32), [192, 64])
  segments = base.SegmentIds(jnp.asarray(ids), jnp.asarray(ids))
  mask = mask_lib.NumpyMask(ids[:, None] == ids[None, :])
  config = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, use_base2_exp=True, interpret=True,
      **_TUNING,
  )

  def run(cfg):
    kernel = splash.make_splash_mha_single_device(mask, config=cfg)
    output, pullback = jax.vjp(lambda q, k, v: kernel(q, k, v, segments), q, k, v)
    return (output, *pullback(do))

  expected = run(config)
  actual = run(dataclasses.replace(
      config, region_trace_mode=mode,
      bwd_dq_transposed_output=dq_transposed,
      bwd_do_seq_minor=do_seq_minor,
      bwd_dq_scratch_seq_minor=True, bwd_dkv_scratch_seq_minor=True,
      bwd_dkv_output_seq_minor=True, bwd_fuse_segment_id_inputs=True,
  ))
  for result, reference in zip(actual, expected):
    assert np.isfinite(np.asarray(result)).all()
    np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("dk_transposed,dv_transposed", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("seqminor", [False, True])
@pytest.mark.parametrize("dq_first", [False, True])
def test_dkv_orientation_preserves_head_dim_72(dk_transposed, dv_transposed, seqminor, dq_first):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [
      jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(27), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR if seqminor else splash.QKVLayout.HEAD_DIM_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR, v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, use_base2_exp=True, max_logit_const=0.0,
      bwd_dq_scratch_seq_minor=True, bwd_dkv_scratch_seq_minor=seqminor,
      bwd_dkv_output_seq_minor=seqminor, bwd_do_seq_minor=seqminor,
      interpret=True, **(_TUNING | dict(bwd_dq_first=dq_first)),
  )
  reference = bench._make_kernel(segments, cfg)
  _, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, bwd_dq_transposed_output=True,
      bwd_dk_transposed_output=dk_transposed, bwd_dv_transposed_output=dv_transposed,
  ))
  actual = bench._backward(candidate, residuals, do)
  for value, wanted in zip(actual, expected):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("dq_transposed", [False, True])
@pytest.mark.parametrize("order", ["dq_first", "dk_first", "dp_early", "dv_last"])
def test_qmajor_probabilities_preserve_asymmetric_tiles(dq_transposed, order):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [
      jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(27), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, use_base2_exp=True, max_logit_const=0.0,
      bwd_dq_scratch_seq_minor=True, bwd_dkv_scratch_seq_minor=True,
      bwd_dkv_output_seq_minor=True, bwd_do_seq_minor=True,
      bwd_fuse_segment_id_inputs=True, interpret=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  _, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, bwd_qmajor_probabilities=True, bwd_dq_transposed_output=dq_transposed,
      bwd_dq_first=order != "dk_first", bwd_dp_before_qk=order == "dp_early",
      bwd_dv_last=order == "dv_last",
  ))
  actual = bench._backward(candidate, residuals, do)
  for value, wanted in zip(actual, expected):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("layout", ["reference", "transposed", "qmajor"])
@pytest.mark.parametrize("dq_first", [False, True])
@pytest.mark.parametrize("dp_early", [False, True])
def test_dv_between_gradient_consumers(layout, dq_first, dp_early):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
                 for key in jax.random.split(jax.random.key(27), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR, softmax_scale=72**-0.5,
      use_base2_exp=True, max_logit_const=0.0,
      bwd_dq_scratch_seq_minor=True, bwd_dkv_scratch_seq_minor=True,
      bwd_dkv_output_seq_minor=True, bwd_do_seq_minor=True,
      bwd_fuse_segment_id_inputs=True, interpret=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  _, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, bwd_dq_transposed_output=True,
      bwd_dq_first=dq_first, bwd_dp_before_qk=dp_early,
      bwd_dv_between_dq_dk=True,
      bwd_dk_transposed_output=layout == "transposed",
      bwd_dv_transposed_output=layout == "transposed",
      bwd_qmajor_probabilities=layout == "qmajor",
  ))
  for value, wanted in zip(bench._backward(candidate, residuals, do), expected):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("seed", [27, 28])
@pytest.mark.parametrize("compute_q", [128, 256])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("unroll", [False, 2, 4])
@pytest.mark.parametrize("loop_mode", ["flat", "nested", "carry"])
def test_backward_internal_q_tiles_against_fp64(seed, compute_q, pipeline, unroll, loop_mode):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_schedule_sweep as sweep

  q, k, v, do = [jax.random.normal(key, (1, 1024, 72), jnp.bfloat16)
                 for key in jax.random.split(jax.random.key(seed), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [512, 496, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=256, block_kv=512, block_kv_compute=128,
      block_q_dkv=512, block_kv_dkv=512, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR, softmax_scale=72**-0.5,
      use_base2_exp=True, max_logit_const=0.0, interpret=True,
      **(_TUNING | sweep._DQ_DK_FIRST),
  )
  reference = bench._make_kernel(segments, cfg)
  _, residuals = bench._forward(reference, q, k, v, segments)
  grads = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, bwd_block_q_compute=compute_q, bwd_qtile_pipeline=pipeline,
      bwd_kv_unroll=unroll,
      bwd_qtile_nested=loop_mode != "flat",
      bwd_qtile_accumulator_carry=loop_mode == "carry",
  ))
  actual = bench._backward(candidate, residuals, do)
  oracle = accuracy.numpy_attention_and_gradients(q[0], k[0], v[0], do[0], ids, ids)
  if pipeline or unroll or loop_mode != "flat":
    sequential = bench._make_kernel(segments, dataclasses.replace(
        cfg, bwd_block_q_compute=compute_q, bwd_qtile_pipeline=False,
        bwd_kv_unroll=False,
    ))
    for value, expected in zip(actual, bench._backward(sequential, residuals, do)):
      np.testing.assert_array_equal(value, expected)
  # Q-compute splitting changes the dK/dV reduction association. Check all
  # gradient elements against the independent FP64 reference at the existing
  # screening bound; CPU results are not a substitute for TPU verification.
  for value, expected, fp64 in zip(actual, grads, oracle[2:]):
    assert value.shape == expected.shape and value.dtype == expected.dtype
    assert np.isfinite(np.asarray(value)).all()
    assert _relative_l2(value[0], fp64) <= _relative_l2(expected[0], fp64) * 1.001 + 1e-7


@pytest.mark.parametrize("kwargs", [
    dict(block_q_dkv=512, bwd_block_q_compute=64),
    dict(block_q_dkv=512, bwd_block_q_compute=384),
    dict(bwd_qtile_pipeline=True),
    dict(bwd_qtile_nested=True),
    dict(block_q_dkv=256, bwd_block_q_compute=128, bwd_qtile_accumulator_carry=True),
    dict(block_q_dkv=256, bwd_block_q_compute=128, bwd_staged_kv_pipeline=True),
])
def test_invalid_backward_q_compute_tile_rejected(kwargs):
  with pytest.raises(ValueError):
    splash.SplashConfig(block_q=128, block_kv=128, **kwargs)


def test_conflicting_dv_order_rejected():
  with pytest.raises(ValueError, match="dV cannot be both"):
    splash.SplashConfig(
        block_q=128, block_kv=128,
        bwd_dv_last=True, bwd_dv_between_dq_dk=True,
    )


@pytest.mark.parametrize("unroll", [True, 2, 4])
@pytest.mark.parametrize("q_block", [128, 256])
@pytest.mark.parametrize("reference_sum_axis", [False, True])
def test_forward_kvmajor_probabilities(unroll, q_block, reference_sum_axis):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy

  q, k, v, do = [
      jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(27), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=q_block, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, use_base2_exp=True, max_logit_const=0.0,
      interpret=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  output, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, fwd_kvmajor_probabilities=True, compact_softmax_scratch=True,
      fwd_output_scratch_seq_minor=True, fwd_kv_unroll=unroll,
      fwd_kvmajor_sum_in_qmajor=reference_sum_axis,
  ))
  actual_output, actual_residuals = bench._forward(candidate, q, k, v, segments)
  actual = bench._backward(reference, actual_residuals, do)
  for value in (actual_output, actual_residuals[6], *actual):
    assert np.isfinite(np.asarray(value)).all()
  np.testing.assert_array_equal(actual_output, output)
  np.testing.assert_array_equal(actual[2], expected[2])
  # Both sum expressions lower non-bitwise on CPU for this orientation. This
  # is a diagnostic reassociation, not a bitwise optimization. Bound LSE
  # rounding and test against independent FP64; TPU acceptance stays separate.
  np.testing.assert_array_max_ulp(np.asarray(actual_residuals[6]), np.asarray(residuals[6]), maxulp=1)
  oracle = accuracy.numpy_attention_and_gradients(q[0], k[0], v[0], do[0], ids, ids)
  for value, wanted, high_precision in zip(actual, expected, oracle[2:]):
    assert _relative_l2(value, wanted) < 1e-4
    ref_error = _relative_l2(wanted[0], high_precision)
    candidate_error = _relative_l2(value[0], high_precision)
    assert candidate_error <= ref_error * 1.001 + 1e-7


@pytest.mark.parametrize("unroll", [True, 4])
@pytest.mark.parametrize("q_block", [128, 256])
def test_forward_fused_normalizer_against_fp64(unroll, q_block, monkeypatch):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy

  q, k, v, do = [jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
                 for key in jax.random.split(jax.random.key(27), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=q_block, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR, softmax_scale=72**-0.5,
      use_base2_exp=True, max_logit_const=0.0, interpret=True, **_TUNING,
  )
  ref = bench._make_kernel(segments, cfg)
  output, res = bench._forward(ref, q, k, v, segments)
  grads = bench._backward(ref, res, do)
  # Interpret mode cannot reproduce TPU's default matmul approximation.
  # Separately guard the contraction request; TPU oracle checks remain required.
  original_dot = splash.lax.dot_general
  normalizer_precisions = []

  def record_dot(lhs, rhs, *args, **kwargs):
    if lhs.shape == (80, 128) and rhs.dtype == jnp.float32:
      assert lhs.dtype == jnp.float32
      normalizer_precisions.append(kwargs.get("precision"))
    return original_dot(lhs, rhs, *args, **kwargs)

  monkeypatch.setattr(splash.lax, "dot_general", record_dot)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, fwd_kvmajor_probabilities=True, fwd_kvmajor_fuse_normalizer=True,
      compact_softmax_scratch=True, fwd_output_scratch_seq_minor=True, fwd_kv_unroll=unroll,
  ))
  actual_output, actual_res = bench._forward(candidate, q, k, v, segments)
  assert normalizer_precisions
  assert all(p == splash.lax.Precision.HIGHEST for p in normalizer_precisions)
  actual_grads = bench._backward(ref, actual_res, do)
  oracle = accuracy.numpy_attention_and_gradients(q[0], k[0], v[0], do[0], ids, ids)
  for value, expected, fp64 in zip(
      (actual_output, *actual_grads), (output, *grads), (oracle[0], *oracle[2:]),
  ):
    assert np.isfinite(np.asarray(value)).all()
    assert _relative_l2(value, expected) < 1e-4
    assert _relative_l2(value[0], fp64) <= _relative_l2(expected[0], fp64) * 1.001 + 1e-7
  np.testing.assert_allclose(np.asarray(actual_res[6][0] / splash.LOG2E), oracle[1], rtol=2e-7, atol=1e-6)


@pytest.mark.parametrize("kvmajor,single_loop", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("physical_seqminor", [False, True])
@pytest.mark.parametrize("fuse_reciprocal", [False, True])
def test_native_output_drain_preserves_all_values(kvmajor, single_loop, physical_seqminor, fuse_reciprocal):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
                 for key in jax.random.split(jax.random.key(28), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR, softmax_scale=72**-0.5,
      use_base2_exp=True, max_logit_const=0.0, interpret=True,
      fwd_kvmajor_probabilities=kvmajor, fwd_output_scratch_seq_minor=True,
      fwd_kvmajor_single_loop=single_loop,
      compact_softmax_scratch=True, fuse_reciprocal=fuse_reciprocal, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  output, residuals = bench._forward(reference, q, k, v, segments)
  grads = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, fwd_native_output_normalization=True,
      fwd_output_seq_minor=physical_seqminor,
      fwd_kvmajor_single_loop=single_loop,
  ))
  actual_output, actual_residuals = bench._forward(candidate, q, k, v, segments)
  actual_grads = bench._backward(candidate, actual_residuals, do)
  for value, expected in zip(
      (actual_output, actual_residuals[6], *actual_grads),
      (output, residuals[6], *grads),
  ):
    assert value.shape == expected.shape
    assert value.dtype == expected.dtype
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("seed", [27, 28])
@pytest.mark.parametrize("q_block", [128, 256])
@pytest.mark.parametrize("mask_all_tiles", [False, True])
def test_shared_segment_loop_against_fp64(seed, q_block, mask_all_tiles):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_accuracy as accuracy

  q, k, v, do = [jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
                 for key in jax.random.split(jax.random.key(seed), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=q_block, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR, softmax_scale=72**-0.5,
      use_base2_exp=True, max_logit_const=0.0, interpret=True,
      fwd_kvmajor_probabilities=True, fwd_output_scratch_seq_minor=True,
      compact_softmax_scratch=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  output, residuals = bench._forward(reference, q, k, v, segments)
  grads = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, fwd_kvmajor_single_loop=True,
      fwd_kvmajor_mask_all_tiles=mask_all_tiles,
  ))
  actual_output, actual_residuals = bench._forward(candidate, q, k, v, segments)
  actual_grads = bench._backward(reference, actual_residuals, do)
  oracle = accuracy.numpy_attention_and_gradients(q[0], k[0], v[0], do[0], ids, ids)
  # Moving the mask branch is not assumed bitwise. Cross-candidate BF16
  # distance is not oracle error: seed 28 dK differs by 1.088e-4, but its
  # FP64-relative error ratio is only 1.000105. Use the existing independent-
  # oracle bound (0.1% + 1e-7); do not widen that accuracy bound to pass.
  for value, expected, fp64 in zip(
      (actual_output, *actual_grads), (output, *grads), (oracle[0], *oracle[2:]),
  ):
    assert np.isfinite(np.asarray(value)).all()
    assert _relative_l2(value[0], fp64) <= _relative_l2(expected[0], fp64) * 1.001 + 1e-7
  np.testing.assert_array_max_ulp(np.asarray(actual_residuals[6]), np.asarray(residuals[6]), maxulp=1)
  np.testing.assert_allclose(np.asarray(actual_residuals[6][0] / splash.LOG2E), oracle[1], rtol=2e-7, atol=1e-6)


@pytest.mark.parametrize("drain", ["reference", "native", "native_output"])
@pytest.mark.parametrize("fast_backward", [False, True])
def test_joint_runner_matches_public_custom_vjp(drain, fast_backward):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_schedule_sweep as sweep

  q, k, v, do = [jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
                 for key in jax.random.split(jax.random.key(28), 4)]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=256, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR, k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR, softmax_scale=72**-0.5,
      use_base2_exp=True, max_logit_const=0.0, interpret=True,
      **(_TUNING | sweep._FWD_KVMAJOR
         | (sweep._DQ_DK_FIRST if fast_backward else {})
         | dict(fwd_native_output_normalization=drain != "reference",
                fwd_output_seq_minor=drain == "native_output")),
  )
  kernel = bench._make_kernel(segments, cfg)

  @jax.jit
  def public_joint(q, k, v, do):
    output, pullback = jax.vjp(lambda q, k, v: kernel(q, k, v, segments), q, k, v)
    return output, *pullback(do)

  expected_values = public_joint(q, k, v, do)
  actual = jax.jit(lambda q, k, v, do: sweep.joint_values(kernel, q, k, v, segments, do))(q, k, v, do)
  for value, expected in zip(actual, expected_values):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("kwargs", [
    {"fwd_native_output_normalization": True},
    {"fwd_output_seq_minor": True},
    {"fwd_kvmajor_single_loop": True},
    {"fwd_kvmajor_mask_all_tiles": True},
])
def test_invalid_native_output_layout_rejected(kwargs):
  with pytest.raises(ValueError, match="requires"):
    splash.SplashConfig(block_q=128, block_kv=128, **kwargs)


def test_invalid_trace_mode_rejected():
  with pytest.raises(ValueError, match="Invalid region_trace_mode"):
    splash.SplashConfig(block_q=128, block_kv=128, region_trace_mode="invalid")


@pytest.mark.parametrize("fixed_shift", [False, True])
@pytest.mark.parametrize("transposed,seqminor", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("v_layout", [splash.QKVLayout.HEAD_DIM_MINOR, splash.QKVLayout.SEQ_MINOR])
@pytest.mark.parametrize("unroll", [True, 4])
def test_forward_pv_orientation_preserves_head_dim_72(fixed_shift, transposed, seqminor, v_layout, unroll):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [
      jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(27), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR, v_layout=v_layout,
      softmax_scale=72**-0.5, use_base2_exp=True,
      max_logit_const=0.0 if fixed_shift else None,
      interpret=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  output, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, fwd_pv_transposed_output=transposed, fwd_output_scratch_seq_minor=seqminor,
      fwd_kv_unroll=unroll,
  ))
  actual_output, actual_residuals = bench._forward(candidate, q, k, v, segments)
  actual = bench._backward(reference, actual_residuals, do)
  for value, wanted in zip((actual_output, actual_residuals[6], *actual), (output, residuals[6], *expected)):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("unroll", [False, 2, 4])
@pytest.mark.parametrize("kv_block", [256, 512])
def test_staged_forward_preserves_outputs_and_gradients(unroll, kv_block):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [
      jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(27), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=kv_block, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=256, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, use_base2_exp=True, max_logit_const=0.0,
      interpret=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  output, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, fwd_staged_kv_pipeline=True, fwd_kv_unroll=unroll,
  ))
  actual_output, actual_residuals = bench._forward(candidate, q, k, v, segments)
  actual = bench._backward(reference, actual_residuals, do)
  for value, wanted in zip((actual_output, actual_residuals[6], *actual), (output, residuals[6], *expected)):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("kv_block", [256, 512])
@pytest.mark.parametrize("unroll", [False, 2, 4, 8])
def test_single_segment_mask_body_preserves_gradients(unroll, staged, kv_block):
  from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_pr13_benchmark as bench

  q, k, v, do = [
      jax.random.normal(key, (1, 512, 72), jnp.bfloat16)
      for key in jax.random.split(jax.random.key(27), 4)
  ]
  ids = jnp.asarray(np.repeat(np.array([1, 2, 3, 0], np.int32), [256, 240, 8, 8]))
  segments = base.SegmentIds(ids, ids)
  cfg = splash.SplashConfig(
      block_q=128, block_kv=256, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=kv_block, block_kv_dkv_compute=128,
      q_layout=splash.QKVLayout.SEQ_MINOR,
      k_layout=splash.QKVLayout.SEQ_MINOR,
      v_layout=splash.QKVLayout.SEQ_MINOR,
      softmax_scale=72**-0.5, use_base2_exp=True, max_logit_const=0.0,
      bwd_dq_scratch_seq_minor=True, bwd_dkv_scratch_seq_minor=True,
      bwd_dkv_output_seq_minor=True, bwd_do_seq_minor=True,
      bwd_fuse_segment_id_inputs=True, interpret=True, **_TUNING,
  )
  reference = bench._make_kernel(segments, cfg)
  _, residuals = bench._forward(reference, q, k, v, segments)
  expected = bench._backward(reference, residuals, do)
  candidate = bench._make_kernel(segments, dataclasses.replace(
      cfg, bwd_single_segment_mask_body=True, bwd_kv_unroll=unroll,
      bwd_staged_kv_pipeline=staged,
  ))
  actual = bench._backward(candidate, residuals, do)
  for value, wanted in zip(actual, expected):
    assert np.isfinite(np.asarray(value)).all()
    np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("fuse_reciprocal", [False, True])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"bwd_kv_unroll": 2},
        {"bwd_kv_unroll": 4},
        {"fwd_loop_carry": True},
        {"fwd_loop_carry": True, "compact_softmax_scratch": True},
        {"compact_softmax_scratch": True},
        {"bwd_dv_last": True},
        {"bwd_dq_contract_ds_axis0": True},
        {"bwd_dq_transposed_output": True},
        {
            "bwd_dq_transposed_output": True,
            "bwd_dq_scratch_seq_minor": True,
            "bwd_dkv_scratch_seq_minor": True,
            "bwd_do_seq_minor": True,
        },
        {"bwd_do_seq_minor": True},
        {
            "bwd_do_seq_minor": True,
            "bwd_keep_kv_seq_minor": True,
            "v_layout": splash.QKVLayout.SEQ_MINOR,
        },
        {
            "bwd_dq_contract_ds_axis0": True,
            "bwd_keep_kv_seq_minor": True,
            "q_layout": splash.QKVLayout.SEQ_MINOR,
            "k_layout": splash.QKVLayout.SEQ_MINOR,
            "v_layout": splash.QKVLayout.SEQ_MINOR,
        },
        {"bwd_dp_before_qk": True},
        {"bwd_head_group_size": 2},
        {"bwd_reuse_bf16_probabilities": True},
        {"bwd_compact_segment_ids": True},
        {"bwd_dq_scratch_seq_minor": True},
        {"bwd_dkv_scratch_seq_minor": True},
        {
            "bwd_dkv_scratch_seq_minor": True,
            "bwd_dkv_output_seq_minor": True,
        },
        {"bwd_fuse_segment_id_inputs": True},
    ],
    ids=[
        "combined",
        "kv_unroll_2",
        "kv_unroll_4",
        "fwd_loop_carry",
        "fwd_loop_carry_compact",
        "compact_scratch",
        "dv_last",
        "dq_contract_axis0",
        "dq_transposed_output",
        "dq_transposed_output_seqminor",
        "do_seq_minor",
        "do_and_v_seq_minor",
        "keep_kv_seq_minor",
        "dp_before_qk",
        "head_group_2",
        "reuse_bf16_probabilities",
        "compact_segment_ids",
        "dq_scratch_seq_minor",
        "dkv_scratch_seq_minor",
        "dkv_scratch_and_output_seq_minor",
        "fuse_segment_id_inputs",
    ],
)
def test_tuning_preserves_segmented_outputs_and_all_gradients(
    fuse_reciprocal, extra
):
  keys = jax.random.split(jax.random.key(42), 4)
  # Four heads cover two distinct grouped-head BlockSpec windows.
  num_heads = 4 if extra.get("bwd_head_group_size", 1) > 1 else 1
  q, k, v, do = [
      jax.random.normal(key, (num_heads, 256, 128), jnp.bfloat16)
      for key in keys
  ]
  q, k = q * 0.25, k * 0.25
  ids = np.repeat(np.array([1, 2], np.int32), [192, 64])
  segment_ids = base.SegmentIds(jnp.asarray(ids), jnp.asarray(ids))
  # Include segment equality in the tile classification: full tiles are proven
  # homogeneous, and the boundary tile must still perform exact masking.
  mask = mask_lib.NumpyMask(ids[:, None] == ids[None, :])
  config = splash.SplashConfig(
      block_q=128,
      block_kv=256,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=256,
      block_kv_dkv_compute=128,
      softmax_scale=128**-0.5,
      use_base2_exp=True,
      fuse_reciprocal=fuse_reciprocal,
      interpret=True,
  )
  tuned = dataclasses.replace(config, **(_TUNING | extra))

  def run(cfg, *, save_residuals=False):
    kernel = splash.make_splash_mha_single_device(
        mask, config=cfg, save_residuals=save_residuals
    )
    return lambda q, k, v: kernel(q, k, v, segment_ids)

  expected, reference_pullback = jax.vjp(run(config), q, k, v)
  actual, pullback = jax.vjp(run(tuned), q, k, v)
  assert _relative_l2(actual, expected) < 0.01
  actual_grads = pullback(do)
  expected_grads = reference_pullback(do)
  for actual_grad, expected_grad in zip(actual_grads, expected_grads):
    assert np.isfinite(np.asarray(actual_grad)).all()
    assert _relative_l2(actual_grad, expected_grad) < 0.01

  exact_flags = {
      flag: False
      for flag in (
          "bwd_compact_segment_ids",
          "bwd_kv_unroll",
          "fwd_loop_carry",
          "bwd_do_seq_minor",
          "bwd_dq_scratch_seq_minor",
          "bwd_dkv_scratch_seq_minor",
          "bwd_dkv_output_seq_minor",
          "bwd_fuse_segment_id_inputs",
      )
      if extra.get(flag, False)
  }
  if exact_flags:
    noncompact = dataclasses.replace(tuned, **exact_flags)
    noncompact_out, noncompact_pullback = jax.vjp(run(noncompact), q, k, v)
    np.testing.assert_array_equal(actual, noncompact_out)
    for actual_grad, noncompact_grad in zip(
        actual_grads, noncompact_pullback(do)
    ):
      np.testing.assert_array_equal(actual_grad, noncompact_grad)

  # Omitting internal max logits must not remove explicitly requested stats.
  (_, actual_stats), _ = jax.vjp(run(tuned, save_residuals=True), q, k, v)
  _, expected_stats = run(config, save_residuals=True)(q, k, v)
  for name in ("max_logits", "logsumexp"):
    assert actual_stats[name] is not None
    assert actual_stats[name].shape == (num_heads, 256)
    np.testing.assert_allclose(
        actual_stats[name], expected_stats[name], rtol=1e-5, atol=1e-5
    )
