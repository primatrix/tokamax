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


@pytest.mark.parametrize("fuse_reciprocal", [False, True])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"compact_softmax_scratch": True},
        {"bwd_dv_last": True},
        {"bwd_dq_contract_ds_axis0": True},
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
        "compact_scratch",
        "dv_last",
        "dq_contract_axis0",
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
