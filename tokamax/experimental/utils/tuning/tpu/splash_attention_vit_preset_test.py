# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Check that the documented opt-in recipe selects the measured schedule."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_schedule_sweep as sweep


def test_documented_recipe_matches_measured_joint_candidate():
  original = dataclasses.replace(
      sweep._config(SimpleNamespace(sequence=32768, head_dim=72, region_trace_mode="none", interpret=False)),
      block_q=2048, block_kv=4096, block_kv_compute=1024,
      block_q_dkv=2048, block_kv_dkv=8192, block_kv_dkv_compute=2048,
      bwd_scheduler=True,
  )
  root = Path(__file__).resolve().parents[5]
  document = (root / "docs/ops/splash_attention/vit_native_layouts.md").read_text()
  recipe = document.split("```python\n", 1)[1].split("```", 1)[0]
  namespace = {"dataclasses": dataclasses, "pr13_fast_config": original}
  exec(compile(recipe, "documented-native-recipe", "exec"), namespace)
  actual = namespace["native_config"]
  expected = dataclasses.replace(
      sweep._config(SimpleNamespace(sequence=32768, head_dim=72, region_trace_mode="none", interpret=False)),
      **dict(sweep.variants("joint"))["joint_q4096_native_dq_compact_ids"],
  )
  assert actual == expected
  assert original.block_q == 2048 and original.bwd_scheduler is True
  assert actual.region_trace_mode == "none"
  assert not actual.fwd_kvmajor_fuse_normalizer
  assert not actual.fwd_staged_kv_pipeline and not actual.bwd_staged_kv_pipeline
  assert actual.bwd_head_group_size == 1


def test_native_schedule_and_profiling_are_opt_in():
  config = splash.SplashConfig(block_q=128, block_kv=128)
  for field in (
      "fwd_kvmajor_probabilities", "fwd_output_scratch_seq_minor",
      "fwd_native_output_normalization", "fwd_output_seq_minor",
      "bwd_dq_scratch_seq_minor", "bwd_dkv_scratch_seq_minor",
      "bwd_dkv_output_seq_minor", "bwd_fuse_segment_id_inputs",
      "bwd_do_seq_minor", "bwd_dq_transposed_output",
      "bwd_dq_output_seq_minor", "bwd_kv_segment_ids_seq_minor",
      "bwd_compact_segment_ids",
  ):
    assert getattr(config, field) is False, field
  assert config.region_trace_mode == "none"
