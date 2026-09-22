# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Check replay capture boundaries independently of TPU profiler availability."""

import contextlib
import json
from pathlib import Path

import pytest

from tokamax.experimental.utils.tuning.tpu import splash_attention_vit_schedule_sweep as sweep


def install_profiler(monkeypatch):
  state = dict(active=False, paths=[], invocations=[], steps=[], options=[])

  @contextlib.contextmanager
  def trace(path, **kwargs):
    assert not state["active"]
    state["paths"].append(Path(path))
    state["options"].append(kwargs)
    state["active"] = True
    try:
      yield
    finally:
      state["active"] = False

  @contextlib.contextmanager
  def step(name, step_num):
    assert state["active"]
    state["steps"].append((name, step_num))
    yield

  monkeypatch.setattr(sweep.jax.profiler, "trace", trace)
  monkeypatch.setattr(sweep.jax.profiler, "StepTraceAnnotation", step)
  monkeypatch.setattr(sweep.bench, "_ready", lambda value: value)
  return state


@pytest.mark.parametrize("counts", [None, [0, 1, 3, 9, 3]])
@pytest.mark.parametrize("kernel_profiling", [False, True])
def test_replay_capture_boundaries(monkeypatch, tmp_path, counts, kernel_profiling):
  state = install_profiler(monkeypatch)
  argument = object()

  def candidate(value):
    assert value is argument
    state["invocations"].append(state["active"])
    return value

  sweep.capture_profiles(candidate, (argument,), out=tmp_path, name="control",
                         phase="backward", repeats=3, repeat_counts=counts,
                         kernel_profiling=kernel_profiling)
  expected_counts = [3] if counts is None else counts
  assert state["invocations"] == [
      active for count in expected_counts for active in [False] + [True] * count
  ]
  assert state["steps"] == [
      ("vit_splash_backward", i) for count in expected_counts for i in range(count)
  ]
  rows = [json.loads(line) for line in
          (tmp_path / "profiling/capture-metadata.jsonl").read_text().splitlines()]
  assert len(rows) == len(expected_counts) == len(set(state["paths"]))
  for index, (row, count) in enumerate(zip(rows, expected_counts)):
    assert row["status"] == "captured"
    assert row["capture_index"] == index
    expected_options = {"tpu_enable_kernel_profiling": True} if kernel_profiling else {}
    assert row["advanced_configuration"] == expected_options
    if kernel_profiling:
      assert state["options"][index]["profiler_options"].advanced_configuration == expected_options
    else:
      assert state["options"][index] == {}
    assert row["requested_replays"] == row["completed_replays"] == count
    assert tmp_path / row["profile_dir"] == state["paths"][index]
    assert (row["host_requested_monotonic_ns"] <= row["host_entered_monotonic_ns"]
            <= row["host_loop_finished_monotonic_ns"] <= row["host_returned_monotonic_ns"])
  if counts is None:
    assert state["paths"] == [tmp_path / "profiling/xprof/control"]


@pytest.mark.parametrize("repeats,counts", [(-1, None), (3, []), (3, [1, -1])])
def test_invalid_replay_plan_has_no_side_effects(tmp_path, repeats, counts):
  def candidate():
    pytest.fail("invalid capture plan executed the candidate")

  with pytest.raises(ValueError, match="profile replay counts"):
    sweep.capture_profiles(candidate, (), out=tmp_path, name="control",
                           phase="backward", repeats=repeats, repeat_counts=counts)
  assert not list(tmp_path.iterdir())


def test_failed_capture_records_completed_replays(monkeypatch, tmp_path):
  state = install_profiler(monkeypatch)
  calls = 0

  def candidate():
    nonlocal calls
    calls += 1
    if calls == 3:  # Warmup, first successful replay, then failure.
      raise RuntimeError("replay failed")

  with pytest.raises(RuntimeError, match="replay failed"):
    sweep.capture_profiles(candidate, (), out=tmp_path, name="control",
                           phase="forward", repeat_counts=[3])
  row = json.loads((tmp_path / "profiling/capture-metadata.jsonl").read_text())
  assert row["status"] == "failed"
  assert row["completed_replays"] == 1 and row["requested_replays"] == 3
  assert "host_returned_monotonic_ns" in row
  assert not state["active"]


def test_profile_modes_reuse_candidate_and_keep_capture_paths_unique(monkeypatch, tmp_path):
  state = install_profiler(monkeypatch)
  argument = object()

  def candidate(value):
    assert value is argument
    state["invocations"].append(state["active"])
    return value

  modes = [*sweep.PROFILE_MODES, "default"]
  sweep.capture_profiles(candidate, (argument,), out=tmp_path, name="control",
                         phase="joint", repeat_counts=[0, 1], profile_modes=modes)
  rows = [json.loads(line) for line in
          (tmp_path / "profiling/capture-metadata.jsonl").read_text().splitlines()]
  assert len(rows) == len(set(state["paths"])) == 12
  assert state["invocations"] == [False, False, True] * len(modes)
  for index, row in enumerate(rows):
    mode_index = index // 2
    mode = modes[mode_index]
    config = sweep.profile_configuration(mode)
    assert row["profile_mode"] == mode
    assert row["profile_mode_index"] == mode_index
    assert row["variant"] == "control"
    assert row["advanced_configuration"] == config
    assert row["requested_replays"] == row["completed_replays"] == index % 2
    assert f"mode_{mode_index:02d}_{mode}" in row["profile_dir"]
    if config:
      assert state["options"][index]["profiler_options"].advanced_configuration == config
    else:
      assert state["options"][index] == {}


@pytest.mark.parametrize("modes,kernel", [([], False), (["unknown"], False),
                                        (["default", "unknown"], False),
                                        (["default"], True)])
def test_invalid_profile_modes_have_no_side_effects(tmp_path, modes, kernel):
  def candidate():
    pytest.fail("invalid profiler plan executed the candidate")

  with pytest.raises(ValueError, match="profile mode"):
    sweep.capture_profiles(candidate, (), out=tmp_path, name="control",
                           phase="joint", profile_modes=modes, kernel_profiling=kernel)
  assert not list(tmp_path.iterdir())


def test_profile_presets_have_documented_options():
  assert sweep.profile_configuration("default") == {}
  assert sweep.profile_configuration("kernel") == {"tpu_enable_kernel_profiling": True}
  assert sweep.profile_configuration("kernel_compute") == {
      "tpu_enable_kernel_profiling": True, "tpu_trace_mode": "TRACE_COMPUTE"}
  periodic = sweep.profile_configuration("periodic")
  assert periodic == {
      "tpu_enable_periodic_counter_sampling": True,
      "tpu_tc_perf_counter_sampling_options": (
          "interval_us:1 scaling:0 counter_size_bits:2 "
          "indices:10 indices:11 indices:56 indices:57 indices:58"),
  }
  assert sweep.profile_configuration("periodic_compute") == {
      **periodic, "tpu_trace_mode": "TRACE_COMPUTE"}
