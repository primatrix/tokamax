# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Compile the segmented ViT backward closure on a CPU host for TPU7x.

Run from a checkout with PYTHONPATH=.: python -m tokamax.benchmarks.vit_backward_aot
--mode compile --output-dir /tmp/vit-aot. Requires a compatible Linux libtpu.
The default validate mode only traces production shapes. Compile mode builds
an executable but does not execute it or establish native numerical results.
This isolates the failing kernel and its epilogue, not the surrounding train step.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Compilation-only devices come from the topology, never a physical TPU backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax.experimental import topologies

from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as masks


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "compile"), default="validate")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topology", default="tpu7x:4x8x8")
    parser.add_argument("--q-length", type=int, default=524288)
    parser.add_argument("--kv-length", type=int, default=1048576)
    parser.add_argument("--block-q", type=int, default=4096)
    parser.add_argument("--block-kv", type=int, default=8192)
    parser.add_argument("--block-kv-compute", type=int, default=1024)
    parser.add_argument("--bwd-vmem-limit-bytes", type=int)
    for name, default in (
        ("bwd-kv-unroll", False),
        ("bwd-dq-first", False),
        ("bwd-cast-before-transpose", True),
        ("bwd-scheduler", False),
        ("use-experimental-scheduler", False),
    ):
        parser.add_argument(
            "--" + name, action=argparse.BooleanOptionalAction, default=default
        )
    return parser.parse_args()


def _config(args):
    if (
        min(
            args.q_length,
            args.kv_length,
            args.block_q,
            args.block_kv,
            args.block_kv_compute,
        )
        <= 0
    ):
        raise ValueError("Lengths and blocks must be positive")
    if args.q_length % args.block_q or args.kv_length % args.block_kv:
        raise ValueError("Blocks must divide their sequence lengths")
    if args.block_kv % args.block_kv_compute:
        raise ValueError("KV compute block must divide the KV memory block")
    return splash.SplashConfig(
        block_q=args.block_q,
        block_kv=args.block_kv,
        block_kv_compute=args.block_kv_compute,
        block_q_dkv=args.block_q,
        block_kv_dkv=args.block_kv,
        block_kv_dkv_compute=args.block_kv_compute,
        q_layout=splash.QKVLayout.SEQ_MINOR,
        k_layout=splash.QKVLayout.SEQ_MINOR,
        v_layout=splash.QKVLayout.SEQ_MINOR,
        dq_reduction_steps=3,
        gpu_numerics_alignment=True,
        softmax_scale=72**-0.5,
        combine_log2_scale=True,
        use_base2_exp=True,
        bwd_scale_after_dot=True,
        bwd_kv_unroll=args.bwd_kv_unroll,
        bwd_dq_first=args.bwd_dq_first,
        bwd_cast_before_transpose=args.bwd_cast_before_transpose,
        bwd_scheduler=args.bwd_scheduler,
        use_experimental_scheduler=args.use_experimental_scheduler,
        bwd_vmem_limit_bytes=args.bwd_vmem_limit_bytes,
        segment_mask_on_partial_only=True,
        compact_stats_output=True,
        omit_unused_max_logits=True,
    )


def _inputs(q_length, kv_length, config):
    grid_capacity = (q_length // config.block_q_dkv) * (
        kv_length // config.block_kv_dkv
    )
    shapes = {
        "q": ((1, q_length, 72), jnp.bfloat16),
        "k": ((1, kv_length, 72), jnp.bfloat16),
        "v": ((1, kv_length, 72), jnp.bfloat16),
        "q_segment_ids": ((q_length,), jnp.int32),
        "kv_segment_ids": ((kv_length,), jnp.int32),
        "logsumexp_base2": ((1, q_length), jnp.float32),
        "do": ((1, q_length, 72), jnp.bfloat16),
        "di": ((1, q_length), jnp.float32),
        "active_rows": ((grid_capacity,), jnp.int32),
        "active_cols": ((grid_capacity,), jnp.int32),
        "block_mask": ((grid_capacity,), jnp.int8),
        "mask_next": ((grid_capacity,), jnp.int8),
        "num_active_blocks": ((1,), jnp.int32),
    }
    return {
        name: jax.ShapeDtypeStruct(shape, dtype)
        for name, (shape, dtype) in shapes.items()
    }


def _closure(config, *, dynamic=True):
    def backward(inputs):
        # Runtime metadata preserves the production dynamic grid and scalar prefetch.
        info = masks.MaskInfo(
            active_rows=inputs["active_rows"] if dynamic else None,
            active_cols=inputs["active_cols"] if dynamic else None,
            num_active_blocks=inputs["num_active_blocks"] if dynamic else None,
            block_mask=inputs["block_mask"],
            mask_next=inputs["mask_next"],
            partial_mask_blocks=None,
            q_sequence=None,
        )
        return splash._splash_attention_bwd_dkv(  # pylint: disable=protected-access
            inputs["q"],
            inputs["k"],
            inputs["v"],
            splash.SegmentIds(inputs["q_segment_ids"], inputs["kv_segment_ids"]),
            inputs["logsumexp_base2"],
            inputs["do"],
            inputs["di"],
            bq=config.block_q_dkv,
            bkv=config.block_kv_dkv,
            bkv_compute=config.block_kv_dkv_compute,
            is_mqa=False,
            mask_info=info,
            mask_value=splash.base.DEFAULT_MASK_VALUE,
            mask_function=None,
            config=config,
            dkv_mask_sparsity=1.0,
        )

    return backward


def main():
    args = _arguments()
    report = {"mode": args.mode, "timings_seconds": {}, "status": "started"}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def phase(name):
        start = time.monotonic()
        print(f"Starting {name}", file=sys.stderr, flush=True)
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            report["timings_seconds"][name] = elapsed
            print(f"Finished {name}: {elapsed:.3f}s", file=sys.stderr, flush=True)

    try:
        with phase("configuration"):
            config = _config(args)
            inputs = _inputs(args.q_length, args.kv_length, config)
            versions = {"python": platform.python_version()}
            for package in ("jax", "jaxlib", "libtpu", "tokamax"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            report.update(
                versions=versions,
                source_sha=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[2],
                    text=True,
                ).strip(),
                source_hashes={
                    "kernel_sha256": hashlib.sha256(
                        Path(splash.__file__).read_bytes()
                    ).hexdigest(),
                    "runner_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                },
                config=dataclasses.asdict(config),
                topology=args.topology,
                inputs={
                    name: {"shape": spec.shape, "dtype": str(spec.dtype)}
                    for name, spec in inputs.items()
                },
                dq_alias={"shape": [3, 1, args.q_length, 72], "dtype": "float32"},
                environment={
                    name: os.environ.get(name)
                    for name in ("JAX_PLATFORMS", "XLA_FLAGS", "LIBTPU_INIT_ARGS")
                },
            )
            print(json.dumps(report, default=str), flush=True)
        with phase("trace_production_shapes"):
            outputs = jax.eval_shape(_closure(config), inputs)
            report["outputs"] = [
                {"shape": x.shape, "dtype": str(x.dtype)} for x in outputs
            ]
        if args.mode == "compile":
            with phase("topology"):
                topology = topologies.get_topology_desc(
                    platform="tpu",
                    topology_name=args.topology,
                    chip_config_name="default",
                    chips_per_host_bounds=(2, 2, 1),
                    num_slices=1,
                    wrap=(True, True, True),
                )
                device = topology.devices[0]
                report["target_device"] = str(device)
                sharding = jax.sharding.SingleDeviceSharding(device)
            with phase("lower_tpu"):
                lowered = jax.jit(
                    _closure(config), in_shardings=sharding, out_shardings=sharding
                ).lower(inputs)
                if args.output_dir:
                    (args.output_dir / "lowered.mlir").write_text(lowered.as_text())
            with phase("compile_tpu"):
                compiled = lowered.compile()
                report["compiled_type"] = type(compiled).__name__
            report["execution"] = "TPU-target AOT compile only; executable not run"
        report["status"] = "success"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        print(json.dumps(report, indent=2, default=str), flush=True)
        if args.output_dir:
            (args.output_dir / "result.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n"
            )
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
