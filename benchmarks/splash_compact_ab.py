# Copyright 2026 Tokamax authors.
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

"""Interleaved real-TPU A/B for precision-preserving Splash optimizations."""

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.experimental.tpu.splash_attention import base
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as sk,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_mask as masks,
)

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--packed", action="store_true")
p.add_argument("--seed", type=int, default=17)
a = p.parse_args()
rows = []


def log(x):
  rows.append(x)
  print(json.dumps(x), flush=True)
  Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


log(
    {
        "python": platform.python_version(),
        "packages": {
            k: importlib.metadata.version(k) for k in ["jax", "jaxlib", "libtpu"]
        },
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "devices": list(map(str, jax.devices())),
        "seed": a.seed,
        "packed": a.packed,
        "shape": [2, 64, 8192, 192, 128],
        "dtype": "bf16",
        "distribution": "q/k normal std=.25, v/cotangent normal std=1",
        "warmup": 5,
        "rounds": 5,
        "repeat": 20,
        "device": "device0 only, no sharding",
        "timing": "compile excluded; synchronized each call; alternating round order",
    }
)
rng = np.random.default_rng(a.seed)


def array(d, amp):
  return jax.device_put(
      jnp.asarray(
          rng.standard_normal((2, 64, 8192, d)).astype(np.float32) * amp, jnp.bfloat16
      ),
      jax.devices()[0],
  )


q, k, v, do = array(192, 0.25), array(192, 0.25), array(128, 1), array(128, 1)
args = (q, k, v, do)
ids = jnp.arange(8192, dtype=jnp.int32) * 25 // 8192
segments = base.SegmentIds(ids, ids) if a.packed else None
base = sk.SplashConfig(
    block_q=2048,
    block_kv=2048,
    block_kv_compute=1024,
    block_q_dkv=2048,
    block_kv_dkv=2048,
    block_kv_dkv_compute=512,
    q_layout=sk.QKVLayout.SEQ_MINOR,
    k_layout=sk.QKVLayout.SEQ_MINOR,
    v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
    use_base2_exp=True,
    softmax_scale=192**-0.5,
    use_experimental_scheduler=True,
)
log(
    {
        "config": str(base),
        "mask": "causal; packed uses floor(token*25/8192)" if a.packed else "causal",
    }
)
compiled = []
outputs = []
for optimized in [False, True]:
  config = dataclasses.replace(
      base,
      compact_residuals=optimized,
      qk_diag_skip=optimized,
      qk_diag_grid=4,
      dq_reduction_steps=3 if optimized else None,
  )
  kernel = sk.make_splash_mha_single_device(
      masks.CausalMask((8192, 8192)), config=config
  )

  def fwd(q, k, v, kernel=kernel):
    return jax.vmap(lambda q, k, v: kernel(q, k, v, segment_ids=segments))(q, k, v)

  def fb(q, k, v, do, fwd=fwd):
    y, pb = jax.vjp(fwd, q, k, v)
    return y, pb(do)

  fn = jax.jit(fb).lower(*args).compile()
  outputs.append(jax.block_until_ready(fn(*args)))
  for _ in range(5):
    jax.block_until_ready(fn(*args))
  compiled.append(fn)
for name, x, y in zip(
    ["output", "dq", "dk", "dv"],
    jax.tree.leaves(outputs[0]),
    jax.tree.leaves(outputs[1]),
):
  max_abs = float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))))
  check = {
      "tensor": name,
      "equal": bool(jnp.all(x == y)),
      "finite": bool(jnp.all(jnp.isfinite(y))),
      "max_abs": max_abs,
  }
  log(check)
  assert check["finite"] and (
      check["equal"] or (name == "dq" and max_abs <= 0.000125)
  ), check
del outputs
all_samples = [[], []]
for round in range(5):
  for i in [0, 1] if round % 2 == 0 else [1, 0]:
    fn = compiled[i]
    for _ in range(5):
      jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(20):
      t = time.perf_counter()
      jax.block_until_ready(fn(*args))
      samples.append((time.perf_counter() - t) * 1000)
    all_samples[i].extend(samples)
    log(
        {
            "round": round,
            "optimized": bool(i),
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        }
    )
b, c = map(statistics.median, all_samples)
log(
    {
        "done": True,
        "baseline_ms": b,
        "optimized_ms": c,
        "latency_reduction_percent": 100 * (b - c) / b,
    }
)
