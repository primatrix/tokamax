# Opt-in native-layout BF16 ViT attention

This change is stacked on `jzh/vit-full-remat-splash` (PR #13,
`2036a9c2f1b901dfccaa20a10feaf235473f865f`), which is based on v0.2.1.
It packages the kernel measured at research commit
`5038a875d5cc78ca4aaa85fdcc25800c0df90252`.
The kernel file is byte-for-byte identical to that measured snapshot
(Git blob `52dce0b2c584e01fe817f64a5e0e782042e50052`).

No new tuning flag is enabled by default. The measured configuration is
`joint_q4096_native_dq_compact_ids`; other default-off scheduling probes
remain in the snapshot for reproducibility, not as recommended settings.
The full research diary, unrelated dot microbenchmarks and later profiler
counter experiments are not included.

## What changes

- Forward computes FP32 probabilities in [KV, Q] orientation and keeps
  output accumulation, normalization and physical writeback sequence-minor.
  The wrapper restores the public logical output shape.
- Backward keeps dO and dQ/dK/dV scratch/output in sequence-minor layouts,
  computes transposed dQ without materializing the large dS transpose,
  and reduces the segment-ID input footprint with compact/native layouts.
- Forward and backward use the measured tile sizes and scheduling below.
  Optional coarse/fine named scopes support region-trace analysis; they are
  disabled in the reported no-scope timings.

BF16 Q/K/V/dO and FP32 probability/accumulator storage are retained. There
is no FP8 path, new BF16 cast of forward probabilities, remat change,
additional compute device, sequence shortening or dropped padding.
FP32 storage does not imply every mixed-precision MXU contraction is an
exact FP32 contraction; this retains PR13's contraction precision choices.

## Measured configuration

Apply these overrides only to PR13's existing guarded zero-shift/base-2
fast configuration, with sequence-minor Q/K/V, `softmax_scale=72**-0.5`,
`use_base2_exp=True`, `combine_log2_scale=True`, and
`max_logit_const=0.0`:

```python
native_config = dataclasses.replace(
    pr13_fast_config,
    block_q=4096, block_kv=4096, block_kv_compute=256,
    block_q_dkv=4096, block_kv_dkv=8192, block_kv_dkv_compute=1024,
    use_experimental_scheduler=False, bwd_scheduler=False,
    bwd_kv_unroll=False, bwd_dq_first=False,
    fwd_kvmajor_probabilities=True, compact_softmax_scratch=True,
    fwd_output_scratch_seq_minor=True, fwd_native_output_normalization=True,
    fwd_output_seq_minor=True, bwd_dq_scratch_seq_minor=True,
    bwd_dkv_scratch_seq_minor=True, bwd_dkv_output_seq_minor=True,
    bwd_fuse_segment_id_inputs=True, bwd_do_seq_minor=True,
    bwd_dq_transposed_output=True, bwd_dq_output_seq_minor=True,
    bwd_kv_segment_ids_seq_minor=True, bwd_compact_segment_ids=True,
)
```

The caller retains PR13's mask metadata, `dq_reduction_steps=3`, 60/63 MiB
forward/backward VMEM budgets and safety fallback. The production-shape
qualification is BF16, N=32768, D=72, ungrouped MHA. This is not an
unconditional replacement for arbitrary attention shapes, masks or GQA.
The kernel checks the native-forward preconditions and rejects unsupported
native gradient output configurations.

MaxText already has an opt-in adapter in
[`segment_splash.py`](https://github.com/primatrix/maxtext/blob/1a7c9bb6a33e140db1da61e6625a32eb220515ce/src/maxtext/layers/segment_splash.py#L129).
It checks the original Q/K range and V bound before using fixed shift;
unsafe values and the original-forward layers retain their old path.
Replacing the Tokamax dependency alone does not enable this schedule.

## End-to-end evidence

Fresh same-process comparison: Falcon `exp-9okx4bonhs`,
artifact `art-fttkelb2vr`, analysis `an-0wkbjmwsvd`.
Both declared analysis outputs were read through Falcon.

The workload is one actual TPU7x device, batch 2, sequence 32768, 27 ViT
layers, 16 heads of dimension 72, BF16 computation, and unchanged full remat.
It times ViT + bridge + embedding fusion + synthetic loss + all parameter
gradients, including materialized training outputs, but **not** optimizer
updates or the language decoder. Both cases use 3 warmups and 20 timed
synchronized executions, without callbacks or a profiler.

| Metric | PR13 optimized | Native layout |
| --- | ---: | ---: |
| Median latency | 2989.606652 ms | 2671.870958 ms |
| Standalone output relative L2 vs reference | 1.005243603% | 1.011532918% |
| Training output relative L2 vs reference | 1.009102911% | 1.014986355% |
| Parameter-gradient relative L2 vs reference | 0.299086375% | 0.311092706% |
| All gradients finite / fusion error | yes / 0 | yes / 0 |

This is **10.6280% lower latency**, saving 317.735694 ms in this measured
scope, not a 10% guarantee for every model or full VL training.
Earlier reproduction `exp-p7hj45ac2d` measured 2672.709698 ms versus the
historical `exp-dmcaqltvwj` baseline 2989.801322 ms; the fresh paired
comparison above is the primary claim.

Raw paired measurements and individual checks are archived in
[`native-precision-repair-result.json`](https://github.com/primatrix/maxtext/blob/3fa8a4542/benchmarks/ling3_vit_full_remat/native-precision-repair-result.json).
The same model/input hashes, initialized parameters, original-forward layer
selection and full-remat policy were retained.

The separate public-VJP operator experiment `exp-l5c175b961` measured
61.599117 ms versus its live PR13 control 70.163140 ms (12.2059% lower
latency), with full-length independent FP32 reference checks on all 32
merged heads. This operator result is not the end-to-end result.

## Numerical interpretation and limits

The attention equations are unchanged, but reducing compute-KV from 1024
to 256 and changing probability orientation/reduction layout changes
floating-point accumulation. Results are not bitwise identical to PR13.
The standalone reference-relative error increases by **0.006289314
percentage points**, not by the approximately 0.956810% direct distance
between the two outputs. Those are different measurements.

Forward-only ablation reproduces the added output difference; backward-only
ablation leaves outputs unchanged. Same-input real-QKV replay at layers 5
and 26, head 0, found 36 and 59 changed elements out of 2,359,296, all one
BF16 ULP. Independent CPU FP64 checks of 36 changed captured layer-5
coordinates found values near BF16 rounding midpoints: 21 were closer to
PR13 and 15 to the native result. This supports rounding-boundary flips,
not a claim that either implementation is uniformly nearer mathematical
truth. It does not identify the exact FP32 instruction behind every flip.

The QKV observer perturbed the full-model outputs, so its sampled layer
trajectory is not used as proof of the uninstrumented model's propagation.
Its initial public-LSE diagnostic also converted natural LSE twice; those
LSE measurements were rejected and are not precision evidence here.
Details: [actual-QKV report](https://github.com/primatrix/maxtext/blob/0258834a9/benchmarks/ling3_vit_full_remat/native-precision-diagnostic.md).

Both PR13 and this candidate miss the historical strict overall-output
<1% gate on this exact fixture. That gate is not silently relaxed or claimed
to pass. The requested review basis is relative to PR13's existing error,
with the observed changes disclosed. There is no real-data, multi-step
training/convergence, all-shape or universal no-regression claim.

## Reproduction

Validated runtime: JAX/JAXlib 0.11.0, libtpu 0.0.48, Flax 0.12.7 for the
MaxText benchmark. The source commit is authoritative; the distribution
metadata still reports Tokamax 0.0.12 on this v0.2.1-based branch.

CPU regressions:

On this packaging branch, the interpreted kernel/oracle suite passed 358
tests; the documented-preset/default-off checks and existing scaling suite
passed another 6. CPU tests do not replace the recorded TPU measurements.

```sh
JAX_PLATFORMS=cpu python -m pytest --noconftest -q \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_vit_tuning_test.py \
  tokamax/experimental/utils/tuning/tpu/splash_attention_vit_accuracy_test.py
```

TPU public-VJP operator reproduction (one compute device):

```sh
JAX_PLATFORMS=tpu python -m \
  tokamax.experimental.utils.tuning.tpu.splash_attention_vit_schedule_sweep \
  --phase joint --sequence 32768 --heads 32 --head-dim 72 --seed 29 \
  --warmup 3 --repeats 30 --region-trace-mode none \
  --variants pr13 joint_q4096_native_dq_compact_ids \
  --oracle-heads 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
                 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --output-dir /tmp/vit-native-validation
```

The runner writes `benchmark/metrics.jsonl` and
`benchmark/schedule-details.jsonl`, including all candidate/reference
errors. Its operator control is the unchanged PR13 kernel with the
standalone operator runner's tiling; it is distinct from the MaxText
PR13 schedule in the end-to-end table. Use `--region-trace-mode coarse`
or `fine` only for separate profiling, not to replace no-scope timings.
Non-bitwise rows are deliberately labelled `needs_accuracy_review`,
not automatically accepted.

The end-to-end runner, exact variant configs and preserved reference
workflow are in
[MaxText source 1a7c9bb6a](https://github.com/primatrix/maxtext/tree/1a7c9bb6a33e140db1da61e6625a32eb220515ce/benchmarks/ling3_vit_full_remat).
Its `ling3_vit_native_precision_diagnostic.py` uses unobserved full-model
executables and records PR13-relative and independent-reference errors
separately. The paired experiment above pins PR13 for reference generation
and this kernel snapshot for the candidate process.
