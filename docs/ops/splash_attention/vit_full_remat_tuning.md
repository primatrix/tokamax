# Experimental ViT full-rematerialization tuning

These opt-in `SplashConfig` controls support the noncausal, single-device ViT
experiment in [MaxText #1156](https://github.com/primatrix/maxtext/pull/1156).
They default to the existing behavior. This change targets `release/v0.2`, whose
`d7893a5451f2ea34a06363ac0b054a2365257f22` commit supplied the tested FP32 softmax
scaling path. Porting to the different stacked-head implementation on `main` is
separate work.

## Controls and contracts

| Controls | Purpose |
| --- | --- |
| `combine_log2_scale` | Combine FP32 softmax scale and LOG2E multiplication when using base-2 exponentials. |
| `bwd_kv_unroll` | Control unrolling of the fused backward KV compute loop. |
| `bwd_dq_first`, `bwd_dv_last` | Reorder independent dQ/dK/dV work in the fused backward kernel. |
| `bwd_cast_before_transpose` | Cast dS before its transpose for dQ. |
| `bwd_scale_after_dot` | Move softmax scaling from dS to FP32 dQ/dK accumulations. |
| `compact_stats_output` | Store replicated statistics in an 8-by-Q layout instead of Q-by-128; public statistics retain their original shapes. |
| `compact_softmax_scratch` | Use the corresponding compact layout for online-softmax scratch. This is experimental and is not enabled in the final MaxText configuration. |
| `omit_unused_max_logits` | Omit an internal max-logit output when reciprocal fusion makes it unnecessary; explicitly requested statistics and the non-fused reciprocal path still retain it. |
| `segment_mask_on_partial_only` | Skip per-element segment comparisons only on tiles classified fully allowed. **The caller must include runtime segment equality in the full-tile proof.** A FullMask combined with arbitrary segment IDs does not satisfy this contract. Partial tiles still check segment IDs. |
| `bwd_parallel_heads` | Permit parallel head dimension semantics for the supported dynamic grid with one Q head per KV head. |
| `bwd_scheduler` | Override the backward LP scheduler independently of forward scheduling. |
| `fwd_vmem_limit_bytes`, `bwd_vmem_limit_bytes` | Set per-kernel compiler VMEM limits; the requested budget must fit the target device. |

Arithmetic reordering can change floating-point rounding. These switches are
advanced experimental controls, not a claim that every combination, causal
configuration, MQA/GQA layout, Ring Attention or distributed setup is supported.
The integration exercises noncausal MHA with context parallelism 1. MaxText owns
the conservative dynamic segment metadata, batch/head merging, guarded fixed
softmax shift and custom full-remat VJP; those model-side changes are not part of
this Tokamax PR.

## Validation and performance scope

Run the CPU-interpreted regression tests from an environment with repository
test dependencies installed:

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_scale_test.py \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_vit_tuning_test.py
```

The new tests exercise segmented forward and all Q/K/V gradients, a mixed
boundary tile, compact statistics and scratch, alternate backward ordering, and
requested statistics with both reciprocal modes. Interpret mode validates
arithmetic and API behavior, not TPU scheduling or VMEM feasibility.

The kernel's executable Python AST matches the previously tested MaxText patch,
ignoring docstrings. That integration used JAX/JAXlib 0.11.0, Flax 0.12.7 and
libtpu 0.0.48 on one TPU7x JAX device, batch 2, 27 ViT layers and full remat.
The latest recorded full-model median is 2.987734 s over 20 synchronized samples
versus 7.176897 s for its default implementation. This measures the combined
MaxText and Tokamax optimization, not an isolated speedup from this PR.

The synthetic benchmark's aggregate output relative L2 is 0.985415% and all
parameter-gradient relative L2 is 0.209864%; bridge-only output error is 1.073971%.
The approximate model-side VJP has no real-data convergence validation. Detailed
configuration, reference definitions, shape and trace evidence remain in
[the MaxText experiment](https://github.com/primatrix/maxtext/tree/jzh/vit-full-remat-3s/benchmarks/ling3_vit_full_remat).
