# ViT Splash attention tuning

These additions target `release/v0.2` and support the ViT integration in
[MaxText #1183](https://github.com/primatrix/maxtext/pull/1183).
Existing `SplashConfig` defaults retain their previous paths.

## Controls and contracts

| Controls | Purpose |
| --- | --- |
| `combine_log2_scale` | Combine FP32 softmax scale and LOG2E multiplication for base-2 exponentials. |
| `bwd_kv_unroll`, `bwd_dq_first`, `bwd_cast_before_transpose`, `bwd_scale_after_dot` | Tune backward loop execution and arithmetic order; floating-point rounding may change. |
| `compact_stats_output` | Use an 8-by-Q statistics layout while preserving public output shapes. |
| `omit_unused_max_logits` | Omit internal max logits when unnecessary; explicitly requested statistics remain available. |
| `segment_mask_on_partial_only` | Skip segment checks on full tiles only when the caller has proved runtime segment equality over the entire tile. A FullMask alone does not establish this. |
| `bwd_scheduler` | Override the backward LP scheduler. |
| `fwd_vmem_limit_bytes`, `bwd_vmem_limit_bytes` | Set compiler VMEM budgets appropriate for the target device. |

Supported BF16 MHA configurations with sequence-minor Q/K/V, zero fixed shift,
combined base-2 scaling, segment IDs and no partial mask blocks automatically use
native layouts. Forward also requires no sinks or runtime shift; backward requires
multiple KV compute tiles. Compact softmax scratch is internal to native forward.
Other configurations retain the general implementation.

Segment IDs may be shared `[sequence]` or grouped `[groups, sequence]`, with groups
dividing both Q and KV head counts. A `[groups, grid]` block mask describes a union
grid; zero skips a tile for that group. `manual_sharding_spec()` replicates groups
and partitions the grid along the query-sequence sharding axis. Callers construct
the mask metadata and supply matching segment IDs.

## Validation

Run the arithmetic and two-device sharding regressions with test dependencies:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 python -m pytest -q \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_scale_test.py \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_vit_tuning_test.py \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_layout_test.py \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_grouped_segments_test.py
```

These tests cover outputs, Q/K/V gradients, statistics, grouped masks and sequence
sharding. CPU interpret skips dynamic-grid GQA because of cross-program alias
limitations. It does not validate TPU compilation, scheduling, VMEM or performance.
Full-model benchmark history belongs to the
[MaxText experiment](https://github.com/primatrix/maxtext/tree/jzh/vit-full-remat-3s/benchmarks/ling3_vit_full_remat);
its older measurements do not establish performance or accuracy for the current
Tokamax revision.
