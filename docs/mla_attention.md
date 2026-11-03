# Sequence-layout MLA attention

`tokamax.experimental.mla_attention` exposes `make_static_splash_kernel` and
`make_attention`. The implementation lives alongside Splash in
`tokamax/_src/ops/experimental/tpu/splash_attention/mla_*.py` and has no MaxText
dependency. It is experimental and has a deliberately narrow geometry contract.

## Ownership and supported configuration

Tokamax owns attention's custom VJP, remote DMA, buffer lifetimes, collective
completion, mask metadata validation and the sequence/head layout boundary.
The model caller owns dispatch, configuration, rematerialization policy,
projection/gate integration and any model-specific precision constraints.

The validated full-model combination is BF16 attention inputs, CP16, four local
heads, DQK192/DV128, 262144 tokens and one local batch item per FSDP group.
It uses causal attention with int32 segment IDs, natural-log softmax,
HEAD_DIM_MINOR inputs and backward blocks Q512/KV4096/compute512 with three
dQ partials and cast-before-transpose. Factory arguments are:

```python
from tokamax.experimental.mla_attention import make_attention, make_static_splash_kernel

kernel = make_static_splash_kernel(mask, splash_config)
attention = make_attention(
    kernel, mesh=mesh, batch_axes="fsdp", record_max_logits=True,
    cp_barrier=False, bwd_input_exchange="kv_hbm_prefetch",
    fwd_input_exchange="external", fwd_output_exchange="remote_dma",
)
```

Forward owns its input exchange and output remote DMA. Backward imports K/V
through HBM staging, overlaps staging with kernel computation, and exports
dQ/dK/dV inside the kernel. Q and dO backward input exchanges remain external.
The separate MTP path uses `bwd_input_exchange="remat"`. The original remat
policy saves Splash output/LSE under `mla_splash_residuals`; this migration
does not change checkpoint tags or the caller's policy.

CP8 and other exchange modes remain available for their existing isolated
experiments. Their availability does not imply full-model acceptance. In
particular, per-head, credit, shadow and traced/serial HBM modes are diagnostic.
The production HBM selector requires CP16 and `cp_barrier=False` in MaxText.
The existing forward-only `splash_attention_a2a_cp_kernel` is an independent
implementation and is not used by this API.

The whole-source SHA256 check against `splash_attention_kernel.py` protects
private Ref ordering and buffer-lifetime assumptions. A Splash source change
requires explicit revalidation; do not simply refresh the expected hash.
`test_pinned_source_and_bad_config_rejection` exercises this guard in CI.

## Migration and validation

The stable migration at Tokamax commit `924d4d9` has a backward implementation
byte-identical to MaxText commit
`a4b23b5add26644bf39e2b620ce144d56d8e154e`. The sequence wrapper changes only
its import of that implementation. Function/kernel names, checkpoint tags,
forward sharding and backward dQ layout are preserved. The base is
`d5edf54` on `epic/absorb_a2a_into_splash_mha_kernel-based-release-v0.2`;
its Splash arithmetic is identical to the previously pinned `5936d8b`.

Before migration, MaxText training run
[41443126564](https://github.com/primatrix/maxtext/actions/runs/41443126564)
completed 20 updates on 512 JAX devices / 256 TPU chips. All 3470 recorded
training scalars matched baseline 40189767276 exactly. The matching 1070–1079
window averaged 24.9758953 s versus 25.3570593 s for baseline. This is a
cross-run observation, not a repeated controlled benchmark. The overall
workflow failed its separate GPU-reference gate; baseline's own 20 steps
produce the identical gate failure.

That device acceptance includes MaxText's C1b BF16 tail custom VJP and cannot
be attributed to this kernel alone. It is evidence for the pre-migration
implementation. A fresh AOT and a device rerun are distinct checks; old AOT
pickles must not be reused after changing repository/dependency identity.

CPU regression checks (JAX/JAXlib 0.11.0) cover abstract production tracing,
saved tensors under the original remat policy, CP peer/gradient ownership,
metadata and DMA boundary contracts:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=16 python -m pytest -q \
  tokamax/_src/ops/experimental/tpu/splash_attention/mla_fused_backward_trace_test.py \
  tokamax/_src/ops/experimental/tpu/splash_attention/mla_forward_output_exchange_test.py
```

These CPU checks do not execute TPU remote DMA or establish device accuracy.

## Experimental Q HBM import

`qkv_hbm_serial` and `qkv_hbm_prefetch` add Q import to the existing prefetched
K/V HBM path. Both keep dO's external exchange and the qualified di reduction.
The serial selector waits until the current head to send Q; the prefetch
selector sends the next head while computing the current head. Neither is
production qualified.

Each head receives Q into a distinct HBM destination. Its first program waits
for every owner's receive exactly once, before any local Q load starts. Three
VMEM banks stream Q tiles: the prefetch of program g+1 reuses the bank from g-2
only after the existing full dQ store for g-2 completes. The final two dQ
stores drain at each head boundary, and every Q send drains before return.
This requires the pipelined dQ path and the compiler global barrier.

The inner Splash arithmetic, dQ accumulation order, grid, block sizes and
checkpoint policy are unchanged. CPU tracing can check shapes and exchange
ownership; CPU TPU compilation, repeated device bitwise comparisons, group
isolation, memory analysis and full-model precision remain separate gates.
