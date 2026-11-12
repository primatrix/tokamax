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

The geometry is derived from the call: CP is the mesh context size, local
heads are global heads / CP (any positive count), head widths are the Q/K and
V input widths, and the sequence must be a multiple of CP x `block_kv_dkv` so
every rank owns whole K/V tiles. Tile sizes come from the Splash config; Splash
and Pallas enforce TPU block alignment. Diagonal dQ finalization requires
`block_kv_dkv == 8 * block_q_dkv` and three dQ partials. Inputs are BF16 with
int32 segment IDs, causal attention, natural-log softmax, HEAD_DIM_MINOR
layouts and cast-before-transpose. Mask
metadata may be concrete NumPy (checked element-wise) or traced, e.g. when the
kernel is built inside topology AOT; traced metadata is checked against the
static causal schedule proof for the call's sequence length. The production
full-model configuration is CP16, four local heads, 262144 tokens.
Factory arguments are:

```python
from tokamax.experimental.mla_attention import make_attention, make_static_splash_kernel

kernel = make_static_splash_kernel(mask, splash_config)
attention = make_attention(
    kernel, mesh=mesh, batch_axes="fsdp", record_max_logits=True,
    cp_barrier=False, bwd_input_exchange="kv_hbm_prefetch",
)
```

Forward uses XLA all-to-all for its input and output exchanges. Backward imports K/V
through HBM staging, overlaps staging with kernel computation, and exports
dQ/dK/dV inside the kernel. Q and dO backward input exchanges remain external.
The separate MTP path uses `bwd_input_exchange="remat"`. The original remat
policy saves Splash output/LSE under `mla_splash_residuals`; this migration
does not change checkpoint tags or the caller's policy.

Backward input exchanges are `remat` (rematerialized head-layout Q/K/V) and
`kv_hbm_prefetch` / `kv_hbm_serial` (in-kernel K/V import; both require
`cp_barrier=False`, i.e. the compiler global barrier). Earlier diagnostic
modes (packed/separate/remote_dma inputs, per-head, credit and shadow imports,
traced HBM regions) were removed. The forward remote-DMA input/output exchanges
(`fwd_input_exchange`/`fwd_output_exchange`) were also removed: production uses
the external all-to-all, and the full-model output remote DMA measured +6.8 ms
per step, within run-to-run spread.
`make_splash_mha_a2a_cp` (`splash_attention_a2a_cp_kernel.py`) is a separate
entry point: its forward fuses the Q/K/V and O exchanges into one Pallas call,
and its backward uses the same fused MLA backward. Its `bwd_input_exchange`
selects how the backward gets head-layout Q/K/V:

- `head` (default): the forward's received buffers. Pass
  `head_checkpoint_names=(q, k, v)` so a remat policy can save them; unnamed
  buffers can only be rebuilt by re-running the forward kernel.
- `kv_hbm_prefetch`: saves the sequence shards; Q uses an XLA all-to-all and
  the kernel imports K/V. No GQA.
- `a2a`: saves the sequence shards; Q/K/V use XLA all-to-alls.

The backward's O residual is the sequence-layout output, already named with
`residual_checkpoint_name` like the LSE. The backward forms Di there and
exchanges only Di, so O is saved once. To avoid re-running the forward kernel,
the remat policy must save `residual_checkpoint_name`; with `kv_hbm_prefetch`
or `a2a` that is the only extra residual.

The whole-source SHA256 check against `splash_attention_kernel.py` protects
private Ref ordering and buffer-lifetime assumptions. A Splash source change
requires explicit revalidation; do not simply refresh the expected hash.
`test_pinned_source_and_bad_config_rejection` exercises this guard in CI.

## Migration and validation

The backward implementation is byte-identical to MaxText commit
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
  tokamax/_src/ops/experimental/tpu/splash_attention/mla_a2a_cp_backward_trace_test.py
```

These CPU checks do not execute TPU remote DMA or establish device accuracy.
