# Opt-in FA4 forward arithmetic boundaries

Base: `v0.2.2`, commit `5936d8b11b1a092b7bf830be5fb53cf0812abf22`.

This option implements three observed boundaries of the GPU Plus-VL ViT
FA4 forward specialization: FP32 QK accumulation followed by combined
`scale * log2(e)` scaling, base-2 exponentiation with online maximum
normalization, and BF16 rounding of the unnormalized probability operand
before PV. The normalization denominator continues to use FP32 exponent
values, and PV requests FP32 accumulation. Defaults are unchanged.

## Direct Splash use

```python
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash

config = splash.SplashConfig.get_default().with_gpu_fa4_forward_numerics(
    softmax_scale=72**-0.5,
)
attention = splash.make_splash_mha_single_device(mask, config=config)
out = attention(q, k, v)  # BF16 MHA, original unscaled Q/K/V
```

Pass the original scale independently of storage padding: a 72-wide head
padded to 128 still uses `72**-0.5`. Mask, packing, tile sizes, and gradient
accumulation configuration remain the caller's responsibility.

MaxText calls Splash directly, so its caller must apply this preset to the
resolved `SplashConfig` and **leave `vit_guarded_fixed_shift=None`**. Merely
setting `tokamax_splash_use_base2_exp=true` can also activate MaxText's
fixed-shift fast path. That is a different implementation and is deliberately
rejected here. This branch does not edit MaxText or change the LLM MLA path.
The generic `tokamax.dot_product_attention` wrapper does not automatically
select this option; do not pre-scale Q and also pass a kernel softmax scale.

## Evidence and limits

GPU job `344815500`, original image
`acr-bj-ant-registry-vpc.cn-beijing.cr.aliyuncs.com/aii/aistudio:17840149-20260813162840`,
captured `observed-000.mlir` and PTX from FA4 interface SHA-256
`f243f5830b63e8ec881a0b2cd8fe4c0b3a86f41442052b31371391fd61db27bb`.
The captured invocation is BF16 `[1024,1,128]`, scale `72**-0.5`, noncausal,
with zero-padded active width 72. Reported GPU is NVIDIA L20C, capability 10.0;
do not extrapolate its instruction identity to every baseline specialization.

Relevant MLIR lines in that artifact: host log2(e) constant 8104, combined
scale 8276, launch argument 8280, score FMA 3691, exp2 3707, and probability
BF16 conversion 3837. PTX line 4106 has `ex2.approx.ftz.f32`.

This is **not full numerical parity**:

- GPU computes scaled-score subtraction using packed FMA and an unscaled
  row maximum. Splash still uses its existing scaled-score maximum and
  subtraction; backend FMA contraction and rounding are not certified equal.
- TPU exponent approximation and matrix multiplication lowering may differ.
- Tile sizes, reduction ordering, probability padding and real input packing
  need matched native-device verification.
- Splash's existing custom backward remains in use: it recomputes FP32
  probabilities, casts P for dV, and casts dS for dQ/dK. Its reconstruction
  uses the combined-scale configuration. The complete native FA4 backward
  instruction graph is not yet verified, so backward parity is **unknown**.
  This option does not differentiate a newly substituted dense softmax.
- No changes to loss, optimizer, CP topology, or MLA are included.

## Validation

CPU Pallas interpretation tests exercise 72/128-wide heads, one/two KV
tiles, an independent blockwise probability-rounding oracle, unrounded
FP32 logsumexp, partial causal masks, and all Q/K/V gradients against a
dense reference with explicit tolerances. Existing scale tests remain intact.
These tests execute the kernel logic on CPU; they do not certify native TPU
compilation or native GPU/TPU error bounds.

```sh
pytest -q tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_fa4_numerics_test.py \
  tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_scale_test.py
```

Before production adoption, capture both native paths on identical Q/K/V,
masks, scale and upstream gradient. Compare outputs, logsumexp and dQ/dK/dV,
including multiple segments, extreme logits, padding, and empty rows. Attribute
differences separately to exponent choice, scale rounding and probability
rounding rather than treating this preset as a proven precision fix.
