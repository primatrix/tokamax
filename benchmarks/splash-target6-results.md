# Splash Attention 6% recovery

## Result

The opt-in training configuration recovers more than the requested 6% at the
production kernel shape in both causal modes:

| Mode | Baseline fwd+bwd | Optimized fwd+bwd | Latency reduction |
|---|---:|---:|---:|
| Unpacked causal | 31.206 ms | 28.840 ms | 7.58% |
| Packed causal (25 segments) | 33.682 ms | 31.492 ms | 6.50% |

Every one of the five interleaved rounds was faster. Each aggregate contains
100 synchronized samples per version, after compilation and warmup were
excluded.

## Configuration

Baseline is Tokamax `d7893a5451f2ea34a06363ac0b054a2365257f22` with the
existing production tiles. The optimized variant changes only:

```python
compact_residuals=True
qk_diag_skip=True
qk_diag_grid=4
dq_reduction_steps=3
```

All new options remain disabled by default. The three changes are independent:

1. `compact_residuals` stores the row-wise softmax residuals as `[H, 8, T]`
   instead of replicating each scalar across `[H, T, 128]`.
2. `qk_diag_skip` divides causal diagonal blocks into four bands and omits QK
   matmuls wholly above the causal diagonal. Entries that can affect the result
   are unchanged.
3. The existing supported `dq_reduction_steps=3` path reduces one partial-dQ
   buffer and its reduction traffic. It changes the floating-point reduction
   order, but not the mathematical result.

## Reproduction environment

- TPU: TPU7x dev-pod, 4 chips / 8 visible devices; benchmark uses device 0
- Python 3.12.12
- JAX 0.11.1, jaxlib 0.11.1, libtpu 0.0.46
- BF16, `B=2`, `H=64`, `T=8192`, `QK=192`, `V=128`
- Q/K normal standard deviation 0.25; V/cotangent standard deviation 1
- native FP32 `softmax_scale=192**-0.5`, base-2 exponential path
- forward tiles `2048/2048/1024`; backward tiles `2048/2048/512`
- layouts: Q/K sequence-minor, V head-dimension-minor
- packed IDs: `floor(token * 25 / 8192)`
- five warmups before timing and before every round; five rounds of 20 calls;
  baseline/optimized order alternates by round; every call is synchronized

Commands from the repository root in the isolated TPU environment:

```bash
PYTHONPATH=$PWD venv/bin/python -P benchmarks/splash_compact_ab.py \
  --out target6-ab.json
PYTHONPATH=$PWD venv/bin/python -P benchmarks/splash_compact_ab.py \
  --packed --seed 29 --out target6-ab-packed.json
```

Raw samples and environment metadata are in `target6-ab.json` and
`target6-ab-packed.json`.

## Correctness

The real-TPU regression suite passed 16/16 cases. It covers packed/unpacked,
natural/base-2 exponentials, fused/unfused reciprocal, native/no scale, compact
residual public stats, and forward plus all three input gradients. The complete
optimized configuration was additionally checked against a pure-JAX FP32
causal-attention forward/VJP reference under the established Splash BF16
tolerances.

In the production A/B, output, dK and dV are bitwise identical. Packed dQ is
also bitwise identical. Unpacked dQ has maximum absolute difference
`0.0001220703125`, caused by the existing three-way dQ reduction order; all
values are finite and the pure-JAX FP32 reference tests pass. The diagonal skip
and compact residual changes themselves are bitwise exact in their dedicated
TPU comparisons.

## Profile evidence

Device-0 means over five profiled steps:

| Device event | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Forward residual kernel | 9.620 ms | 8.981 ms | -6.64% |
| Fused dKV kernel | 18.256 ms | 16.477 ms | -9.75% |
| dQ reduction | 0.813 ms | 0.657 ms | -19.28% |
| Final copy | 0.778 ms | 0.777 ms | unchanged |

The compact forward output metadata changes the two FP32 residuals from
`[2,64,8192,128]` to `[2,64,8,8192]`, reducing their combined storage from
1 GiB to 64 MiB. The complete XPlane and compressed trace inputs are retained
outside Git because they total more than 50 MiB; the checked-in A/B JSON files
contain all timing samples and environment metadata.
