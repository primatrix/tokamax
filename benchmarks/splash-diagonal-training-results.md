# Causal diagonal training experiments (2026-09-10)

## Scope and reproduction

This is a kernel-level experiment, not a MaxText step-time measurement.
The checkout is `fdz/splash-fp32-scale-perf`, based on PR12 commit
`d40cd97cf3eaf5c7866908c0825099e5273feab4`, with the local diagonal-skip changes.
JSON records the actual kernel source SHA256, all samples, and package versions.

- Pod: `default/ldl-kda-performance-test-slice-0-0-jbql5`.
- Isolated directory: `/tmp/fdz-sv-validation-0910`; original Pod environment untouched.
- Python 3.12.12, JAX/JAXLIB 0.11.1, libtpu 0.0.46, BF16.
- 4-chip TPU Pod, 8 visible devices; benchmark executes on device 0 only.
- Inputs `[B,H,T,D]`: B=2, H=64, T=8192, Dq=Dk=192, Dv=Dout=128.
- NumPy default_rng: seed 17 unpacked, 29 packed; Q/K normal std=0.25,
  V and output cotangent normal std=1. No input tensor sharding.
- Causal mask, offset 0. Packed segment IDs: `floor(token * 25 / 8192)`.
- Native FP32 scale `192**-0.5`, base-2 exp, Q/K SEQ_MINOR, V HEAD_DIM_MINOR.
- Forward tiles 2048/2048/1024; backward tiles 2048/2048/512.
- Baseline enables compact residuals, QK skip grid=4 and dQ reduction=3.
- Five warmups; five alternating-order rounds of 20 synchronized calls each.
  Compilation excluded; reported times are medians of 100 calls per variant.

From the isolated checkout, execute with `.venv/bin/python` and `PYTHONPATH=.`:

```bash
python benchmarks/splash_compact_ab.py --sv-only --sv-grid 2 --out sv-grid2-unpacked.json
python benchmarks/splash_compact_ab.py --sv-only --sv-grid 2 --packed --seed 29 --out sv-grid2-packed.json
python benchmarks/splash_compact_ab.py --dv-only --out dv-unpacked.json
python benchmarks/splash_compact_ab.py --dv-only --packed --seed 29 --out dv-packed.json
python benchmarks/splash_compact_ab.py --bwd-only --out bwd-unpacked.json
python benchmarks/splash_compact_ab.py --bwd-only --packed --seed 29 --out bwd-packed.json
```

`--dv-only` and `--bwd-only` compare against **baseline plus SV grid=2**.
Their gains are incremental, not measured relative to the original FP32-scale kernel.
Add `--forward-only` for forward-only timing; do not use that mode to evaluate
backward optimizations. All new kernel switches remain disabled by default.

## Completed measurements

| Change | Input/mode | Baseline ms | Candidate ms | Reduction |
|---|---|---:|---:|---:|
| SV grid=4, grouped prefixes | unpacked fwd+bwd | 28.837 | 28.672 | 0.58% |
| SV grid=4, grouped prefixes | packed fwd+bwd | 31.465 | 31.580 | -0.37% |
| SV grid=2, grouped prefixes | unpacked forward | 9.050 | 8.848 | 2.23% |
| SV grid=2, grouped prefixes | packed forward | 10.921 | 10.498 | 3.87% |
| SV grid=2, grouped prefixes | unpacked fwd+bwd | 28.876 | 28.712 | 0.57% |
| SV grid=2, grouped prefixes | packed fwd+bwd | 31.601 | 31.353 | 0.79% |
| dV suffix crop, on top of SV2 | unpacked fwd+bwd | 28.896 | 28.762 | 0.46% |
| dV suffix crop, on top of SV2 | packed fwd+bwd | 31.187 | 30.961 | 0.73% |
| dV+dP+dK+dQ crop, on top of SV2 | unpacked fwd+bwd | 28.842 | 27.095 | 6.06% |
| dV+dP+dK+dQ crop, on top of SV2 | packed fwd+bwd | 31.339 | 29.387 | 6.23% |

Small differences need repeat-run confirmation; percentages from separate runs
must not be added to claim a measured total gain.

The full-backward candidate passed production checks with bitwise-identical
output, dQ, dK and dV in BOTH input modes (all finite, max_abs=0).
The kernel SHA256 is
`acff4aba39c3f52e6d923f9af85cc25f608019e553050f73166289db99b92a0a`.

Regression evidence: 76 existing/SV/dV cases passed in `test-bwd.log`; the
initial full-backward case found an unsupported zero-sized Mosaic vector.
After fixing the first compute tile to use the original contraction, all 16
full-backward FP32-reference cases passed in `test-bwd-all.log`.
An extra 16 cases without QK skip were subsequently added but have not run yet.
The final whole-suite rerun and second timing repetition could not start because
another workload acquired the TPU (`test-release.log` records the lock error).
No other workload was killed. Do not report this as a completed stability study.

Local raw evidence is retained in `diagonal-results-0910/`, including the final
JSON results, full-backward reference-test log, and package freeze.
The complete archive (all intermediate logs plus source snapshots) remains on
the Pod at `/tmp/fdz-diagonal-results-0910.tar.gz`, SHA256
`fe62a1cd5d835a163435e988fd2f7c6e131fd7c3fad03a2628114b5235ada151`.
Its bulk download hit a kubectl stream EOF; the partial local archive is clearly
named `results.incomplete.tar.gz` and must not be used as complete evidence.
No MaxText end-to-end run is included in this experiment.

## Precision and limitations

FP32 scale and accumulation types are unchanged. SV contractions with shortened
K dimensions are not universally bitwise identical: a 512-token / 256-compute
test found two output differences out of 131072 values, maximum 1.52588e-5.
The tests compare both variants with an independent FP32 reference using the
existing Splash BF16 tolerances, rather than claiming bitwise equivalence.

The complete-config reference sequence is now 2048 with 512-token KV blocks:
more than three KV blocks are required to actually enable three-way dQ reduction.
An earlier 1536-token test used only three blocks and took the automatic fallback.

The backward optimization removes only wholly causal-masked Q prefixes inside
diagonal blocks. It does not implement segment-aware scheduling, change the
attention mask, lower precision, or optimize non-diagonal blocks.

`dv_diag_skip=True` crops only `P.T @ dO`. `bwd_diag_skip=True` includes that
optimization and additionally crops `V @ dO.T`, `dS @ Q`, and `dS.T @ K`.
For compute tile i the omitted Q prefix is `i * block_kv_dkv_compute`.
The first tile has no prefix and uses the full contraction; non-diagonal blocks
retain the original implementation. Both switches require square backward
blocks and a square, aligned, zero-offset CausalMask.
