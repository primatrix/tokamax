# ViT Splash region tracing: 2026-09-22

## Scope and controls

This investigation measures BF16 noncausal segmented ViT attention on **one
v7x JAX device**, batch 2, 16 heads, sequence 32768, head dimension 72. The Falcon
reservation is v7x-8 (4 chips / 8 visible devices), but the kernel benchmark is
not distributed. Segment lengths are 16384, 16368, 8, 8; the last segment is
padding. Forward tiles are Q=1024 / KV=8192 / compute-KV=256; backward tiles are
Q=4096 / KV=8192 / compute-KV=1024. This is a kernel investigation, not a change
to model rematerialization or training precision.

All source changes are on `codex/vit-pr13-50pct`, based on the v0.2.1 line.
The measurements below use JAX/JAXlib 0.11.0 and libtpu 0.0.44.1. Do not mix them
with earlier full-model timings using a different libtpu build.

`region_trace_mode` (commit `363a921`) selects:

- `none` (default): no added in-kernel scope markers.
- `coarse`: initialization, the entire inner KV loop (full/partial masks
  separately), and output formatting/store regions. No scopes between inner
  dot, softmax, vector, or accumulator operations.
- `fine`: additionally names QK, masking, softmax, PV, accumulator updates,
  backward recomputation, dV, dP, softmax-grad, dQ, and dK.

The benchmark flag is `--region-trace-mode {none,coarse,fine}`. Capture a trace
with `--profile-dir PATH --profile-repeats 3` and libtpu flags
`--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true`.
Use a separate process with region tracing disabled for performance promotion.

## Instrumentation sanity check

Commit `8609c96` has the original unconditional fine scopes. The exact-layout
configuration additionally enables sequence-minor dQ/dK/dV scratch, dK/dV
outputs and fused segment-ID inputs. The arithmetic and tiling are unchanged.

| Source/configuration | Region tracing | Forward ms | Backward ms | Sum ms |
| --- | --- | ---: | ---: | ---: |
| `8609c96`, PR13 settings | off | 21.160 | 50.159 | 71.319 |
| `8609c96`, exact-layout settings | off | 20.768 | 49.577 | 70.345 |
| `8609c96`, exact-layout settings | fine/on | 134.973 | 120.085 | 255.058 |

These are medians of synchronized host measurements, not summed trace regions.
The trace-on run used 7 timing repeats; trace-off used 10. The large trace-on
slowdown exists outside the XProf capture window too: timing happens before
`jax.profiler.trace`. Merely retaining the named scopes with the libtpu region
flag disabled restores approximately the earlier 21/50 ms performance.
This establishes severe instrumentation perturbation; it does **not** establish
the exact compiler or runtime mechanism responsible.

Evidence:

- Trace-on: `exp-201qqtwflg`, artifact `art-eaajwlgouz`; operator analysis
  `an-fmqakb6yhy`, LLO summary `an-k2hb17b2af`.
- Trace-off: `exp-dbsr48zagu`, artifact `art-01xkuswxl8`; declared measurements
  in `an-kkwufm1nir` (`metrics.json`, `report.md`). The standard operator plugin
  reported NO_METRICS because this run wrote `pr13.jsonl` and `seqminor.jsonl`
  instead of its required `rank-*/benchmark/metrics.jsonl` path. That was an
  artifact naming error, not an experiment/runtime failure.
- Corrected interval analysis: `an-pa0colt2gv` (`timeline.json`).

## What the fine trace actually establishes

Use TPU 0's XLA TraceMe lane for measured regions. The separate Tensor Core
instruction-marker lane reports zero device duration (rendered as 1 ps), so it
cannot be summed as region elapsed time. The `dP` region is nested within
`softmax_grad`; interval union removes that double count.

Select each call using its enclosing **device XLA module**, not the host
StepTraceAnnotation interval. The first host step extends into the following
device forward execution; using it creates a spurious 122 ms forward gap.

| First instrumented call | Forward | Backward |
| --- | ---: | ---: |
| XLA module duration, ms | 133.518 | 118.742 |
| First-to-last named scope span, ms | 132.805 | 117.482 |
| Named-scope interval union, ms | 85.197 | 103.182 |
| Uncovered span, ms | 47.609 | 14.300 |
| QK / recomputation occurrences | 65536 | 4096 |

The largest uncovered boundary groups (instrumented call only):

- Forward: accumulator -> next input/state scope 30.644 ms; output -> next
  initialization 10.108 ms; accumulator -> output 6.168 ms.
- Backward: dQ output -> next dQ initialization 5.536 ms; dK -> dQ output
  4.769 ms; dQ initialization -> input scope 2.172 ms; dK/dV output -> next
  dK/dV initialization 1.805 ms.

These are **uncovered software-scope intervals**, not proven DMA waits or MXU
idle intervals. Likewise, names ending in `_mxu` include surrounding preparation
or accumulation; their durations are not hardware MXU utilization. The generic
XLA event `sf-local-wait` spans the entire backward kernel in this trace and must
not be interpreted as 117 ms of idle work. No real-duration DMA hardware lane
has yet been matched to these boundaries.

The coarse/no-scope experiment is `exp-8emfljk5ob`, pinned to `363a921`, with the
same exact-layout configuration. It succeeded, and coarse tracing is not
materially slower than the control (20.841 / 49.676 ms, versus no-scope
21.267 / 49.564 ms). Coarse timings are in `an-pc9u2planx`, the interval
analysis is `an-twim4lqtmd`, and the LLO summary is `an-cz0ae3ey0n`. No-scope
timings are from the Falcon experiment log: this two-variant run overwrote
its root metrics file with the last variant; both XProf captures remain intact.

### Low-perturbation coarse result

| First coarse-traced call | Forward | Backward |
| --- | ---: | ---: |
| Device XLA module, ms | 20.071 | 48.403 |
| First-to-last internal scope span, ms | 19.389 | 47.159 |
| KV loop regions combined, ms | 18.687 | 46.687 |
| Initialization + output store regions, ms | 0.273 | 0.272 |
| Uncovered intervals inside scope span, ms | 0.428 | 0.200 |
| Loop fraction of internal scope span | 96.4% | 99.0% |

The forward loops comprise 1504 full and 544 partial-mask memory tiles; backward
has 352 full and 160 partial-mask tiles. The largest forward boundary totals
are loop -> next loop (0.232 ms across the full/partial combinations) and
output -> initialization (0.192 ms). Backward dQ output -> next initialization
totals 0.148 ms; dK/dV output -> initialization totals 0.049 ms.

Therefore outer tile transitions/output boundaries are not the main exposed
latency in this configuration. This **does not mean inner-loop MXU/vector
overlap is perfect**: nearly all of that work is inside a coarse loop region.
Investigate dot preparation, vector/layout work, and scheduling within that
loop. The fine trace cannot be used to quantify their uninstrumented costs.

The final-LLO audit `an-j7uf5uhxv7` inspects one final dump per compilation,
not the sum of intermediate compiler passes. The dumped function takes VMEM
Refs and contains body operations (`vmatprep`, `vmatmul`, `vxpose`, loads/stores,
and trace markers). It does not contain the outer DMA pipeline; zero DMA/wait
counts in these dumps are not evidence of zero runtime transfers/waits.

One concrete layout hypothesis is `dO`: unlike Q/K/V in this configuration,
its backward BlockSpec is head-dimension-minor. The opt-in `bwd_do_seq_minor`
trial feeds `[head_dim, Q]` directly to equivalent dP/dV contractions. It keeps
the arithmetic/dtypes and all tiling unchanged. Accept this only after an
on-TPU bitwise gradient check and a same-config timing comparison.

## Validation and next decision

The ViT tuning regression suite passes 37 tests. New tests compare outputs and
all dQ/dK/dV arrays bitwise at head dimension 72 for none/coarse/fine tracing
plus the exact-layout flags and dO layout variants. These are CPU-interpreted arithmetic checks, not
TPU compiled numerical certification or training convergence validation.

The next scheduling decision must come from low-perturbation evidence: compare
coarse loop body time against initialization/output/grid-transition time, then
map exposed intervals to compiler or hardware evidence. A dot-only microbench
cannot prove attention overlap is saturated. No 40–50% kernel gain is claimed
by the tracing/layout work above.
