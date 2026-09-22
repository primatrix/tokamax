# ViT Splash region tracing: 2026-09-22

## Current result

The public custom-VJP joint benchmark now measures **62.555 ms versus live
PR13 70.098 ms**, 10.8% lower latency / 1.121x throughput (seed 29, region
tracing disabled, `exp-qi1gilizpq`). It combines native KV-major Q4096 /
memory-KV4096 / compute-KV256 forward, native normalization/physical output
layout, and sequence-minor backward scratch/dO with transposed dQ and
**dK before dQ**. Q2048/compute-KV512 is a second retained setting at about
62.7–63.1 ms. These are measurements of one
compiled forward/backward invocation, not added independent measurements.
It is not a full-model or rematerialization benchmark. The combined
attention target of 20–30% improvement has **not** been reached.

Forward-only native physical output measures 18.906 ms (Q2048) or 18.754 ms
(Q4096/memory-KV4096), versus matched live PR13 21.221/20.826 ms. Coarse trace
attributes a concrete gain to output formatting: Q2048 drain time falls
from 0.833 to 0.025 ms while KV-loop time stays near 17 ms. Independent
backward-only no-scope timing remains 45.313 versus 49.738 ms.

The subsequent branchless-forward, internal-Q backward pipeline and
small-body backward-unroll screens add **no retained speedup**. The best
new backward case is Q1024/unroll-4 at 46.070 ms versus the retained
45.201 ms in the same experiment. All remain default-off. The expanded
CPU kernel/oracle suite passes 292 tests; measured TPU screens preserve finite
outputs and essentially unchanged sampled independent-oracle errors.

All new orientation/pipeline controls remain default-off. BF16 inputs,
FP32 softmax, and model rematerialization are unchanged. The joint candidates
are non-bitwise. Seed 29 now checks all **32 full-length FP32 oracle heads**:
output and gradient maximum absolute errors are unchanged on every head;
their aggregate L2 errors are essentially unchanged. LSE's maximum absolute
error never increases, though per-head L2 error can differ. This is numerical
screening, not training-convergence certification. Seed-27/no-scope
reproduction measures 63.079 ms for Q2048
and 62.570 ms for Q4096/memory-KV4096, versus live PR13 70.272/70.286 ms.
The latter is 11.0% lower latency / 1.123x throughput. All submitted
experiments and requested analyses in this iteration are terminal and read.
The DEFAULT-precision fused-normalizer probe is rejected for TPU LSE/gradient
accuracy regression. Its faster timings are excluded from retained results.

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

In the coarse capture the generic XLA `net-router-barrier` event spans the
forward kernel, just as `sf-local-wait` spans backward. Neither event label
alone establishes network communication or device idleness.

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

### dO layout result

`exp-p9hpmj75rg` (source `99cc365`, artifact `art-xx9fh7kcjj`) passed the full
production-shape on-TPU bitwise checks for dQ, dK and dV; all are finite. The
same-process reference backward median is 49.941 ms and the sequence-minor dO
median is 49.672 ms (20 samples each), only about 0.5% apart. This is not a
meaningful progress claim toward the 40–50% goal; the option remains disabled
by default. Evidence: `an-8874gr5eti/evidence.json`, operator analysis
`an-jixi3p0um6`, region analysis `an-l9f02389hp`.

The coarse loop span is essentially unchanged: 46.705 ms inside backward KV
loops. The device module is 48.147 ms, versus 48.403 ms in the previous capture;
the roughly 0.27 ms difference is outside the loop and is consistent with
avoiding the dO layout copy, not improving internal MXU/vector overlap.

The stage-mapped final-LLO audit `an-70ah1z3gru` assigns 10240 of 13824 static
transpose-family matches to the dQ scope for the exact-layout baseline. That
early parser normalizes `llo.vxpose` and its result-readout operation into one
family; these are not 13824 independent transpose instructions. They also
include both mask branches and are **not dynamic counts or a latency
fraction**. They motivate a separate opt-in `bwd_dq_transposed_output`
trial: compute `dQ.T = K.T @ dS`, then accumulate in the existing sequence-minor
scratch, rather than forming the large dS transpose for `dQ = dS.T @ K`.
The reference check disables only this orientation change, retaining the same
other tuning/layout flags.

The full-shape dQ trial is `exp-0cw1kyzxf0`, source `4b803ec`, artifact
`art-q8qyvvh9t1`. It retains sequence-minor dO in both candidate and reference;
the orientation flag is the only difference in the same-process backward
comparison. Precision is checked before timing, and any bit-pattern or finite
check failure aborts the benchmark rather than relaxing the threshold.

**Rejected:** the full-shape TPU check reported dQ and dV not bitwise equal;
dK remained bitwise equal and all three remained finite. The experiment failed
at the explicit accuracy gate, before timing/profiling, not during compilation
or provisioning. The magnitude and cause of the discrepancies have not been
quantified, so this is not a claim of a particular model-level error rate.
There is no latency result for this candidate. Its 43 CPU tests had passed;
CPU interpretation was insufficient to certify compiled TPU numerics.
The orientation flag and implementation were subsequently removed from the
active branch. Commit `4b803ec` preserves the rejected trial for investigation.

The counter inventory `an-5leius2bxu` also exposes actual MXU busy-state and XLU
transpose counters in the dO capture. They are encoded as zero-duration events
with `counter_value`, not Chrome `ph=C` events. Their measurement window is not
per named region, and chip/die attribution must be resolved before deriving
kernel utilization. In particular, do not divide the entire-capture MXU busy
counter by a single forward/backward duration or count another die's activity
as attention parallelism.

## Validation and next decision

The retained ViT tuning regression suite passes 45 tests at `41877a1`. New tests compare outputs and
all dQ/dK/dV arrays bitwise at head dimension 72 for none/coarse/fine tracing
plus the exact-layout flags and dO layout variants. These are CPU-interpreted arithmetic checks, not
TPU compiled numerical certification or training convergence validation.

The next scheduling decision must come from low-perturbation evidence: compare
coarse loop body time against initialization/output/grid-transition time, then
map exposed intervals to compiler or hardware evidence. A dot-only microbench
cannot prove attention overlap is saturated. No 40–50% kernel gain is claimed
by the tracing/layout work above.

## Scheduling screen after the region audit

`exp-v9le0h9j6n` (`1c1a2f7`, artifact `art-afybaczla0`) screens 12 backward
configurations on the production shape and seed 27. Each candidate has 12 timing
samples and is followed by a six-sample live PR13 reference check. Forward
residuals are shared unchanged. Backward tiling changes replace the residual
MaskInfo as well as the configuration; otherwise they would not test the
requested tiling correctly. No BF16 probability reuse is enabled.

| Configuration | Backward ms | Live PR13 ms | Full-shape precision |
| --- | ---: | ---: | --- |
| PR13 | 50.187 | 50.175 | bitwise |
| Sequence-minor layouts, including dO | 49.510 | 49.981 | bitwise |
| Layouts + unroll 2 | 65.588 | 50.140 | bitwise |
| Layouts + unroll 4 | 58.623 | 50.249 | bitwise |
| Layouts + compute-KV 512 | 51.533 | 50.035 | review required |
| Layouts + compute-KV 512 + unroll 2 | 50.208 | 49.978 | review required |
| Layouts + Q 2048 | 51.134 | 49.980 | review required |
| Layouts + Q 2048 + unroll 2 | 50.181 | 50.261 | review required |
| Layouts + Q 2048 + compute-KV 512 + unroll 2 | 51.472 | 50.046 | review required |
| Layouts + scheduler | 49.834 | 49.726 | bitwise |
| Layouts + compute-KV 512 + scheduler | 51.352 | 50.200 | review required |
| Layouts + early dP / late dV | 55.475 | 49.969 | review required |

All gradients are finite. Non-bitwise candidates are diagnostic measurements,
not automatically accepted. For example compute-KV 512 changes 2394 of
75497472 dQ elements, relative L2 `2.14896e-5`, max absolute `0.000244140625`;
dK/dV remain bitwise. Q 2048 produces relative L2 errors between `7.39487e-6`
and `1.60743e-5`. These are differences from PR13, not errors against an FP32
oracle or a training-convergence certification. None of these candidates has a
material speed benefit, so there is no reason to trade precision for them.

Evidence: operator `an-2a2iwpmbj5`; numerical details and raw timing samples
`an-nm9p3xz5rf/details.json`; LLO inventory `an-y5u381yz1i`; final-LLO and device
scope audit `an-94t2rxgo7e/audit.json`. The built-in operator inventory reports
zero traces for this nested variant-directory layout; the custom audit confirms
three `.trace.json.gz` plus three `.xplane.pb` files. Zero in that inventory does
not mean no profiles were captured.

The unroll-2 trace is especially informative: its first device module takes
64.106 ms, but the coarse KV-loop regions total 46.569 ms, close to PR13's
46.528 ms (48.681 ms device module). Uncovered time within the first-to-last
scope span grows from 0.202 ms to 16.266 ms. The regression is therefore exposed
outside the named loop regions, not measured as slower dots inside the loop.
This does not identify DMA, instruction loading, or another scheduling mechanism;
the new configuration also needs a trace-disabled control before attribution.

The control `exp-t5dvkzvg39` (`3468420`, artifact `art-8k6i9uo5v6`) disables both
the region flag and named scopes, uses 20 candidate samples, and confirms the
regression: PR13 49.879 ms, layouts 49.550 ms, unroll-2 65.475 ms and unroll-4
57.915 ms. All gradients remain bitwise. Thus the unroll regression is not
caused by region tracing. Evidence: `an-5dd2b6y0te/report.md`, operator
`an-h7lpc62i1b`, LLO inventory `an-mktew80r4n`.

The detailed boundary audit `an-ukdq2frpfy/audit.json` locates unroll-2's gaps:
5.426 ms from full-loop end to dQ output; 6.307 ms from dQ output to the next
dQ initialization; 2.060 ms across dK/dV output-to-initialization; 2.472 ms from
dQ initialization to the partial loop. Individual gaps are about 15–17 us.
Instruction markers in one largest gap jump from address `0xffe4` to `0x29d`
after about 15.55 us. This motivates a code-footprint/overlay hypothesis, but
does not prove it: `an-lc8fk3dr5c` finds no actual events on the TC Overlay lane.

The broader hardware-counter audit `an-cnfj0heksw/audit.json` provides stronger
evidence for instruction-memory traffic. Each profile captures three backward
calls, but counters are capture aggregates, not per-region windows. For DIE0:

| Counter | PR13 | Unroll 2 | Compute-KV 512 + unroll 2 |
| --- | ---: | ---: | ---: |
| OCI TCS destination Any2IMEM descriptors | 3 | 4623 | 3 |
| OCI TCS destination Any2IMEM bytes | 79872 | 8098139136 | 79872 |
| TC IMEM writes | 624 | 63266712 | 624 |
| TC IMEM DMA active | 624 | 63397694 | 624 |

These are instruction-memory counters, not KV data-transfer estimates. The
massive increase together with the repeating boundary gaps and trace-disabled
regression strongly supports repeated instruction loading after unrolling.
Do not use the all-capture MXU idle counts to infer kernel utilization: capture
startup/idle durations differ substantially between profiles.

## Forward loop-carried state probe

Commit `41877a1` adds default-off `fwd_loop_carry`: carry softmax m/l and output
accumulators through the inner loop, committing them to scratch only at the
memory-tile boundary. Operation order and intermediate precision are retained.
The hypothesis is reduced explicit VMEM traffic and more compiler scheduling
freedom, not an assumed speedup. The final coarse baseline body still contains
24320 vector loads, 8456 unmasked stores and 12288 MXU multiply instructions
across both mask branches; these are static counts, not dynamic costs.

The forward screen is `exp-406vxkirrx`, artifact `art-7vslcgvgd9`, pinned to
`41877a1`, with the same JAX/libtpu versions, production shape and seed. It checks
output, logsumexp and dQ/dK/dV, and screens carry, compact scratch, Q/compute-KV
sizes and scheduler combinations. CPU-interpreted checks pass, but the flag
remains experimental and off by default until compiled TPU evidence is reviewed.

**Not promoted:** all nine variants completed. Plain carry is 21.719 ms versus
its live reference 20.826 ms; compact carry is 22.479 versus 21.053 ms; Q=2048
carry variants regress to approximately 42 ms. The fastest carry variant,
scheduler+carry, is 20.709 versus 20.889 ms, not a significant target-sized gain.
Scheduler without carry is bitwise and 20.806 versus 21.079 ms, still only a
small screening difference. The carry and compact-carry outputs have relative
L2 difference `2.07090e-5`, dQ `3.22644e-5`, and dK `4.89105e-5`; logsumexp/dV
remain bitwise. All values are finite. No accuracy/performance threshold is
relaxed to accept these results.

The first coarse forward module changes from 20.073 ms to 20.842 ms with carry;
its KV-loop total changes from 18.687 to 19.460 ms. Unlike backward unrolling,
this regression is inside the loop. Candidate final bodies have substantially
fewer explicit loads/stores (e.g. 2504 vector loads and 776 unmasked/512 masked
stores, versus baseline 24320 and 8456/8448), with unchanged MXU multiply count.
Reducing explicit scratch traffic did not improve the critical path. These
compiler-body counts still do not include outer pipelines or prove why the
machine-level schedule is slower.

Forward evidence: `an-q3iucqlhsu/details.json`, region/final-LLO
`an-mq1zkr3lqg/audit.json`, operator `an-f1siy0wbb4`, LLO inventory
`an-btynsn829z`.

## Single segment-mask body hypothesis

Commit `86a3fd4` adds default-off `bwd_single_segment_mask_body`, restricted to
segment-only masks (no arbitrary mask Ref/function). Full and partial tiles use
one masked body; full tiles do redundant segment comparisons but do not change
the allowed attention entries. This removes the full/partial body duplication
before unrolling, testing the code-footprint hypothesis above. It is not a claim
that the MXU/vector pipeline is already optimal.

The production-shape screen is `exp-pij8860e8i`, artifact `art-vlh8hwttfv`, with
layouts-only control and single-body unroll factors 1/2/4/8, raw timing samples,
full-shape precision checks and coarse traces. The source remains on our branch;
no production defaults changed. The CPU suite passes 49 tests at `86a3fd4`.

The screen completes with four measured configurations and one explicitly
recorded compile failure. Layouts-only is 49.172 ms; the single-body version is
50.516 ms, unroll-2 is 49.999 ms and unroll-4 is 53.145 ms. All measured dQ/dK/dV
arrays are bitwise equal to PR13. Unroll-8 exceeds compiler VMEM capacity:
69.11M requested versus 63.94M available, a 5.17M excess. This is a candidate
compilation failure, not a failed Falcon reservation or a measured runtime.

Single-body unroll-2 recovers the earlier approximately 65.5 ms regression to
50.0 ms. That is recovery relative to a regressed candidate, **not** a 20–30%
improvement over PR13. No production setting is promoted from this screen.

The final audit closes the hypothesis loop. Single-body unroll-2 reduces DIE0
Any2IMEM descriptors back to 3 and bytes back to 79872 per three-call capture;
its uncovered internal scope time is 0.195 ms, versus 16.266 ms for the old
two-body unroll-2. Its device module is 48.718 ms, with 47.280 ms in the masked
loop. This strongly corroborates instruction-loading/code-footprint as the
cause of the old unroll-2 regression. The extra masking still makes the loop
slightly slower than layouts-only (46.706 ms), so eliminating this regression
does not improve the baseline critical path.

Single-body unroll-4 again creates instruction traffic: 18444 descriptors and
16008055296 bytes per capture, with a 52.201 ms module, 48.999 ms loop total and
1.952 ms uncovered scope time. More instruction bytes do not translate directly
into proportionally more exposed latency; location and overlap matter. The
next pipeline design must control both code footprint and VMEM live ranges.

Evidence: numerical details `an-092z7rzwx4`, region/counters/final LLO
`an-to8chbbdf2`, operator `an-nxiwba0v4z`, LLO inventory `an-dds4p1n550`.
This run records wall-clock compilation windows, allowing final dumps to be
mapped to variants directly rather than by compile order alone.

## Two-stage backward KV pipeline probe

Commit `6713969` adds default-off `bwd_staged_kv_pipeline`, restricted to the
BF16 ViT exact-layout configuration and a single segment-mask body. The
prologue computes P/dS for tile 0; each iteration prepares tile i before
consuming tile i-1 in dV/dQ/dK; the epilogue consumes the last tile. P remains
FP32 in the softmax derivative. The loop carries BF16 P/dS only at the cast
boundaries already used by the reference gradient dots, retaining tile order
for the accumulations. The hypothesis is extra cross-tile scheduling freedom,
not that this source ordering guarantees hardware overlap.

`exp-tgu93neqym`, artifact `art-8awxxa6vz9`, measures production shape with
20 samples per candidate and live PR13 checks. All values below are host
medians, with unchanged JAX/libtpu, BF16 inputs and one benchmark device.

| Variant | Backward ms | Live PR13 ms | Numerical result |
| --- | ---: | ---: | --- |
| Exact layouts, no pipeline | 49.584 | 50.012 | Bitwise |
| Pipeline compute-KV 512 | 59.350 | 50.180 | Non-bitwise, finite |
| Pipeline compute-KV 256 | 61.190 | 50.094 | Non-bitwise, finite |
| Pipeline Q=2048 / compute-KV 512 | 61.818 | 49.996 | Non-bitwise, finite |
| Pipeline compute-KV 512 + scheduler | 64.305 | 50.007 | Non-bitwise, finite |
| Pipeline compute-KV 1024 | Compile failure | 50.021 | Not executed |

The 1024 case needs 70.57M VMEM versus 63.94M available; register-allocation
spill slots account for 40.95M. This diagnoses the failed configuration, not
the precise spill cost of the configurations that compile. For compute-KV 512,
dK is bitwise; dQ relative L2 is `2.14896e-5` and dV `3.22779e-6`. Changing the
compute tile also changes reduction grouping; the comparison against PR13 does
not isolate numerical changes caused solely by pipelining. No candidate is
promoted, and the precision requirement is not relaxed.

The first device module grows from 48.152 ms (layouts only) to 58.031 ms
(compute-KV 512 pipeline); KV-loop time grows from 46.708 to 56.583 ms.
Uncovered internal scope time stays about 0.199 ms. All three profiled variants
have DIE0 TCS Any2IMEM descriptors=3, bytes=79872, and IMEM writes=624 for the
three-call capture. Thus this regression is inside the loop, without the large
instruction-loading traffic from the earlier unrolling regression. Simply
carrying an additional tile did not improve the critical path.

The CPU suite passed 61 tests before this submission. Evidence: details
`an-uqnqhim26g`, device regions/counters/final LLO `an-otlg1kfq4m`, operator
`an-a916zaobsu`, LLO inventory `an-dq0udj0fpa`. The generic operator report
recognizes five successful metric rows; the failed compilation is preserved
separately in `details.json`, rather than reported as a measured latency.

## Forward unrolling and staged pipeline controls

Commit `9f032bb` adds `fwd_kv_unroll` with its original default `True` and a
default-off fixed-logit-shift `fwd_staged_kv_pipeline`. The latter prepares
QK/exp for tile i before consuming tile i-1 in PV and the accumulators.
Probabilities stay FP32 in PV; this is not a BF16-probability or FP8 shortcut.
Forward outputs, logsumexp and all three gradients are checked, not just the
backward function in isolation. The CPU suite passes 67 tests.

Production-shape experiment `exp-77nlvju2ci`, artifact `art-jhiex75o69`, has
20 timing samples per candidate. All eight compute-KV=256 configurations
(including PR13) are bitwise equal for output, logsumexp, dQ, dK and dV.

| Variant | Forward ms | Live PR13 ms |
| --- | ---: | ---: |
| PR13, fully unrolled | 21.070 | 20.863 |
| Rolled control | 43.925 | 21.447 |
| Unroll 2 | 32.570 | 20.782 |
| Unroll 4 | 26.459 | 20.828 |
| Unroll 8 | 23.707 | 21.015 |
| Staged pipeline, rolled | 44.393 | 20.866 |
| Staged pipeline, unroll 2 | 33.003 | 20.954 |
| Staged pipeline, unroll 4 | 28.234 | 20.934 |
| Staged pipeline, compute-KV 512 / unroll 2 | 30.632 | 21.144 |

The final row is finite but non-bitwise: output relative L2 `2.72818e-5`,
logsumexp `3.38810e-8`, dQ `1.40507e-4`, dK `1.44960e-4`, dV `1.27368e-4`.
There is no performance gain to justify further promotion work on this
configuration. None of this screen changes the default implementation.

The device trace locates the forward regression inside the KV loop:

| Profile | First device module ms | KV loops ms | Uncovered internal ms |
| --- | ---: | ---: | ---: |
| PR13 | 20.072 | 18.687 | 0.428 |
| Rolled | 42.454 | 41.026 | 0.447 |
| Pipeline + unroll 2 | 31.604 | 30.210 | 0.425 |

All three captures have DIE0 TCS Any2IMEM descriptors=3 and bytes=79872.
Reducing code expansion does not help this forward body. The controlled
unroll sequence demonstrates an important cross-iteration scheduling effect,
but these coarse scopes do not directly measure which operations overlap.
At matched unroll factors the explicit two-stage version is no faster than
the ordinary loop. Do not label the improvement over the rolled control a
gain over PR13.

Evidence: details `an-tysswrc1uk`, region/counters/final LLO `an-o8d6osf38w`,
operator `an-6dzppjbuq4`, LLO inventory `an-y1sy0xg6k2`. The generic plugin's
trace count is zero because it does not discover this nested XProf layout;
the region analyzer reads all three captures and nine device calls directly.

## Raw device metadata and remaining interpretation limits

The trace-JSON clock search `an-d4neowuln6` found no validated clock or
counter sampling window. A separate read-only decoder of the original
XPlane files, `an-x4q19ck8ry/metadata.json`, uses the
[OpenXLA XPlane schema](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto).
For `/device:TPU:0` it exposes `peak_teraflops_per_second=1028.75`,
`peak_hbm_bw_gigabytes_per_second=3686.1556817920005`,
`has_megacore=0`, and `has_merged_vmem=1`. It does not expose a TensorCore clock
statistic or a validated hardware-counter duration. Peak-capability metadata
must not be treated as a measurement of operating frequency or kernel
utilization. The large idle/startup component of all-capture counters remains
a reason not to divide them by one kernel duration.

This pair of pipeline probes has not achieved the 20–30% target. The next
useful experiment must reduce live ranges or layout work while retaining the
effective unrolling schedule, rather than assuming that a larger carried
tile will create useful overlap. Arithmetic changes remain subject to
separate numerical review; CPU equality alone cannot certify TPU behavior.

## Revisit dQ orientation with complete scheduling controls

Commit `f6d44aa` restores default-off `bwd_dq_transposed_output` for a diagnostic
screen that records numerical errors and timings even when the bitwise check
does not pass. This is not promotion of the earlier rejected candidate, nor a
relaxation of accuracy requirements. The original attempt stopped before
timing and could not establish whether this direction was worth investigating.
The source/CPU suite now has 77 passing kernel tests, including head width 72.

`exp-9tk3m3tvck`, artifact `art-9vgcbvvihf`, uses the same production shape,
runtime and 20-sample method, with unchanged forward residuals. The critical
matched controls are:

| Variant | Backward ms | Live PR13 ms | Precision versus PR13 |
| --- | ---: | ---: | --- |
| Layouts only, dQ first | 49.322 | 49.926 | Bitwise |
| Layouts only, dK first | 49.735 | 49.973 | Bitwise |
| Transposed dQ, dQ first | 58.044 | 50.246 | dQ/dV non-bitwise |
| Transposed dQ, dK first | 44.999 | 49.817 | Only dQ non-bitwise |

The last candidate is a screening speedup of 1.1071x, not the 20–30% target.
It differs in 303 of 75,497,472 dQ elements; dQ relative L2 is `6.34282e-6`
and maximum absolute difference `1.22070e-4`. dK/dV are bitwise and all values
finite. Candidate precision remains under review. Plain transposed dQ also
changes 5721 dV values (relative L2 `1.49402e-5`) and is slower.

Other controls: moving dV last without transposition is 51.291 ms; moving dP
early without transposition is 57.828 ms. Transposition with dV last or dP
early measures 48.174 or 47.896 ms; scheduler and single-mask-body variants
remain about 57.8 ms. Thus transposition alone is insufficient; the order of
consuming dS is consequential for the compiled schedule.

The coarse device evidence corroborates an internal-loop improvement:

| Profile | Device module ms | KV loops ms | Uncovered internal ms |
| --- | ---: | ---: | ---: |
| Layouts only | 48.150 | 46.706 | 0.197 |
| Transposed dQ, dQ first | 56.638 | 55.186 | 0.203 |
| Transposed dQ, dK first | 43.917 | 42.475 | 0.198 |

All captures have only 3 TCS Any2IMEM descriptors / 79872 bytes. Final-body
static `llo.vxpose` counts fall from 6912 to 3968 (counting result readout
separately); `llo.vmatmul.mubr` counts fall from 10240 to 8832. The two
transposed orderings have the same instruction counts despite their large
runtime difference, so operation-count savings alone do not explain success.

Hardware issued-instruction counters strengthen the compiler-body evidence.
Across three calls, the summed DIE0 BF16 VREG matmul count on MXU0+MXU1 drops
from 62,914,560 to 54,263,808 (13.75% fewer) for either transposed ordering.
Per-XLU transpose counts drop from 18,788,352 to 8,368,128. These are dynamic
instruction counts, not cycle utilization or time fractions. The dK-first
ordering distributes matmul issues 28,311,552 / 25,952,256 across MXU0/1;
dQ-first distributes them evenly at 27,131,904 each, yet is slower.

Evidence: detailed metrics `an-nupwjcci8e`, device/counter/final-LLO audit
`an-m9z2g48d7n`, operator `an-tj5xy698y4`, LLO inventory `an-0aq7is1i9u`.

## Independent numerical reference

Commit `7945d86` adds an untimed FP32 attention/gradient oracle for selected
full-length heads. It uses all keys per query chunk, FP32 softmax and gradient
arithmetic, and highest-precision XLA matmuls. It does not reuse Splash's
forward residuals, BF16 probability casts, or custom backward. Natural-log
LSE is compared after converting Splash's base-2 residual statistic.

Seven tests validate the oracle against independent dense FP64 autodiff at
head width 72, two query block sizes, and input scales 0.25/1/2, including
segment 0. Backward and forward CLI smoke tests also complete at sequence 512.
Together with the kernel tests, 84 tests pass. These tests validate the
reference implementation, not production-shape TPU numerical acceptance.
`--oracle-heads` records both baseline and candidate error against this oracle;
it deliberately does not change `needs_accuracy_review` to acceptance.

### TPU oracle failure and stage-isolation diagnostic

`exp-9giz2uwx42` (`5077115`, 30 samples) repeats transposed-dQ/dK-first at
45.264 ms versus live PR13 50.048 ms (1.1057x). The dQ mismatch statistics
repeat exactly; dK/dV remain bitwise. The other unroll candidates in this
screen retain **dQ-first**, so their negative results do not test unrolling
on the fast dK-first schedule. Single-body unroll-2/4 gives 51.249/52.718 ms;
ordinary unroll-2/4 gives 66.875/58.949 ms. Unroll-2 has 65.525 ms device
duration but only 48.691 ms in KV loops; the single-body version recovers
50.156 ms device duration with 48.707 ms loops.

The independent FP32 reference is invalid in this run: its dQ/dK are
nonfinite on all four sampled heads. Baseline/candidate gradients themselves
are finite. Do not use the NaN error fields as precision acceptance.
Evidence: details `an-0ok99or7wa`, regions `an-28mp6gz5bg`, operator
`an-pjn1f3dxnn`, LLO `an-yayxcwarct`.

`exp-0yldvlqyf6` (`7006fa7`, artifact `art-dc9q50pz61`) isolates the reference
on one full-length head with identical seed-27 input generation:

| Reference | Nonfinite O | Nonfinite dQ | Nonfinite dK | LSE / dV |
| --- | ---: | ---: | ---: | --- |
| Original, negative-infinity mask | 1,179,648 | 1,179,648 | 2,359,296 | Finite |
| Finite mask (-1e30) | 0 | 0 | 0 | Finite |
| Negative-infinity mask + stage barriers | 0 | 0 | 0 | Finite |
| Finite mask + stage barriers | 0 | 0 | 0 | Finite |

The original O/dQ failures begin at query row 16384. Barriers preserve the
equations, dtypes, and infinite mask; the result implicates compiled fusion,
but does not identify the exact failing compiler transformation. Do not
claim that finite values alone establish numerical correctness. A strict
finite check now rejects invalid references before benchmarking. Commit
`28a157b` enables barriers by default and adds independent CPU/NumPy FP64
reference validation on the exact TPU-generated full head. That full-head
cross-check completed in `exp-r3ryu5qvp6` (results below). Local validation is
127 passing tests, including the twelve dK/dV orientation cases; the 26
oracle tests also validate NumPy against dense FP64 autodiff.

Oracle-health analyses: details `an-0y7ukdkw0a`, operator `an-gxgj72824k`.

## Forward PV orientation: instruction-loading regression

`exp-spb936izpj` (`60b8750`, artifact `art-m57a7l9w2v`) transposes the PV
contraction without casting FP32 probabilities to BF16. Output scratch
orientation is independently controlled. Defaults remain unchanged.

| Variant | Forward ms | Live PR13 ms | Precision versus PR13 |
| --- | ---: | ---: | --- |
| PR13 control | 21.036 | 21.386 | Bitwise |
| Sequence-minor output scratch | 91.288 | 20.968 | Bitwise |
| PV transposed | 91.989 | 20.695 | Bitwise |
| Both | 118.143 | 21.050 | Bitwise |
| Both, Q512 | 91.974 | 21.299 | Bitwise |
| Both, Q2048 | 83.430 | 20.834 | Bitwise |
| Both, compute KV512 | 33.199 | 20.621 | Needs review |
| Both, experimental scheduler | 117.676 | 20.919 | Bitwise |

The bitwise checks include output, LSE, dQ, dK, and dV on the full shape.
This is a negative screen, not a retained optimization.

| Capture | Device ms | KV-loop ms | Internal uncovered ms | TCS IMEM bytes, 3 calls |
| --- | ---: | ---: | ---: | ---: |
| PR13 | 20.071 | 18.687 | 0.429 | 79,872 |
| PV transposed | 90.649 | 50.729 | 38.915 | 26,830,094,848 |
| Both | 116.773 | 70.034 | 45.617 | 36,116,313,600 |

TCS Any2IMEM descriptor counts are 3 / 36,335 / 36,882. This corroborates an
instruction-loading problem in addition to slower inner loops. No positive-
duration TC Overlay events were decoded, so the byte counters and scope
gaps are evidence, not a precise overlay-time decomposition.

Final forward-body `llo.vmatmul.mubr` counts fall from 12,288 (initial
reference compile) to 5,376 for the transposed variants, but `llo.vxpose`
counts rise from 4,224 to 12,416 / 28,928. Sequence-minor scratch alone raises
transposes to 20,736 without changing MXU matmul counts. These static counts
show why reduced dot issue work does not imply a faster complete kernel.
Partial-unroll controls are needed to separate code expansion from the
underlying PV schedule before closing this direction.

Evidence: details `an-6924vtlmtq`, regions/counters/final-LLO `an-915yfg2e51`,
operator `an-6s2a2bppgp`, LLO `an-8t7snn4y9r`.

### Removing the forward instruction-loading confound

`exp-1bg5vj2c1m` (`a07f44a`, artifact `art-ylgxjwddyq`) controls partial
unrolling while retaining FP32 probabilities. PV transpose with unroll 2/4/8
measures 40.424 / 35.100 / 38.624 ms. Adding sequence-minor output scratch
measures 56.757 / 57.233 / 65.619 ms. Q2048 with unroll-4 measures 53.867 ms;
compute-KV512 with unroll-4 measures 32.374 ms. Live PR13 stays 20.8–21.3 ms.
All cases except compute-KV512 are bitwise for output, LSE and all gradients.

Unroll-4 restores normal TCS instruction loading: every captured variant has
3 Any2IMEM descriptors and 79,872 bytes. Its PV-transpose device module is
33.800 ms, with 32.398 ms in KV loops and 0.431 ms of internal scope gaps.
With sequence-minor scratch, these are 55.839 / 54.330 / 0.424 ms. Baseline is
20.072 / 18.685 / 0.430 ms. Thus instruction loading explains part of the
fully-unrolled regression, but eliminating it does **not** produce a faster
inner loop. This PV orientation is not a retained optimization.

Evidence: details `an-mrfjzwjqjs`, regions `an-80gswv1rj1`, operator
`an-56hkdzfkjt`, LLO `an-bzkbrpsn27`.

## Whole-head CPU validation and extended backward controls

`exp-r3ryu5qvp6` (`28a157b`, artifact `art-jklpzc9yb8`) validates the repaired
TPU oracle against independent NumPy/CPU FP64, using every element of head 0
at sequence 32768. CPU computation evaluates each equality-masked segment
separately and does not reuse TPU residuals or gradients. For the
negative-infinity-mask + stage-barrier oracle:

| Array | Relative L2 vs CPU FP64 | Max absolute error |
| --- | ---: | ---: |
| Output | 1.38677e-7 | 1.52202e-7 |
| Natural-log LSE | 4.46738e-8 | 1.30225e-6 |
| dQ | 1.96810e-7 | 2.04853e-7 |
| dK | 1.91563e-7 | 2.03390e-7 |
| dV | 1.39792e-7 | 1.43716e-7 |

These values validate this full-head reference case, not all possible inputs.
The original unbarriered oracle still reproduces NaNs. Evidence:
`an-tkpp5u1yul/details.json`.

The extended backward screen explicitly uses the fast dK-first schedule:

| Variant | Backward ms | Live PR13 ms |
| --- | ---: | ---: |
| Layouts only | 49.549 | 49.765 |
| Transposed dQ, dK first | 45.278 | 50.102 |
| Single mask body | 46.689 | 50.231 |
| Single body, unroll 2 | 46.682 | 50.316 |
| Single body, unroll 4 | 53.075 | 49.894 |
| Q2048 | 47.299 | 50.028 |
| Compute-KV512 | 48.004 | 50.120 |
| Additionally transpose dK | 48.478 | 50.168 |
| Additionally transpose dV | 51.429 | 50.019 |
| Transpose all gradient outputs | 50.878 | 49.902 |
| All transposed, single body, unroll 2 | 53.612 | 50.060 |

Transposing all outputs reduces final-body `vmatmul.mubr` count to 6016
from the fast candidate's 8832 (baseline 10240), but raises `vxpose` to 4480
from 3968. Its device module / KV loops are 49.578 / 48.126 ms versus the
fast candidate's 43.920 / 42.476 ms. IMEM bytes remain 79,872 for all captures.
This is another inner-loop scheduling/layout regression despite fewer dot
issues, not an outer instruction-loading stall.

For the fast candidate, FP32-oracle dQ relative-L2 errors across heads
0/15/16/31 are 0.002888517 / 0.002921635 / 0.002910684 / 0.003081070.
PR13 errors are 0.002888520 / 0.002921634 / 0.002910684 / 0.003081072.
Each head's max-absolute error is unchanged; dK/dV are bitwise with PR13.

Evidence: details `an-42z5r75nvz`, regions/final-LLO `an-c5m6uq5mdc`, operator
`an-tnyy5kohl3`, LLO `an-gptkw3umh8`.

### Independent seed and no-region-scope reproduction

`exp-i3pvqp34m4` (`28a157b`, artifact `art-e3o3uukric`) uses seed 28 and
`region_trace_mode=none`, 30 timing samples and the same four oracle heads.
The best candidate is 45.313 ms versus live PR13 49.738 ms. Layouts alone
measure 49.529 ms. Additional dK / dV / both transpositions measure
48.412 / 51.538 / 50.725 ms and are not improvements over the best candidate.

All candidate arrays are finite. Best-candidate dQ differs in 302 elements,
relative L2 `7.70493e-6`, max absolute `2.44141e-4` versus PR13. dK/dV are
bitwise. Against the independent FP32 oracle, dQ errors on heads 0/15/16/31
are 0.003188923 / 0.003116066 / 0.002905411 / 0.002933370 versus PR13
0.003188935 / 0.003116070 / 0.002905410 / 0.002933371. The worst ratio of
candidate/reference L2 error is 1.000000561; max-absolute error is unchanged
on every sampled head. These are diagnostic evidence, not an implicit
relaxation of a user-defined accuracy threshold or training validation.

Evidence: details `an-gcmcxfizuf`, operator `an-7hg8gz2au1`, LLO
`an-uqz3roun7j`.

## Whole-backward probability-layout probe

Commit `d90a633` adds default-off `bwd_qmajor_probabilities`: construct P/dS
as `[Q, KV]` throughout backward, rather than independently transposing each
gradient dot on the old `[KV, Q]` intermediates. dK/dV directly consume this
orientation and write existing sequence-minor scratch. dQ orientation and
stage order remain independent controls. All BF16 dot boundaries and FP32
probability/softmax arithmetic are preserved; forward residuals are unchanged.
Eight CPU tests compare all gradients bitwise with asymmetric Q256/KV128
tiles at head width 72. The combined regression suite passes **147 tests**.

`exp-hio7co26dh` (artifact `art-m6b2e4474k`, 30 samples, four oracle heads)
has completed. It does not beat the earlier fast candidate:

| P/dS layout and schedule | Backward ms | Live PR13 ms | Numerics vs PR13 |
| --- | ---: | ---: | --- |
| Old layout, transposed dQ, dK first | 45.071 | 49.939 | Only dQ differs |
| Q-major, normal dQ, dQ first | 48.012 | 49.759 | Bitwise all gradients |
| Q-major, normal dQ, dK first | 52.830 | 49.736 | Bitwise all gradients |
| Q-major, transposed dQ, dQ first | 50.217 | 49.820 | Only dQ differs |
| Q-major, transposed dQ, dK first | 51.622 | 50.140 | Only dQ differs |
| Previous row, compute-KV512 | 61.535 | 50.047 | Only dQ differs |
| Previous base, Q2048 | 53.365 | 50.033 | All gradients differ |

Moving dP before QK in the transposed-dQ/dK-first configuration does not
compile: 66.75M VMEM requested versus 63.94M available, including 37.12M
register-allocator spill slots. This is a rejected configuration, not a
numerical or runtime failure of the other variants. All measured arrays are
finite. At unchanged tiles, dQ-transposed variants reproduce the same 303
dQ differences; dK/dV stay bitwise.

| Capture | Device ms | KV loops ms | Internal uncovered ms | Static matmul / transpose ops |
| --- | ---: | ---: | ---: | ---: |
| Layouts baseline | 48.146 | 46.706 | 0.197 | 10240 / 6912 |
| Q-major, normal dQ | 46.839 | 45.392 | 0.199 | 7424 / 5248 |
| Q-major, transposed dQ, dK first | 50.246 | 48.792 | 0.203 | 6016 / 4224 |

All three captures have 79,872 TCS IMEM bytes. Static vector loads/stores are
3664/2880 for each final body. Fewer MXU/transpose instructions do not explain
the runtime ordering; the regression is inside the KV loop. This still does
not establish inner-loop MXU/vector overlap as optimal.

Evidence: details `an-a6iipk3bno`, regions/final-LLO `an-9qlt4trpqz`, operator
`an-xmsp4oyowg`, LLO `an-purr19lvhz`.

## Native KV-major forward probabilities

Commit `0610663`, experiment `exp-re816zfnid` (artifact `art-bhjrldqdnr`),
implements default-off `fwd_kvmajor_probabilities`: form P as `[KV, Q]`
directly from QK and feed `V @ P` with sequence-minor output/state scratch.
P remains FP32; BF16 inputs, fixed logit shift and segment semantics remain
unchanged. CPU interpret tests cover both sum expressions and partial
unrolling. The 159-test kernel/oracle suite passed before this experiment.

| Forward variant | Median ms | Live PR13 ms |
| --- | ---: | ---: |
| Native KV-major, Q1024 | 27.037 | 21.242 |
| Express sum on transposed Q-major P | 96.941 | 21.021 |
| Native, unroll 4 | 30.749 | 21.069 |
| Q-major sum, unroll 4 | 57.926 | 20.902 |
| Native, unroll 8 | 28.619 | 21.025 |
| Native, compute-KV512 | 21.797 | 21.150 |
| Native, Q512 | 38.822 | 21.108 |
| Native, Q2048 | 21.497 | 21.065 |
| Native, experimental scheduler | 26.985 | 20.882 |

No variant beats PR13. Device trace separates two kinds of regressions:

| Capture | Module ms | KV loops ms | Internal uncovered ms | IMEM bytes, three calls |
| --- | ---: | ---: | ---: | ---: |
| PR13 | 20.073 | 18.687 | 0.429 | 79,872 |
| Native KV-major | 25.842 | 23.787 | 0.435 | 79,872 |
| Q-major sum | 95.724 | 50.846 | 43.216 | 27,663,487,488 |
| Native, unroll 4 | 29.267 | 27.218 | 0.422 | 79,872 |

The Q-major sum expression causes instruction-loading/code-expansion
regression (36,339 IMEM descriptors versus 3). Native KV-major's regression
is instead primarily inside the KV loop, without that instruction-loading
confound. Its final body has 5,376 `vmatmul.mubr` and 6,656 `vxpose` ops;
the reference body has 12,288 and 4,224 respectively. Counts include both
mask branches and unrolling and are not time/utilization fractions. The
Q-major sum increases `vxpose` to 23,040 with unchanged matmul count.

All measured arrays are finite but not bitwise. Native Q1024/Q2048 have
the same precision statistics against PR13: output 6,056 mismatches out
of 75,497,472, relative L2 `2.23422e-5`; stored base-2 LSE 39,071 mismatches
out of 1,048,576, max absolute `1.90735e-6`. dQ/dK/dV relative L2 differences
are `8.23397e-5` / `9.13902e-5` / `7.33053e-5`. Against the independent
FP32 oracle on heads 0/15/16/31, the worst candidate/reference relative-L2
error ratios are 1.000002964 (output), 1.001046173 (LSE), 0.999998000 (dQ),
1.000004494 (dK), and 1.000010185 (dV); per-head max-absolute errors are
unchanged. These are diagnostic results, not automatic accuracy approval.

Evidence: details `an-sbrdkniwh3`, regions/final-LLO `an-ez9s93lf2r`,
operator `an-f6049yyh1j`, LLO `an-7actj3l6w8`.

## Next bounded experiment: fused normalization reduction

The next default-off probe appends eight BF16 constant-one rows to V,
computing `[V; 1] @ P` to produce output and softmax denominator together.
The extra rows match the compact statistics' sublane layout. P remains
FP32; this avoids a separate vector traversal/reduction of P but changes
reduction association, so CPU FP64 tests and independent TPU FP32 checks
are required. Larger Q2048/4096/8192 tiles independently test amortization.
No speedup or TPU numerical acceptance is claimed before those runs finish.
The combined CPU kernel/oracle suite with four new fused-normalizer tests
passes **163 tests** (118.07 seconds).

### Completing gradient-consumer ordering controls

An additional default-off `bwd_dv_between_dq_dk` control allows dV between
dK/dQ, instead of only before or after both. This completes all six consumer
orders without altering dot operands, accumulation dtypes, or residuals.
The staged pipeline rejects this unsupported combination; contradictory
middle/last options fail configuration validation. Twelve new CPU cases
cover both dQ/dK orders, early/late dP, and original/transposed/Q-major
layouts. With the invalid-option test, **176 tests pass** (124.34 seconds).

The next backward screen uses the existing fast transposed-dQ/dK-first
candidate as its starting point and varies dV position and dP preparation.
Forward tiling controls also include reduced KV memory blocks and partial
unrolling, so a larger Q tile can be tested without simultaneously expanding
the fully-unrolled body. No additional TPU gain is claimed yet.

## Fused normalizer: default contraction fails TPU accuracy

`exp-b511scebng` (`fdb6cef`, artifact `art-imuncig1o3`) completed. Native
Q2048/4096/8192 take 21.465 / 32.507 / 27.568 ms. Fused normalization takes
26.370 ms at Q1024, 20.816 ms at Q2048, 31.251 ms at Q4096, and 26.858 ms
at Q8192. Q4096 with unroll 8 reaches 19.342 ms versus live PR13 21.281 ms,
but **this is rejected, not a usable optimization**, because the denominator
does not preserve FP32-reduction accuracy on TPU.

For fused Q2048, output/dQ/dK/dV relative L2 differences from PR13 are
0.0010281 / 0.0018805 / 0.0017851 / 0.0015005. Stored base-2 LSE differs in
1,027,851 of 1,048,576 elements, max absolute 0.00309324. Against independent
FP32 reference heads, the worst natural-LSE L2-error ratio is 53.4589x and
max-absolute-error ratio 855.75x. Gradient L2-error ratios are 1.03724x (dQ),
1.05213x (dK), 1.04360x (dV); worst max-absolute ratios reach 1.03867x,
1.21562x, 1.33895x. The 19.342 ms variant has essentially the same regression.
Passing CPU interpret tests did not catch this TPU contraction behavior.

Local JAX 0.11 Mosaic lowering leaves the contraction precision attribute
unset for DEFAULT, while HIGHEST requests `#tpu.contract_precision<fp32>`.
An FP32 P array alone does not enforce the latter. The corrected probe now
explicitly requests HIGHEST for the fused output/denominator contraction;
the original separate vector reduction and non-fused PV are unchanged.
Its CPU suite passes 176 tests (128.09 seconds), plus four focused tests
assert that the denominator dot explicitly requests HIGHEST. TPU accuracy
and speed still require a new source-pinned run.

Coarse trace confirms an independent code-footprint regression at Q4096:

| Variant | Device ms | KV loops ms | Internal uncovered ms | IMEM bytes, three calls |
| --- | ---: | ---: | ---: | ---: |
| PR13 | 20.074 | 18.687 | 0.428 | 79,872 |
| Native Q2048 | 20.469 | 18.706 | 0.209 | 14,006,784 |
| Native Q4096 | 31.308 | 17.926 | 11.809 | 9,976,869,888 |
| Rejected fused Q2048 | 19.615 | 17.851 | 0.210 | 13,337,088 |
| Rejected fused Q4096 | 29.823 | 16.510 | 11.743 | 9,702,053,376 |

Q4096 has 9,714 instruction-load descriptors, versus 12 at Q2048 and 3 for
PR13. Native/fused versions have identical final-body matmul/transpose
counts at matched tiles; fusion removed vector reduction work, not dot
count. The corrected next experiment controls contraction accuracy and code
footprint separately. These captures do not measure fine-grained overlap.

Evidence: details `an-olchn8c8oh`, regions/final-LLO `an-akbm0kgcz0`,
operator `an-mup5nz5vi2`, LLO `an-wy9n9h3iux`.

## Complete backward consumer-order results

`exp-6533l1trog` (`f9dc55c`, artifact `art-7e2xye75w5`) does not improve
on the existing 45.201 ms transposed-dQ/dK-first candidate (live PR13
50.038 ms). Moving dV last/middle gives 46.603/46.849 ms; dQ-first with
dV between gives 48.143 ms. Early dP with dV first/last/middle gives
48.069/47.480/47.739 ms. Transposing all outputs with dV last/middle gives
50.570/50.830 ms. All arrays are finite. The best candidate reproduces 303
dQ differences, dK/dV bitwise; its worst four-head FP32-reference L2-error
ratio is 1.000000319 and max-absolute errors are unchanged.

The first seven reordered transposed-dQ bodies all have the same 8,832
matmul, 3,968 transpose, 3,664 load, and 2,880 store instructions. Yet device
KV-loop time is 42.497 ms with dV first, 43.786 with dV last, 44.035 with
dV between, and 45.403 with early dP. Each capture loads only 79,872 IMEM
bytes, with approximately 0.199 ms internal uncovered time. This confirms
the ordering regression inside the loop, not instruction-load expansion;
it does not establish that every possible overlap schedule is optimal.

Evidence: details `an-jn386pfpkl`, regions/final-LLO `an-zfc38ffpjv`,
operator `an-m9cx4wn4e0`, LLO `an-0sxzjrjhnr`.

## Forward footprint controls with the separate FP32 reduction

`exp-jtoxrrfgxe` (`f9dc55c`, artifact `art-k3c8v39kkc`) obtains the first
small forward improvement without the rejected fused denominator:

| Native KV-major variant | Forward ms | Live PR13 ms |
| --- | ---: | ---: |
| Q4096, unroll 8 | 20.352 | 21.108 |
| Q4096, memory-KV4096 | 19.953 | 20.797 |
| Q8192, memory-KV2048 | 42.298 | 20.989 |
| Q2048, compute-KV512 | 19.881 | 21.049 |

For Q4096/memory-KV4096, device module / KV loop / internal uncovered time
are 18.714 / 16.971 / 0.219 ms, with 13,655,040 IMEM bytes. For
Q2048/compute-KV512 these are 18.841 / 17.082 / 0.211 ms and 12,862,464 bytes.
PR13 is 20.069 / 18.679 / 0.431 ms and 79,872 bytes. Reducing the fully
expanded body removes the earlier approximately 12 ms gaps at Q4096.

All arrays remain finite. The compute-KV512 candidate differs from PR13 in
6,542 output elements (relative L2 `2.54598e-5`), with dQ/dK/dV relative-L2
differences `1.39964e-4` / `1.43627e-4` / `1.27219e-4`. Against independent
FP32 heads 0/15/16/31, worst L2-error ratios are 1.000002415 for output,
0.990491405 for LSE, 0.999995360 for dQ, 1.000013208 for dK, and 1.000008779
for dV. All per-head maximum absolute errors are unchanged. These are
promising numerical diagnostics, not training acceptance or a no-scope
reproduction. Forward and backward gains have not yet been jointly measured.

The two fused variants in this run are still the old DEFAULT-precision
implementation. Their 18.873/19.002 ms timings remain rejected on accuracy.

Evidence: details `an-8iou8bne9a`, regions/final-LLO `an-wu7yvy4xut`,
operator `an-xnkdcw6cx7`, LLO `an-12wxbqo8vs`.

### FP32-contraction compilation constraint

`exp-qkj0w25zju` (`d2033db`, artifact `art-uqbkz5exir`) rejects the mixed
BF16-left/FP32-right HIGHEST contraction at TPU compilation with `Bad lhs
type`. The experiment runner records these as per-candidate errors; a
SUCCEEDED experiment is not success of these kernel configurations.
The follow-up widens BF16 V values exactly to FP32 before this contraction.
Four focused CPU tests pass and assert both FP32 lhs and HIGHEST precision;
another TPU run is required. No model input dtype is changed.

Evidence: details `an-413tzmgvae`, region/LLO audit `an-jb7jzfzd67`,
operator `an-q8htv688fh`, LLO `an-ofg46hnyto`. The seed-28/no-scope corrected
run is `exp-7tdj35x71p`, pinned to `5bcf73f` (artifact `art-y41qjesibl`).

## Forward seed-28 reproduction with region tracing disabled

`exp-7tdj35x71p` completed with `region_trace_mode=none` and libtpu's custom
region trace flag disabled. Thirty samples per candidate confirm:

| Variant | Forward ms | Live PR13 ms | Decision |
| --- | ---: | ---: | --- |
| Native Q2048, compute-KV512 | 19.658 | 20.907 | Retain for joint validation |
| Native Q4096, memory-KV4096 | 19.945 | 20.802 | Smaller gain |
| Native Q4096, unroll 8 | 20.409 | 20.963 | Smaller gain |
| FP32-contract fused Q4096, unroll 8 | 60.521 | 21.029 | Reject |
| FP32-contract fused Q2048, compute-KV512 | 81.520 | 21.127 | Reject |

For the retained Q2048/compute-KV512 candidate, all arrays are finite. Output
differs from PR13 at 6,516 / 75,497,472 elements, relative L2 `2.48309e-5`,
max absolute `2.44141e-4`. dQ/dK/dV relative L2 differences are
`1.38337e-4` / `1.40596e-4` / `1.28228e-4`. Against four full-length FP32
oracle heads, worst candidate/reference L2-error ratios are 1.000009753
(output), 0.993629862 (natural LSE), 1.000052810 (dQ), 1.000008662 (dK), and
1.000003912 (dV). All per-head maximum absolute errors are unchanged.
Together with seed 27, this supports the numerical screening result but
does not establish joint forward/backward accuracy or training convergence.

Exact BF16-to-FP32 V widening makes HIGHEST compile, and fixes the fused
denominator's earlier LSE regression. It does not make fusion useful: output
L2 error versus the oracle improves (worst ratio about 0.8053), but dK L2
error rises by about 2.59% and its max-absolute error by 10.83%, while latency
triples or quadruples. More accurate forward PV does not guarantee every
quantized-backward error metric improves. These fused probes stay rejected.
The current CPU kernel/oracle regression suite passes **176 tests** in
124.43 seconds, including explicit HIGHEST/FP32-operand assertions.

Evidence: details `an-u3hfr26giw`, operator `an-gxdqkovxwh`, LLO
`an-adv2gcsp6h`. All experiments submitted in this iteration and their
requested analyses are terminal and have been inspected through Falcon.

## Motivation for the native output-drain probe

First jointly measure and validate the retained native forward candidate
with the existing transposed-dQ/dK-first backward candidate. Do not infer
joint speedup or accuracy by adding independent screening numbers.

The forward region audit `an-wu7yvy4xut` identifies a concrete remaining
layout cost: `splash_fwd_output_drain` is 0.836 ms for native
Q2048/compute-KV512 and 0.812 ms for Q4096/memory-KV4096, versus PR13's
0.177 ms. Initialization is only 0.023 ms for either native variant versus
0.098 ms for PR13. Thus approximately 0.66 ms of the loop improvement is
lost in normalization/output formatting. A bounded next kernel probe is
direct sequence-minor normalization/writeback, with the public output
layout and precision preserved and any outer conversion included in timing.
Another independent option is removing duplicated full/partial loop bodies
without changing segment semantics. The following sections implement and
test these probes; inner-loop overlap is still not proven optimal.

## Native normalization and physical output layout

Source `c9b88ded464a808a7e192b3ae50c2e32e70a8969` adds two default-off flags:
`fwd_native_output_normalization` keeps FP32 output/state scratch in its
native sequence-minor layout for reciprocal/multiply, then casts to BF16;
`fwd_output_seq_minor` also stores physical Pallas output in that layout.
The wrapper restores the public logical output with `out.mT`. Any outer
conversion is inside the compiled, timed function. No FP8, lower-precision
probability conversion, fused denominator, or remat change is enabled.

`exp-qfrgjymw1x`, artifact `art-6xnz4p09fy`, seed 28, 30 samples:

| Forward variant | Host median ms | Live PR13 ms | Device module ms | Output drain ms |
| --- | ---: | ---: | ---: | ---: |
| PR13 | 20.934 | 20.915 | 20.075 | 0.176 |
| Q2048/compute-KV512, old drain | 20.137 | 20.892 | 18.796 | 0.833 |
| Same, native normalization | 19.443 | 21.379 | 18.034 | 0.103 |
| Same, native physical output | 18.906 | 21.221 | 17.684 | 0.025 |
| Q4096/memory-KV4096, old drain | 19.774 | 21.133 | 18.692 | 0.810 |
| Same, native normalization | 19.326 | 20.950 | 17.941 | 0.088 |
| Same, native physical output | 18.754 | 20.826 | 17.594 | 0.025 |

Device entries use the first of three captured calls, not host timings.
For Q2048, final-body MXU count stays 10,752, while `llo.vxpose` falls
7,168 -> 6,272 -> 6,144. KV-loop time is 17.042 -> 16.993 -> 16.991 ms.
Thus the measured drain benefit is not an inner-dot-count reduction.
Capture-wide IMEM input bytes fall from 12,862,464 / 12 descriptors to
79,872 / 3 descriptors for both native-drain variants. These counters
include three calls and other capture activity; they are not utilization.
For Q4096, IMEM remains about 12.7–13.7 MB and 12 descriptors.

All arrays are finite. Within each tile family, old/native/physical-output
variants have identical reported precision statistics against PR13 and
identical independent-oracle statistics. That is not a direct pairwise
bitwise comparison of the complete arrays. Q2048's worst four-head
candidate/reference L2-error ratios are 1.000009753 (output), 0.993629862
(LSE), 1.000052810 (dQ), 1.000008662 (dK), and 1.000003912 (dV), with all
max-absolute errors unchanged. CPU direct drain comparisons are bitwise
for both output layouts, both reciprocal modes, and both probability
orientations; the initial suite passed 192 tests in 138.78 seconds.

Evidence: details `an-cxxgbm3ymc`, regions/final-LLO `an-2knakdxw5r`,
operator `an-muoegrncny`, LLO `an-gmplgxg5n4`.

## Public custom-VJP joint measurement

The new `--phase joint` compiles public `jax.vjp(kernel)` and returns output
plus dQ/dK/dV. This avoids assuming that manually composing internal
forward/backward functions is identical to the public API: CPU physical-
output probes showed a few gradient differences for the internal
composition. Residual LSE is checked through a separate, untimed standalone
forward and is explicitly labelled `joint_lse_source`; it is not an extra
forward inside joint timing. Actual joint outputs/gradients feed precision
and independent-oracle checks. No model/remat performance is implied.

`exp-p9myb5scfh`, artifact `art-ehvo30nf7n`, source `c9b88de`, seed 28:

| Joint configuration | Median ms | Live PR13 ms |
| --- | ---: | ---: |
| PR13 | 70.028 | 70.058 |
| Backward optimization only | 64.860 | 69.888 |
| Native forward, old drain only | 68.795 | 70.044 |
| Both, old forward drain | 64.095 | 70.046 |
| Both, native forward normalization | 63.162 | 69.847 |
| Both, native physical forward output | 62.742 | 70.023 |

For the last three rows, the precision/oracle statistics are identical:
output differs from PR13 in 6,516 elements (relative L2 2.48309e-5);
dQ/dK/dV differ in 123,213 / 398,073 / 289,210 elements with relative L2
1.38552e-4 / 1.40596e-4 / 1.28228e-4. All are finite. Worst per-head
FP32-reference L2-error ratios are 1.000009753 / 0.993629862 / 1.000053451 /
1.000008662 / 1.000003912 for output/LSE/dQ/dK/dV. All per-head maximum
absolute errors match PR13. As before, this is not automatic acceptance.

Joint device-module time falls from 68.739 to 61.428 ms. The best capture
has approximately 17.174 ms forward KV loops and 42.499 ms backward KV
loops. Forward drain is 0.025 ms and backward dQ/dKV drains total 0.200 ms.
About 0.866 ms is uncovered inside the combined scope span; this includes
the transition between forward and backward and must not be interpreted
as stall time or a utilization metric. Further substantial gains must
address loop work, not just the now-small forward output drain.

Evidence: details `an-1ycmwindq2`, regions/final-LLO `an-p0m9ycy4o0`,
operator `an-svzdx6ic5i`, LLO `an-yfkfyvtvhs`.

## Shared-loop probe

Source `91f4230810abf4bbda261174f3c1ec44e726f8a8` adds default-off
`fwd_kvmajor_single_loop`: only segment masking retains the full/partial
conditional; QK, exp, denominator reduction, and PV share one loop body.
It preserves the original full-tile contract and ID 0 == 0 semantics.
The purpose is to test instruction footprint, not to assume improved
hardware overlap from fewer static instructions.

The first CPU prototype always applied segment equality. Both that and
the narrower conditional-mask implementation are non-bitwise at seed 28.
For the latter, output has 3 differences / 36,864 elements; dQ/dK have
20/64, and dV is bitwise. dK cross-candidate relative L2 is 1.08829e-4,
slightly above an initial 1e-4 diagnostic gate, while its independent-FP64
error ratio is 1.000104978. The final test explicitly gates independent
oracle error with the existing 0.1% + 1e-7 bound, not cross-candidate
distance, and retains strict bitwise tests for drain-only changes. No
independent-oracle tolerance was widened. The complete suite passes
**201 tests** in 143.95 seconds; TPU accuracy remains a separate gate.

`exp-fma6erh63p` (artifact `art-n8jxvv7ipb`) completes the coarse-trace
screen. Shared-loop Q2048 and Q4096/memory-KV4096 give 48.165 and 48.890 ms,
versus matched native-output controls 18.858 and 18.693 ms. Q4096 with
memory-KV8192 gives 46.398 ms (compute-KV256) or 42.982 ms (compute-KV512).
All four shared-loop variants are rejected on performance; all arrays are
finite and matched tile families have unchanged reported oracle statistics.

| Layout | Device ms | KV loop ms | Internal uncovered ms | IMEM bytes / descriptors |
| --- | ---: | ---: | ---: | ---: |
| Q2048 native output | 17.690 | 16.997 | 0.228 | 79,872 / 3 |
| Q2048 shared loop | 46.894 | 37.316 | 9.085 | 15,054,240,768 / 15,381 |
| Q4096/KV4096 native output | 17.593 | 16.903 | 0.222 | 12,710,400 / 12 |
| Q4096/KV4096 shared loop | 47.479 | 38.621 | 8.372 | 15,963,829,248 / 15,381 |

The shared Q2048 body's static MXU count halves (10,752 -> 5,376) while
transpose count falls only from 6,144 to 5,632. This is a failed code-
footprint hypothesis: the new placement of runtime mask branches produces
far more dynamic instruction loading despite fewer counted body operations,
and the loop itself also slows. The capture does not separate instruction
stalls inside the coarse loop from MXU/vector scheduling stalls, so neither
is assigned an invented utilization percentage. All IMEM figures are
capture-wide three-call counters, not per-call memory bandwidth.

Evidence: details `an-8dc4z0pe1v`, regions/final-LLO `an-djuyb88mny`,
operator `an-qcifutgsxm`, LLO `an-eoofhe2twb`.

## Seed-27 joint reproduction without region tracing

`exp-7anbtxoyij` (artifact `art-rl3v5hamql`, source `91f4230`) uses
`region_trace_mode=none` and disables libtpu custom-region tracing. Thirty
samples confirm Q2048 at 63.079 ms versus live PR13 70.272 ms; native
Q4096/memory-KV4096 gives 62.570 versus 70.286 ms. All arrays are finite.

The Q2048 joint candidate has worst four-head FP32-reference L2-error
ratios 1.000002415 / 0.990491405 / 0.999995440 / 1.000013208 / 1.000008779
for output/LSE/dQ/dK/dV. For Q4096 these are 0.999999780 / 1.001046173 /
1.000001200 / 1.000005045 / 1.000010185. All per-head maximum absolute
errors are unchanged. Q4096's full-array relative differences from PR13
are 2.22873e-5 (output), 8.29178e-5 (dQ), 9.19117e-5 (dK), and 7.33053e-5
(dV); base-2 LSE differs by at most 1.90735e-6. LSE remains an untimed
standalone diagnostic; output and gradients come from the actual joint VJP.

Evidence: details `an-7sv0zibbw0`, operator `an-j57hn85lhc`, LLO
`an-p3p25zfxuq`. All three reports have been inspected through Falcon.

## Third seed and all-32-head independent precision validation

`exp-qi1gilizpq` (artifact `art-izvad6ndxl`, source `91f4230`) completes
seed 29 with all 32 full-length FP32 oracle heads, not only 0/15/16/31.
It preserves the same three variants and no-region-trace configuration.
Every reference/candidate oracle array and every full-shape kernel array
is finite. Q2048 measures 62.663 ms versus live PR13 69.893 ms; Q4096 is
62.555 versus 70.098 ms. Thirty-sample p95, computed by linear interpolation,
is 63.097 / 63.126 ms for Q2048/Q4096. Q4096 has one 68.254 ms maximum,
so its small median advantage should not be treated as a universal latency
win over Q2048; neither setting is promoted to default.

The following are candidate/reference error ratios against the independent
FP32 oracle. Aggregate L2 ratios use summed squared error norms over all
heads (`sum(rms_abs**2 * elements)`), not the average of head ratios.

| Array | Q2048 worst head L2 ratio | Q2048 aggregate L2 ratio | Q4096 worst head L2 ratio | Q4096 aggregate L2 ratio |
| --- | ---: | ---: | ---: | ---: |
| Output | 1.000008206 | 1.000000076 | 1.000008658 | 1.000000572 |
| Natural LSE | 0.992838881 | 0.986998417 | 1.004493902 | 0.999930442 |
| dQ | 1.000026383 | 0.999997901 | 1.000028169 | 1.000002145 |
| dK | 1.000052582 | 0.999995215 | 1.000043750 | 0.999997547 |
| dV | 1.000040526 | 0.999997362 | 1.000023891 | 1.000002588 |

Output and all three gradients have exactly the same maximum absolute
oracle error as PR13 on each of the 32 heads. LSE max-absolute error is
unchanged or smaller on every head: Q2048 improves heads 6/12/13/14 and
Q4096 improves head 1. Q4096's worst per-head LSE L2-error ratio rises by
0.4494%, while its aggregate LSE error is slightly lower; do not describe
every LSE error metric as identical. Against PR13, full-array base-2 LSE
differences remain at most 1.90735e-6. The candidate is non-bitwise and
these random-input checks do not certify training convergence or all inputs.

Evidence: details `an-ofzxqfm1ld`, operator `an-j6jwd8cja3`, LLO
`an-y2et4tz3zq`. All reports and declared precision output were inspected.

## Follow-up hypotheses after the joint reproduction

The measured joint improvement is about 11% lower latency / 12% higher
throughput, not the 20–30% target. Forward output formatting is no longer a
large exposed cost. Retain the external full/partial branch and native
drain as the current control. A shared loop with unconditional equality
masking was the next ablation; its forward-only implementation and negative
results are recorded below. Backward KV-loop work remains the largest measured region;
coarse traces still do not prove that its MXU/vector overlap is optimal.

## Branchless shared-loop forward control

Source `c7943ab286d7787f4df48fc18e1e59ea66dd4eed` introduces default-off
`fwd_kvmajor_mask_all_tiles`, restricted to the shared native forward loop.
It applies segment equality unconditionally and removes the per-compute-tile
scalar branch without modifying the backward mask policy. The CPU suite
passes 206 tests, including both mask policies against FP64 at two seeds
and two Q-compute shapes.

`exp-gyakmt7le1` (artifact `art-3t27aopsk3`) completes seven forward cases:

| Variant | Forward ms | Live PR13 ms | Device KV-loop ms | IMEM bytes / descriptors |
| --- | ---: | ---: | ---: | ---: |
| Q2048/compute-KV512 native output | 18.861 | 20.955 | 16.991 | 79,872 / 3 |
| Same, branchless shared loop | 22.353 | 21.006 | 20.414 | 79,872 / 3 |
| Q4096/memory-KV4096 native output | 18.759 | 21.467 | 16.903 | 12,710,400 / 12 |
| Same, branchless shared loop | 23.817 | 21.008 | 22.063 | 79,872 / 3 |
| Branchless Q4096/memory-KV8192 | 28.739 | 21.061 | 22.316 | 6,478,321,152 / 6,162 |
| Previous row, compute-KV512 | 22.049 | 21.261 | 20.501 | 13,879,296 / 12 |

These are all slower than the retained native-output configuration. At
Q2048, removing the runtime mask branch reduces the prior shared-loop
capture's IMEM traffic from about 15 GB to 79,872 bytes and internal
uncovered time from 9.085 to 0.222 ms. Yet the branchless loop still takes
20.414 ms versus the retained outer-branch loop's 16.991 ms. Thus fixing
instruction loading alone is insufficient; extra mask work and/or changed
inner scheduling still leave a regression. No utilization percentage is
inferred. Conditional and branchless shared Q2048 bodies have identical
counted MXU/transpose/load/store instructions (5,376 / 5,632 / 8,560 / 2,824),
despite very different runtime instruction-loading behavior.

All arrays are finite, and matched tile families have the same reported
precision/oracle statistics as their controls. No automatic acceptance or
direct pairwise bitwise equality is claimed from equal summary statistics.
Evidence: details `an-yiq2sv8k95`, regions/final-LLO `an-h3t62ozw40`, operator
`an-tptwoj1jdx`, LLO `an-vlhnnzlap4`; all declared reports were inspected.

## Internal backward Q-compute tiles and bounded pipeline state

Source `09e0b3cad0ae3149859a6719ce69278c098d63d5` adds default-off
`bwd_block_q_compute` and `bwd_qtile_pipeline`. The production outer Q4096 /
KV8192 DMA blocks and gradient output/reduction handling remain unchanged.
Only internal Q compute is subdivided. The paired sequential/staged probes
compute QK, FP32 P, dP and dS before consuming BF16 P/dS in dV, dK, dQ order.
FP32 P is retained through dS; casts occur at the reference gradient-dot
boundaries, not as a lower-precision softmax shortcut. Q subdivision can
reassociate dK/dV reductions and therefore requires independent accuracy
review. The staged version prepares tile i before consuming tile i-1;
this source order alone does not prove hardware overlap.

The hypothesis follows the earlier staged-KV VMEM failure: shrinking the
compute-Q dimension should reduce the simultaneously live probability and
gradient arrays without shrinking the outer transfer blocks. Cases pair
sequential and staged Q1024/KV1024, Q512/KV1024, and Q2048/KV512 compute.
The full CPU suite passes **218 tests** (157.42 seconds); another 12 focused
tests pass after preserving the configured unroll policy. At sequence 1024,
the staged/sequential gradients are directly bitwise equal for both tested
Q-compute sizes and seeds, and all gradients satisfy the existing FP64
oracle criterion. A sequence-2048 CLI smoke with `qtile512_pipeline` also
completes with finite gradients; it is non-bitwise versus PR13.

TPU validation completed as `exp-9j205abzub`, artifact `art-a16riith7j`,
with all eight cases compiling and producing finite gradients. The full-size
backward timings and first device-call coarse regions are:

| Variant | Backward ms | Live PR13 ms | KV-loop ms | Internal uncovered ms |
| --- | ---: | ---: | ---: | ---: |
| PR13 | 49.961 | 50.123 | 46.526 | 0.202 |
| Retained transposed-dQ/dK-first | 45.383 | 50.020 | 42.473 | 0.198 |
| Q1024/KV1024 sequential | 49.080 | 50.067 | 46.255 | 0.198 |
| Q1024/KV1024 staged | 59.048 | 49.990 | 56.285 | 0.203 |
| Q512/KV1024 sequential | 53.001 | 50.125 | 50.170 | 0.198 |
| Q512/KV1024 staged | 62.500 | 50.179 | 59.539 | 0.204 |
| Q2048/KV512 sequential | 49.769 | 50.249 | 46.855 | 0.198 |
| Q2048/KV512 staged | 59.348 | 50.186 | 56.418 | 0.202 |

Every capture reports only 79,872 DIE0 TCS IMEM bytes / 3 descriptors.
Unlike the earlier large-body unroll failures, there is no repeated
instruction-loading surge. Initialization and gradient drain costs remain
near the retained layout's costs. The staged Q1024 loop adds 10.030 ms
versus its matched sequential loop, while internal uncovered time rises by
only about 0.005 ms: the regression is inside the loop, not an exposed
inter-block gap. This does not identify its stall type or prove optimal
MXU/vector overlap.

For final LLO bodies, Q1024 sequential has 2,208 MXU, 1,280 transpose,
2,944 vector-load and 2,448 vector-store instructions. Its staged counterpart
has 4,416 / 2,048 / 4,080 / 2,880. These totals include different prologue,
epilogue and branch structures, so they are **not** dynamic instruction
counts and do not establish a twofold matmul cost. The smaller static bodies
make limited unrolling a separate, still unmeasured scheduling hypothesis.

Independent full-length FP32 checks on heads 0/15/16/31 find unchanged
maximum absolute gradient errors for all variants. Worst per-head L2-error
ratios across the new candidates are 1.000000561 (dQ), 1.000007347 (dK),
and 1.000005690 (dV). Paired sequential/staged variants have equal reported
precision statistics; summary equality alone is not a direct bitwise test.
The retained layout still changes only 302 dQ elements versus PR13, with
dK/dV bitwise. These are diagnostics, not training-convergence certification.
No Q-tiled variant is promoted: all are slower than the retained control.

Evidence: details `an-2wvzbyu88w`, regions/final LLO `an-7fzehbn9df`,
operator `an-newu7r3zxn`, LLO `an-ortghmetnl`; all declared reports and
filtered timing/precision/compiler outputs were inspected.

### Separate steady-state loops from the pipeline's first/last tile

The structure audit `an-i2wor9qwqf` shows SCF loops in these final-LLO
body dumps, not fully allocated machine code. Its corrected loop-region
parser `an-k8jcbrw3ds` separates the loop body from the first/last tile:

| Non-partial loop body | Compute-Q coverage | Carried vector SSA values | MXU | Transpose | Vector loads | Vector stores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained | 4096 | 0 | 4416 | 1728 | 792 | 432 |
| Q1024 sequential | 1024 | 0 | 1104 | 384 | 432 | 216 |
| Q1024 staged steady state | 1024 | 1024 | 1104 | 384 | 468 | 216 |

Thus four sequential Q-subtiles cover the retained loop's Q extent with
the same counted MXU operations but 1,728 versus 792 explicit vector loads
and 864 versus 432 stores. This identifies a data-reuse cost of subdivision;
it is not a measurement of register spills or elapsed memory time. The
staged steady-state body adds only 36 explicit loads versus sequential,
not a doubled matmul body; it also carries 1,024 vector SSA values across
iterations. A source-level two-stage pipeline is therefore not evidence
of useful physical overlap.

Hardware counters independently confirm that every Q-tiled candidate and
the retained layout issue **54,263,808** DIE0 BF16 VREG matmuls across the
three captured calls. Q-tiled variants split these evenly between MXU0/1;
the retained layout splits them 28,311,552 / 25,952,256. PR13 issues
62,914,560. These are dynamic instruction counts, not utilization. Different
execution times at equal matmul counts require scheduling/dataflow analysis.
An initial loop-parser attempt (`an-gqvxlvorvt`) failed to accept SCF closing
braces with trailing attributes; it was corrected and rerun successfully.

## Small-body backward unrolling

This screen tests unroll 2/4 on internal Q1024 and unroll 4 on Q512, paired sequential
and staged. The outer DMA blocks and numerical expressions are unchanged.
The purpose is to test cross-iteration scheduling with a smaller compiled
body than the previously rejected whole-Q unroll probes. It is not a claim
that fewer source loops or more live tiles improve hardware overlap.

Source `953970a1f73940e0d164fb7c2f1a6132c8b049a1` adds the runner cases
and unroll 2/4 precision tests; all **234 CPU tests pass** in 172.17 seconds.
At the tested small shapes, unfolded sequential/staged results are directly
bitwise equal to the corresponding rolled sequential Q-tile implementation.
TPU experiment `exp-wj29g6uxq1` (artifact `art-ussa0ijycy`) completed all
ten cases, with all gradient and sampled oracle arrays finite:

| Backward variant | Latency ms | Live PR13 ms | KV loops ms | Internal uncovered ms | IMEM bytes / descriptors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained | 45.201 | 49.989 | 42.476 | 0.198 | 79,872 / 3 |
| Q1024 sequential, rolled | 49.054 | 50.051 | 46.257 | 0.198 | 79,872 / 3 |
| Q1024 sequential, unroll 2 | 46.878 | 50.279 | 43.954 | 0.200 | 79,872 / 3 |
| Q1024 sequential, unroll 4 | 46.070 | 50.194 | 43.050 | 0.200 | 79,872 / 3 |
| Q1024 staged, rolled | 59.032 | 50.043 | 56.288 | 0.203 | 79,872 / 3 |
| Q1024 staged, unroll 2 | 52.129 | 50.236 | 49.413 | 0.201 | 79,872 / 3 |
| Q1024 staged, unroll 4 | 58.759 | 50.186 | 46.207 | 9.854 | 7,120,811,520 / 9,071 |
| Q512 sequential, unroll 4 | 46.789 | 50.184 | 43.948 | 0.204 | 79,872 / 3 |
| Q512 staged, unroll 4 | 49.786 | 49.862 | 46.957 | 0.208 | 79,872 / 3 |

Limited unrolling recovers some internal-loop cost but does not beat the
retained implementation. Q1024 staged/unroll-4 shrinks its measured loop
regions while creating about 9.85 ms of uncovered intervals and 7.12 GB of
instruction-loading traffic. This is strong evidence that code loading
offsets its internal-loop gains, not evidence of a faster overall kernel.
Its final body has 17,664 MXU / 6,656 transpose instructions, compared with
8,832 / 3,584 for Q1024 sequential/unroll-4. These static totals include
expanded prologue/epilogue and branches, not doubled dynamic work.

Within each Q-compute family, all reported precision statistics and
per-head FP32 oracle errors match the rolled counterpart. Every candidate
retains exactly the reference's maximum absolute gradient error on heads
0/15/16/31; worst L2-error ratios are 1.000000561 (dQ), 1.000005405 (dK),
and 1.000005690 (dV). Equal summaries are not direct pairwise bitwise proof,
and this negative-performance screen does not certify training convergence.
There is no new candidate to promote into a joint forward/backward run.

Evidence: details `an-j7etg5hdj2`, regions/final LLO `an-30odu1zl05`,
operator `an-h54fjs11r6`, LLO `an-gfitg60zhv`; all are terminal and their
declared reports and filtered numerical/trace outputs were read.

## Remaining direction

The unchanged retained joint result is 62.555 versus 70.098 ms: the target
is still unmet. The Q-subtile probes reveal more explicit accumulator
traffic per covered Q extent. A next bounded dataflow experiment is to
retain dK/dV accumulators across inner Q subtiles and drain once per compute
KV block, instead of repeatedly loading/storing them for every subtile.
Keep the FP32 addition order and BF16 dot boundaries, compare against the
same Q-tile control, and recheck both oracle errors and VMEM/code footprint.
The implementation and negative results of this experiment are recorded
below. The current evidence does not establish optimal MXU/vector overlap.

## Nested Q sweep and dK/dV accumulator reuse

The next implementation adds default-off `bwd_qtile_nested` and
`bwd_qtile_accumulator_carry`. For each compute-KV block, the nested path
visits all Q subtiles before proceeding to the next KV block. The carry
path loads that block's existing FP32 dK/dV accumulators once, preserves
the source addition order over Q subtiles, and stores once at the end.
dQ still accumulates directly in its original scratch slice. BF16 dot
operands, FP32 P through dS, softmax/scale expressions, outer DMA tiles,
and final gradient reduction/formatting are unchanged.

The nested-but-scratch-updating control separates loop restructuring from
accumulator reuse. Only the inner Q loop is unrolled; the outer compute-KV
loop stays rolled. Existing flat-loop paths and all defaults are unchanged.
Both sequential and staged inner-Q schedules have CPU tests; the initial
TPU screen focuses on sequential schedules to isolate accumulator reuse.
This does not guarantee that the compiler keeps accumulators in registers
or that the larger live range is profitable. Independent precision checks,
timings and trace/compiler evidence remain required before promotion.

Source `01f5f5d094ce9155015cb7676529f80128f2e407` passes 78 focused CPU
tests and all **284 kernel/oracle tests** (219.86 seconds). Nested/carry,
sequential/staged and unroll 1/2/4 paths are directly bitwise equal to their
matched flat rolled Q-tile controls at the tested CPU shapes/seeds.

`exp-m2iiz18ivs` (artifact `art-ikyymnjlwr`) completes nine full-size BF16
backward cases. No new configuration is faster than the retained control:

| Variant | Latency ms | Live PR13 ms | KV-loop ms |
| --- | ---: | ---: | ---: |
| Retained | 45.543 | 50.107 | 42.475 |
| Q1024 flat / unroll 4 | 45.976 | 50.107 | 43.054 |
| Q1024 nested / unroll 4 | 45.676 | 49.778 | 43.022 |
| Q1024 accumulator carry / rolled | 50.042 | 50.046 | 47.254 |
| Q1024 accumulator carry / unroll 2 | 47.799 | 49.997 | 44.972 |
| Q1024 accumulator carry / unroll 4 | 45.683 | 50.051 | 43.058 |
| Q512 nested / unroll 4 | 46.862 | 50.133 | 43.956 |
| Q512 accumulator carry / unroll 4 | 47.225 | 50.173 | 44.510 |

All captures have 79,872 DIE0 TCS IMEM bytes / 3 descriptors and only
0.198–0.202 ms of internal uncovered intervals. There is no instruction-load
regression masking a gain here. In the non-partial Q1024/unroll-4 loop,
accumulator carry really reduces explicit vector loads **1,728 -> 1,008**
and stores **864 -> 432**, with unchanged 4,416 MXU / 1,536 transpose
instructions. The partial loop's loads fall **2,240 -> 1,520**. These
changes do not improve measured loop time. Thus reducing these explicit
loads/stores alone is insufficient; it is not valid to assert that the
control compiler had already performed the same reuse or that every saved
instruction was on the critical path. These dumps are still SCF/SSA LLO,
not a direct post-allocation spill or hardware-utilization measurement.

The rolled nested carry has 144 vector SSA values in its inner Q-loop
state. Its non-partial inner loop has 216 loads / 72 stores per Q subtile,
plus the one-time outer accumulator load/drain. Counts for nested parent
and child loop bodies overlap and must not be summed as independent work.

All candidate arrays and independent FP32 oracle arrays are finite.
Within each Q-compute family, reported precision statistics match prior
controls; per-head maximum absolute gradient errors are unchanged versus
PR13 on heads 0/15/16/31. Worst L2-error ratios across new variants remain
1.000000561 (dQ), 1.000005405 (dK), 1.000005690 (dV). This is sampled
numerical evidence, not direct full-size pairwise bitwise equality or a
training-convergence certification. No carry configuration is promoted.

Evidence: details `an-8bxjbpaumz`, regions/final LLO `an-ba4kvehezg`,
loop structure `an-4px0bd91gf`, operator `an-s2ugujzhff`, LLO
`an-4oi2sc0ihh`. All five analyses succeeded and their reports were read.

## Larger backward Q aspect-ratio screen

The next bounded screen goes back to the retained, non-Q-subtiled path:
increase memory/compute Q together to 8192 or 16384, while shrinking
compute-KV to keep QK/P/dS tile area bounded. Memory-KV stays 8192, so
the dQ reduction-slot schedule and BF16 output-partial boundaries do not
change merely from adding more memory-KV blocks. Compare against both the
retained Q4096/compute-KV1024 and Q4096/compute-KV512 controls. Larger Q
can amortize accumulations and outer-block setup, but grows Q/dO/dQ buffer
requirements; VMEM capacity, precision and loop/drain timing must be checked.
This is a hypothesis, not a measured speedup.

Source `9f711d330f09392acd3cbb161eb2d0c40119a6d9` passes eight new
sequence-4096 FP64 aspect-ratio tests and all **292 CPU tests** in 235.99
seconds. TPU `exp-dqatlnpnen` (artifact `art-0e4zwcugvb`) finishes with five
measured configurations and two compile failures; the successful Falcon job
status does not mean that every candidate compiled.

| Q / compute-KV | Backward ms | Live PR13 ms | KV-loop ms | Internal uncovered ms |
| --- | ---: | ---: | ---: | ---: |
| Retained 4096 / 1024 | 44.928 | 49.747 | 42.476 | 0.198 |
| 4096 / 512 | 47.620 | 49.808 | 45.154 | 0.199 |
| 8192 / 512 | 46.174 | 49.925 | 43.509 | 0.099 |
| 8192 / 256 | 55.664 | 49.653 | 53.178 | 0.100 |
| 16384 / 256 | VMEM failure | 49.868 | — | — |
| 16384 / 128 | VMEM failure | 50.065 | — | — |

At compute-KV512, larger Q reduces the loop total by 1.645 ms and halves
coarse loop-region count from 512 to 256, with dQ drain 0.171 -> 0.153 ms.
That is evidence of amortization at matched compute-KV, but it is still
slower than retained Q4096/compute-KV1024. All five captured configurations
have 79,872 IMEM bytes / 3 descriptors, so instruction-loading traffic does
not explain the slower Q8192/compute-KV256 loop. No new tile is promoted.

The compiler reports 75.10M required VMEM for Q16384/compute-KV256 versus
63.94M available, including 21.10M register-allocator spill slots. At
compute-KV128 these are 66.35M required and 12.35M spill slots, still over
capacity by 2.41M. Both errors identify an **8 MiB** double-buffered
`s32[8192,128]` KV segment-ID window and an **8 MiB** double-buffered
`bf16[1,1,16384,72]` dQ alias window. These are concrete allocations in
failed configurations, not measured stall costs in the retained kernel.
Neither failed candidate has a valid TPU timing or precision result.

All five measured variants and their four-head FP32 oracle arrays are
finite. For the Q8192 variants, per-head maximum absolute gradient errors
are unchanged versus PR13. Worst L2-error ratios are 1.000005239 (dQ),
1.000002761 (dK), 1.000005480 (dV). Compared with PR13, Q8192/compute-KV512
changes 2,372 dQ, 8,922 dK and 5,640 dV elements; it is non-bitwise and
not a training-convergence certificate. The compute-KV256 case changes
2,838 dQ elements with the same reported dK/dV statistics.

Evidence: details `an-ops4rxklot`, regions/final LLO `an-fi6ofr0h3y`,
operator `an-v9nn7elqbq`, LLO `an-dfxjim3m6m`; all reports and filtered
precision/trace outputs were inspected. All experiments and analyses in
these two follow-up screens are terminal; no job is left running.

## Remaining evidence-led checks

The retained joint result remains 62.555 versus 70.098 ms, below the target
improvement. No new default is enabled. Two concrete follow-ups remain:

1. Complete the compiler-scheduler control on the actual retained
   transposed-dQ/**dK-first** configuration. Existing `dq_transposed_scheduler`
   uses the old dQ-first order; it does not answer this matched question.
   Compare scheduler false/true/default with unchanged dataflow and the
   public joint VJP before attributing any gain to overlap.
2. Explore native-layout segment-ID and dQ alias/output windows to reduce
   padding and double-buffered VMEM pressure. The allocation evidence above
   motivates this, but fitting a rejected tile is not itself a speedup.
   Preserve dQ partial dtype/rounding and segment equality, and include all
   wrapper conversions in joint timing. The existing compact-ID flag alone
   changes logical width, not a proven native-layout allocation reduction.

## Matched compiler-scheduler controls on retained joint VJP

Source `ad59926` adds controls which differ from the retained Q4096 native
forward plus dK-first/transposed-dQ backward only in scheduler flags.
The backward compute-KV remains **1024**, not 256 (256 is the retained
forward compute-KV). Ten focused configuration/public-VJP tests passed.
`exp-beth17rdj8` / `art-8212q4rkcg` runs seed 28, 30 timing samples and
four full-length oracle heads with coarse scopes. No scheduler is promoted.

| Scheduler FWD / BWD | Joint ms | Live PR13 ms | FWD loops ms | BWD loops ms |
| --- | ---: | ---: | ---: | ---: |
| false / false (retained) | 62.5565 | 70.1456 | 16.9013 | 42.4748 |
| false / true | 62.6617 | 70.1755 | 16.9025 | 42.4518 |
| false / compiler default | 62.5304 | 70.0853 | 16.9036 | 42.4749 |
| true / false | 62.7030 | 69.9170 | 17.1379 | 42.4755 |
| true / true | 62.7514 | 69.8477 | 17.1359 | 42.4556 |

Region columns use the first of three profiled calls. Default and false
backward scheduling have essentially the same region durations; the
0.026 ms wall-time difference is not a credible new optimization. Enabling
backward scheduling shifts time between full/partial loops but saves only
about 0.023 ms total, while dK/dV drain grows 0.0277 -> 0.0627 ms. Enabling
forward scheduling worsens total forward loop time and grows its drain
0.0246 -> 0.0703 ms. Joint scope gaps remain 0.860–0.894 ms; these include
inter-kernel transitions and are not a hardware utilization measurement.

Final body instruction counts are unchanged across these scheduler controls:
backward 8,832 MXU / 3,968 transpose / 3,664 load / 2,880 store instructions;
forward 10,752 / 9,216 / 23,968 / 10,768. The capture-wide DIE0 TCS IMEM
bytes are 21,533,184 (retained/default), 21,688,320 (BWD true), 21,823,488
(FWD true), and 21,978,624 (both), each 18 descriptors across three joint
calls. This joint footprint must not be confused with the earlier
79,872-byte standalone backward captures. Static counts do not quantify
overlap, and these controls do not establish optimal scheduling.

All candidate arrays and oracle arrays are finite. Every scheduler variant
reports the same PR13-distance and four-head oracle statistics as the
retained candidate; matching summaries are not direct pairwise bitwise
proof. Output/dQ/dK/dV/LSE maximum absolute oracle errors are unchanged
versus PR13 on all four heads. Worst per-head L2 error ratios are output
1.000003214, dQ 1.000012902, dK 1.000008502, dV 0.999997155, and LSE
1.001825766. No threshold was relaxed.

All declared reports read: details `an-dgaq2t645g`, regions/final LLO
`an-bxayjxmye2`, operator `an-11xegeqsrv`, LLO `an-a0gtq83hha`.
The generic operator plugin does not discover nested per-variant trace
directories; the custom region reader explicitly reads all six profiles.

## Native dQ partial output and alias layout

Source `c7897072efa9c667f40d90c406cbb5e63e3497cd` adds default-off
`bwd_dq_output_seq_minor`. The Pallas partial output and optional aliased
input use `[reduction_slot, head, D, Q]`; the scratch drain no longer
transposes FP32 dQ back to Q-major inside the custom call. Partial dtype,
BF16 cast before alias addition, `j % 3` slot assignment, and the final
slot reduction are preserved. Public Q-major shape is restored after
reduction and this conversion is included in timing.

The flag requires fused backward and sequence-minor scratch; it rejects
MQA/GQA, grouped heads, and the direct-output path without dQ scratch.
Sixteen public-VJP CPU cases cover two seeds, two asymmetric Q blocks,
BF16 and FP32 non-aliased partials, three-slot alias collisions, and
segmented tails including allowed `0 == 0`. They are directly bitwise
equal to their matched controls. All **314 kernel/oracle tests** passed
in 249.56 seconds. TPU evidence is recorded below, not inferred from CPU.

`exp-a57ghp768i` / `art-rqryw8kod2` (seed 28, coarse regions, 30 samples)
shows a small backward improvement, but does not rescue large-Q performance:

| Backward variant | Wall ms | Live PR13 ms | Device module ms | KV loops ms | dQ drain ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained Q4096 / compute-KV1024 | 45.0283 | 49.9645 | 43.9261 | 42.4792 | 0.1721 |
| + native dQ output | 44.7237 | 49.8759 | 43.3462 | 42.4881 | 0.0388 |
| Q8192 / compute-KV512 | 46.0029 | 50.0100 | 44.8441 | 43.5129 | 0.1539 |
| + native dQ output | 45.5922 | 49.9268 | 44.2572 | 43.5041 | 0.0389 |
| Native Q16384 / compute-KV128 | 72.3823 | 49.8481 | 71.2666 | 70.5601 | 0.0383 |

Device/region columns are the first profiled call, not wall-time components
measured in the unprofiled samples. At the retained tile the primary loop
does not improve; dQ drain saves 0.133 ms. Module time outside the coarse
scope span also drops about 0.464 ms. The latter includes surrounding
operations and cannot be attributed to one HLO without finer evidence.
Final backward body MXU count stays 8,832; transpose falls 3,968 -> 3,456,
loads 3,664 -> 3,296, stores increase 2,880 -> 3,024. All measured captures
still have 79,872 DIE0 TCS IMEM bytes / 3 descriptors. Thus this is a native
output/alias formatting improvement, not improved main-loop overlap.

Native Q16384/compute-KV128 now compiles where the old layout required
66.35M versus 63.94M VMEM, but its 70.56 ms main loop rejects it. Native
Q16384/compute-KV256 still fails: 68.02M required versus 63.94M, compared
with 75.10M before. Its explicit compiler spill allocation is 21.02M
(previously 21.10M), and the doubled KV-segment-ID window still occupies
8.00M. The new alias operand has `[3,32,72,32768]` physical shape. The
failed candidate has no timing or precision result despite the runner's
overall SUCCEEDED state; it is not counted as a measured configuration.

All measured outputs and oracle arrays are finite; four-head maximum
absolute oracle gradient errors are unchanged. Native and non-native
controls at each matched Q tile have identical reported PR13-distance
and oracle statistics. At the retained tile dK/dV remain bitwise equal
to PR13; dQ differs in 302 elements (max 0.000244140625, relative L2
7.70492534e-6), exactly the retained control's statistics. Worst dQ oracle
L2 ratio is 1.000000561. No threshold is relaxed and matching summary
statistics do not establish pairwise bitwise equality.

Evidence: details `an-hx4589f7gp`, regions/final LLO `an-e5n6mv2gvl`,
operator `an-ds0l7yoay3`, LLO `an-0mnggklqob`; all reports read. The next
check is the actual joint public VJP, tracing disabled, with a different
seed and all 32 full-length oracle heads; standalone backward timings
alone do not promote a joint candidate.

### Joint no-scope, all-head oracle validation

`exp-fyifkyaus0` / `art-cj9ttjfbsh` uses the same `c789707` source,
seed 29, all **32** full-length FP32 HIGHEST-precision oracle heads, and
both `region_trace_mode=none` and custom-call region tracing disabled.
The public VJP timing returns output/dQ/dK/dV and includes all wrapper
conversions. LSE is checked separately using the untimed standalone forward.

| Joint variant | Median ms | Live PR13 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| PR13 self-control | 69.9817 | 69.9252 | 70.2044 | 70.3253 |
| Previous retained Q4096 native FWD | 62.3787 | 69.9968 | 62.5916 | 62.6063 |
| + native dQ output/alias | **61.9334** | **70.2271** | **62.2037** | **62.2455** |

p95 is linearly interpolated over 30 samples. The new opt-in candidate
saves 0.4453 ms (0.714%) against the matched retained candidate in this run.
Against its live PR13 measurement it is **11.810% lower latency / 13.391%
higher throughput**. Reference drift is exposed in the table, not hidden
by comparing only against an old run. This is a small, trace-supported
layout gain, not achievement of the 20–30% goal or a full-model/remat result.
It remains default-off and requires no precision, model or device-count
change.

All full-size candidate arrays and all 32-head oracle arrays are finite.
The new candidate's PR13-distance summaries and every head's complete
oracle statistics are identical to the previous retained candidate in
this run. Output/dQ/dK/dV max absolute oracle errors are unchanged versus
PR13 at every head; LSE max absolute error never increases (one head
improves). As before, this is not direct pairwise bitwise proof or a
training-convergence guarantee.

| Value | Worst per-head L2 error ratio vs PR13 | Aggregate L2 error ratio |
| --- | ---: | ---: |
| Output | 1.000008658 | 1.000000572 |
| dQ | 1.000028169 | 1.000002145 |
| dK | 1.000043750 | 0.999997547 |
| dV | 1.000023891 | 1.000002588 |
| LSE | 1.004493902 | 0.999930442 |

The LSE worst-head relative metric is not claimed unchanged versus PR13;
its tiny absolute errors and aggregate metric must be read alongside it.
These statistics reproduce the earlier retained Q4096 seed-29 validation.
No accuracy acceptance bound is loosened for the new layout.

Evidence: details `an-0syf524g2x`, operator `an-lx22rutgkm`, LLO
`an-4g0a3e3a4o`. All analyses reached SUCCEEDED; declared reports were read.

The next bounded layout check is a genuinely sequence-minor KV segment-ID
window, not the existing `(KV,1)` compact-width flag. The failed Q16384
allocation still exposes 8 MiB in its double-buffered `(8192,128)` int32
ID window. A new reader must cover ordinary, Q-tiled and Q-major backward
mask paths without changing segment equality or forward behavior. Fitting
another tile will still require separate timing, precision and region
validation; no large-tile or overlap improvement is assumed.

## Native backward KV segment-ID windows

Source `6b619e95608a6eb0ad35fafa1b16fd981180f1c6` adds default-off
`bwd_kv_segment_ids_seq_minor`. It exposes the int32 KV IDs as `(1, KV)`
and maps each backward memory tile to `(1, block_kv_dkv)`. Local mask
construction reads the compute slice as a logical column, without the
old input's per-token 128-column broadcast. The original compact-width
flag is independent: `(KV,1)` is not assumed to remove physical padding.

Ordinary and staged backward masking share the layout-aware mask helper;
Q-compute-tiling and Q-major probability paths use the same ID-column
reader. Forward inputs, integer width/equality (including `0 == 0`),
BF16 dot operands, FP32 softmax/accumulation and gradient reduction
boundaries remain unchanged. No fast path is enabled by default.

All 44 new CPU cases are directly bitwise equal to their matched controls.
They cover two seeds, ordinary/Q-tiled/staged-Q/Q-major/staged-KV paths,
compact IDs on/off, partial-only masks on/off, negative and >8-bit IDs,
zero-ID tails, NumPy structural masks, causal mask functions and multiple
heads. The initial staged test lacked its required single-mask-body
configuration; fixing that test premise made all cases pass without
changing numerical tolerances. All **358 kernel/oracle tests** then passed
in 289.25 seconds. This CPU evidence does not certify TPU scheduling or
allocation behavior; those measurements follow below.

`exp-xq27dol6o8` / `art-mhew4bxhtf` (seed 28, coarse scopes) completes six
measured cases and three caught compilation failures:

| Backward case | Wall ms | Live PR13 ms | KV loops ms |
| --- | ---: | ---: | ---: |
| Native dQ Q4096 / compute-KV1024 | 44.6911 | 49.8002 | 42.4831 |
| + native KV IDs | 44.5778 | 50.0490 | 42.4282 |
| Native dQ Q8192 / compute-KV512 | 45.4925 | 50.1197 | 43.5028 |
| + native KV IDs | 45.2778 | 49.6161 | 43.3482 |
| Native dQ+IDs Q16384 / compute-KV256 | 54.2643 | 50.2267 | 52.3098 |

The small matched improvements occur mainly in partial-mask regions:
Q4096 partial loop 13.6814 -> 13.6275 ms while the full loop remains
28.802 ms; Q8192/compute-KV512 partial loop 16.6130 -> 16.4581 ms while
the full loop remains 26.890 ms. All measured captures have 79,872 DIE0
TCS IMEM bytes / 3 descriptors. Q4096 final body MXU/store counts remain
8,832/3,024; explicit loads fall 3,296 -> 3,169, transpose rises 3,456 ->
3,584. The smaller input needs a local transpose for the logical mask.
These are static final-body counts, not runtime utilization.

The failed **matched Q8192/compute-KV1024** compilations provide direct
allocation evidence: KV-ID input window `s32[8192,128]`, two buffers,
**8,388,608 bytes**, becomes `s32[1,8192]`, two buffers,
**65,536 bytes (64 KiB)**. No physical-padding assumption is needed.
Total required VMEM falls 71.00M -> 64.33M against a 63.94M limit, but
register-allocation spill slots rise 36.75M -> 38.02M, leaving a 404 KiB
overflow. Both cases remain unexecuted, with no precision/timing result.
This distinguishes saved input-window capacity from the compiler's
resulting total working set.

Q16384/compute-KV256 now fits (the dQ-only layout needed 68.02M), but its
54.26 ms runtime rejects it. Q16384/compute-KV512 still needs 78.80M with
39.73M explicit spill slots, so it is not a promising near-fit tile.

All measured candidate/oracle arrays are finite, and all four heads'
maximum absolute gradient oracle errors are unchanged. At both matched
Q4096 and Q8192/compute-KV512 tile sizes, new and old ID layouts report
identical PR13-distance and complete oracle statistics. Q4096 dK/dV remain
bitwise PR13; the 302 differing dQ elements and worst dQ oracle L2 ratio
1.000000561 are unchanged. The new Q16384/256 sample's worst ratios are
dQ 1.000005239, dK 1.000007177, dV 1.000006744; it is rejected for speed,
not promoted based on fitting. Summary equality is not direct pairwise
bitwise proof. No precision bound changes.

Evidence: details `an-oltv3ic5qx`, regions/final LLO `an-kar7yfujxd`,
operator `an-txr50gvah2`, LLO `an-en4zff4u2r`, all reports read.
The ID-layout gains are small; joint no-scope/all-head validation is still
required before replacing the retained 61.933 ms configuration.

Source `14f38a052b9d82d17b315599dc9cb9df70e0350f` adds two runner-only
controls: combine native KV IDs with the existing compact Q-ID input at
Q4096, then try Q8192/compute-KV1024. This is motivated by the measured
404 KiB overflow, not by assuming a larger tile is faster. The combination
is already covered by the 44 new bitwise CPU cases above.

### Compact Q-ID follow-up

`exp-cmx0nxrr8i` / `art-ob2a65jroa` keeps seed 28 and coarse regions.
At Q4096, adding compact Q IDs to native KV IDs improves backward median
44.5187 -> 44.2301 ms (live PR13 50.0990 / 50.1461 ms). The trace's full
KV loop remains 28.802 ms; the partial-mask loop improves 13.6297 ->
13.2962 ms, making total loop time 42.4313 -> 42.0969 ms. Drain timings,
0.203 ms internal scope gaps, and 79,872 IMEM bytes / 3 descriptors are
unchanged. Final body MXU/transpose/store counts remain 8,832/3,584/3,024;
loads rise 3,169 -> 3,173. This is a partial-mask-layout/scheduling benefit,
not fewer gradient dots, and counts alone do not prove its detailed
hardware-overlap mechanism.

Both Q4096 controls have identical PR13-distance and complete four-head
oracle statistics: all finite, unchanged maximum absolute oracle errors,
dK/dV bitwise PR13, and the same 302 dQ mismatches / 1.000000561 worst
dQ L2 error ratio. No acceptance threshold changes. Joint no-scope and
all-head validation remains required before promotion.

Q8192/compute-KV1024 advances to a different compiler failure:
`CompileTimeScopedVmemOom`, requiring **63.74M** against the configured
**63.00M** scoped limit, exceeding it by 760 KiB. It has no runtime or
precision result. This is distinct from the prior capacity-stage 64.33M
versus 63.94M failure; the next test must address the explicit kernel
budget, rather than claiming another layout change is required.

Evidence, all reports read: details `an-bjxt1r895n`, regions/final LLO
`an-6wuik67qu0`, operator `an-oozrvudnpy`, LLO `an-bgdyc32skt`.

Source `5038a875d5cc78ca4aaa85fdcc25800c0df90252` adds the joint native
dQ + native KV-ID + compact Q-ID validation case. Source
`73aef2ba43eabf9db54882c224629c4839716c6b` adds a Q8192/1024 control
with `bwd_vmem_limit_bytes=64*1024**2`. The JAX 0.11 local compiler-parameter
documentation requires an enclosing `xla_tpu_scoped_vmem_limit_kib` strictly
above that budget, so the budget screen uses 65537 KiB for **all** cases,
including remeasured PR13 and the Q4096 control. Hardware-capacity checks
remain enabled; changing a budget does not create additional VMEM.

### Joint all-head validation of native compact IDs

`exp-l5c175b961` / `art-8ms3gjs8dy`, source `5038a875d5cc78ca4aaa85fdcc25800c0df90252`,
uses seed 29, no custom scopes, public `jax.vjp`, 30 timing samples and
independent full-length FP32 oracle checks for all 32 heads.

| Joint configuration | Median ms | p95 ms | Live PR13 ms |
| --- | ---: | ---: | ---: |
| Previous native dQ output | 62.0467 | 62.3203 | 70.1894 |
| + native KV IDs and compact Q IDs | 61.5991 | 61.8511 | 70.1631 |

The matched improvement is 0.4476 ms (0.7214% latency reduction). Relative
to live PR13, the new candidate reduces latency by **12.2059%** and raises
throughput by **13.9028%**. This replaces 61.9334 ms as the retained
configuration, not as an assertion of a noise-free cross-run improvement.
The 20–30% goal remains unmet.

All 32 heads' complete candidate oracle statistics and all full-array
PR13-distance statistics match the previous native-dQ control exactly.
All arrays are finite. Against PR13, per-head maximum absolute output and
gradient oracle errors are unchanged; LSE maxima do not increase. The
worst per-head L2-error ratios remain output 1.000008658, dQ 1.000028169,
dK 1.000043750, dV 1.000023891 and LSE 1.004493902. Aggregate ratios remain
output 1.000000572, dQ 1.000002145, dK 0.999997547, dV 1.000002588 and
LSE 0.999930442. These statistics do not prove pairwise bitwise equality
to the previous candidate or all-input/training convergence equivalence.
No precision tolerance is relaxed. LSE is checked separately and is not
a fifth timed joint output.

Evidence: details `an-hc2nwv7gxm`, operator `an-4pr4skkqtm`, LLO
`an-6q92m9jaid`, all declared reports read. The generic operator plugin's
zero trace count does not imply no nested variant traces were captured.

### Q8192 budget probe: fitting does not make it fast

`exp-mk1nalz0vh` / `art-0veplufqel`, source
`73aef2ba43eabf9db54882c224629c4839716c6b`, successfully compiles and
executes Q8192/compute-KV1024 with a 64 MiB kernel budget. PR13, Q4096 and
Q8192 use identical enclosing process flags. Medians are PR13 49.9276 ms,
compact/native Q4096 44.4623 ms, and Q8192 **51.8902 ms** (its live PR13
50.0220 ms). Q8192 is rejected for speed.

The first device call is 42.9152 -> 50.6456 ms. Its KV-loop union changes
only 42.1068 -> 42.4904 ms: full regions 28.8067 -> 26.2100 ms, partial
regions 13.3002 -> 16.2804 ms. Internal uncovered scope time instead rises
**0.2047 -> 7.5274 ms**. The three-call capture's DIE0 Any2IMEM traffic
rises from **79,872 bytes / 3 descriptors** to **3,710,180,352 bytes /
2,319 descriptors**. This strongly motivates a code-footprint test; it
does not establish a hardware-utilization percentage or prove every
uncovered interval is instruction DMA.

Final-body static MXU/transpose/load/store counts change
8,832/3,584/3,173/3,024 -> 17,664/5,632/4,761/4,032. They include both
full and partial bodies and are not dynamic instruction counts. Q8192
has the same 302 dQ mismatches to PR13 as Q4096; dK/dV have 8,922/5,640
mismatches. All four sampled heads are finite and have unchanged maximum
absolute gradient oracle errors. Worst per-head L2-error ratios are
dQ 1.000000561, dK 1.000002761 and dV 1.000005374. No promotion follows
from these diagnostic precision checks.

Evidence: details `an-zb7iuy4x4m`, regions/final LLO `an-obhmndjjto`,
operator `an-tf3f739u3a`, LLO `an-yyf1gm785n`, all reports read.
The next bounded test reuses the existing segment-only shared body with
the newly compact/native ID layouts at Q4096 and Q8192. It must measure
both instruction-loading gaps and redundant full-tile masking cost;
eliminating the Q8192 regression alone is not progress over the retained
Q4096 baseline.
