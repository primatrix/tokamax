# ViT Splash region tracing: 2026-09-22

## Current result

Best screened backward setting: sequence-minor scratch/dO plus transposed dQ,
with **dK before dQ**. Across repeated full-shape runs it measures about
45.3 ms versus PR13 about 50 ms. With region scopes removed and seed changed
to 28, it measures 45.313 versus 49.738 ms: 8.9% lower latency / 1.098x
throughput. Forward remains about 21 ms; the combined attention target of
20–30% improvement has **not** been reached. No full-model speedup is claimed.

All new orientation/pipeline controls remain default-off, BF16 inputs/FP32 softmax are preserved,
and model rematerialization is unchanged. dK/dV are bitwise equal for the best
candidate; dQ differs in 302–303 of 75,497,472 elements across seeds 27/28.
Independent-reference errors are essentially unchanged on four full-length
heads per seed, but this is not training-convergence certification.

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
