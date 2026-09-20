# Results guide

This is the compact index of retained `qwen38-flash.cpp` measurements. It
separates production results, historical controls, specialized upper bounds,
and rejected models. Exact prompts, hashes, distributions, and sampling rules
remain in the linked source documents.

Status: the optimization phase was closed on 2026-09-19 at the user's request.
VQ remains the daily-use baseline; no runtime defaults or historical scores
changed in the [documentation closeout](daily-use-2026-09-19.md).

## How to read the numbers

- **PP tok/s** is prompt processing. **Decode tok/s** is generated tokens per
  second. They are different phases and must not be combined.
- **Single-stream** measures one request. **Aggregate** measures multiple
  independent requests and is not per-user latency.
- **Target-only** executes the authoritative model once per token. **MTP** is
  speculative and depends strongly on draft acceptance.
- A small IFBench pilot is a development gate, not a substitute for all 300
  prompts. HumanEval scores are comparable only when the prompt/stopping
  protocol is identical.
- `GiB footprint` and `GB MLX peak` come from different accounting tools. They
  are preserved as measured and should not be directly equated.

Common hardware unless a source row says otherwise: Apple M5 Pro MacBook Pro,
18 CPU cores, 64 GiB unified memory, macOS 26.5, AC power, and no active thermal
controller.

Latest follow-up: [September 19 qualification](vq-quality-preserving-2026-09-19.md)
documents a QSA rollback correctness fix, full-program HumanEval scoring,
and rejected PP/decode experiments. No new speed record is claimed by that fix.

The [SWE-bench Verified three-task pilot](swebench-verified-pilot-2026-09-20.md)
adds bounded agentic-coding evidence. Under one fixed mini-SWE-agent scaffold,
v1 resolved 2/3 and v2 resolved 1/3; v2 also took more calls and wall time.
This is a directional development subset, not a full 500-task score. The run
also found and fixed an FP32 packed-QSA threadgroup-memory overflow.

## Current production target: VQ 2.1bpw

The upstream mixed-codebook [v2 upgrade evaluation](vq-v2-upgrade-2026-09-19.md)
is complete for the bounded short-context suite: HumanEval 153/164 versus the
historical v1 146/164, but fresh paired IFBench 14/30 versus 18/30. Paired warm
PP is 535.59 versus 526.68 tok/s; short target-only decode and physical footprint
are essentially unchanged. Both revisions are retained and v1 remains the daily
default. The later three-task agentic pilot also favored v1. The tables below
remain the v1 historical scorecard.

The [persistent-verifier follow-up](persistent-verifier-2026-09-19.md) was
rejected: S=5 target verification took 84.57 ms versus 62.84 ms for the existing
path. A native single-group QSA selector bug was fixed; production MTP and its
scorecard remain unchanged.

The latest [production-dataflow experiment](vq-production-dataflow-2026-09-19.md)
retains experimental native state/cache/recovery improvements, but does not
change the production scorecard: an inherited EOS rule was fixed (native IF3
recovered to 3/3), while compact verifier / GPU-indirect PP prototypes were slower.

Model: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, native packed-14 routed
VQ, exact top-10 routing, Q8 dense backbone, checkpoint PLE, optional native Q6
MTP sidecar, MLX 0.32.2.

| Area | Protocol | Result |
|---|---|---:|
| Prefill | 7,091 repository tokens, same-process cold/warm | 455.45 / **519.91 PP tok/s** |
| Prefill | 14,173 repeated repository tokens, cold/warm | 446.3 / **479.1 PP tok/s** |
| Decode | fixed-input 64 steps, three starts | 30.55 / 30.72 / 30.59; **30.59 median** |
| Native MTP | 64 outputs, coding fixture, 49/56 accepted | **57.94 tok/s median** |
| HumanEval original tests | 164 full-program chat tasks, greedy/no-thinking, max 768 | **146/164 MTP**, 46.93 aggregate decode tok/s; **147/164 target-only**; [protocol](vq-quality-preserving-2026-09-19.md#full-quality-qualification) |
| 32K context | 32,024 prompt tokens, Q8 KV, MTP off | **22.68 tok/s**, 39.5 GiB |
| IFBench | first 30, native MTP, greedy/no-thinking | **18/30 strict and loose**, 35.28 aggregate tok/s |
| IFBench control | same first 30, target-only | **14/30 strict and loose**, 28.55 aggregate tok/s |
| Thinking pilot | first 10, sampled xhigh bounded thinking | **8/10 strict and loose**, 28.60 aggregate tok/s |
| SWE-bench Verified pilot | three fixed tasks, mini-SWE-agent, native MTP | **2/3 resolved**; v2 control **1/3**; directional subset only |
| Exact B=2 probe | two 64-step independent streams | **41.35 vs 30.61 aggregate tok/s (1.351x)** |
| HTTP B=2 gate | concurrent IFBench keys 20/70 | **25.46 vs 23.76 aggregate tok/s**, byte-identical |

Qualified VQ workloads peak around 36.3--39.6 GiB. The current target has not
been requalified at 128K and no VQ 128K performance is claimed.

Sources: [detailed VQ evaluation](benchmark-history.md#current-vq-evaluation),
[public quality evaluation](public-quality-evaluation.md#vq-21bpw-native-quality-gate),
and [prior-research ledger](prior-research-ledger.md).

## Cross-model comparison

These rows answer different questions and are not one universal leaderboard.

| Short name | Tested artifact or lineage |
|---|---|
| VQ 2.1bpw | `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, revision `64b0fb0f98a552d91fb9abd5531d547b2e78c8a8` |
| REAP-288 Q4 | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` lineage plus the project's verified Q8/Q4 MTP and SSD-PLE assets |
| Qwen3.8-27B Q4 | `mlx-community/Qwen3.8-27B-4bit` through the external `mlx-vlm` control runner |
| Qwen3.8-27B Splash | The same MLX Q4 target lineage repacked for Splash 1.0, plus its model-specific five-layer DFlash 2 draft |
| Qwen3.6-35B-A3B Splash | Upstream MLX 4-bit MoE repacked for Splash 1.0, plus its model-specific six-layer DFlash 2 draft |
| Niwaki 99B | `neopolita/Qwen3.8-Flash-Next-99B-A5B-Niwaki-3bit-mlx` |
| Niwaki 113B | `neopolita/Qwen3.8-Flash-Next-113B-A5B-Niwaki-3bit-mlx` |

| Checkpoint | Single-stream decode | Best MTP / special decode | Quality evidence | Memory evidence | Decision |
|---|---:|---:|---|---:|---|
| VQ 2.1bpw | **30.59** | **57.94** native MTP fixture; **46.93** full HumanEval run | HumanEval 146/164 MTP; IFBench best-retained 18/30; thinking pilot 8/10 | 36.3--39.6 GiB | Current target |
| REAP-288 Q4 | **41.06** | **71.06** automatic MTP | IFBench full 300: 34.67% strict; EvalPlus HumanEval 90.85% | 38.3--40.8 GiB | Historical reference |
| Qwen3.8-27B Q4 | **17.21** on EvalPlus run | 50.22 aggregate in four-request smoke | Thinking IFBench subset 70%; EvalPlus HumanEval 91.46% | 17.29 GB MLX | Quality/control runner |
| Qwen3.8-27B Splash | **45.71** fixed thinking B=1 aggregate; 80.00 HumanEval median | **102.61** fixed thinking B=4 aggregate | HumanEval **154/164**; IFBench thinking/16K **30/30**; MBPP+ **294/378** | 23.39 GiB constrained 128K peak; 28.14 GiB paired campaign peak | External quality choice; not a Flash-Next runtime |
| Qwen3.6-35B-A3B Splash | **139.56** fixed thinking B=1 aggregate; 271.87 HumanEval median | **255.19** fixed thinking B=4 aggregate | HumanEval **149/164**; IFBench thinking/16K **27/30**; MBPP+ **284/378** | 21.78 GiB constrained 128K peak; 39.22 GiB extended campaign peak | External throughput default; not a Flash-Next runtime |
| Niwaki 99B Q3/Q4 | **41.60** at retained 128K needle | **70.08** external MTP at 16K | Ten-case/broad gates insufficient for promotion | 39.30 GiB at retained 128K | Research only |
| Niwaki 113B 3-bit | 37.76--40.07 in pilots | No qualified MTP | 0/3 corrected bounded-thinking gates | 26.6--41.3 GiB | Rejected |

The corrected external comparison changes the daily-use decision. Dense 27B is
the stronger local quality model, while 35B-A3B is about 3.05x faster at B=1
and 2.49x faster at B=4 on the fixed thinking workload. The current VQ
checkpoint saves weight storage, not necessarily runtime footprint: its VQ
codebooks, dense backbone, state, and Metal resources still place the qualified
server near 40 GiB. See the [paired Splash report](splash-comparison-2026-09-20.md).

## Quality lookup

| Benchmark and protocol | VQ 2.1bpw | REAP-288 Q4 | Qwen3.8-27B Q4 | Niwaki 113B |
|---|---:|---:|---:|---:|
| IFBench greedy/no-thinking, first 30 | 60.00% with native MTP; 46.67% target-only | 50.00% in historical concurrent control | Not run under this exact 30-row protocol | Not run |
| IFBench bounded thinking, first 10 | **80%** pilot | 50% on stratified 10 | **70%** on stratified 10 | Not run |
| IFBench bounded thinking, corrected 3-case | Not the retained VQ gate | **2/3** | Not run | **0/3** |
| IFBench full 300, greedy/no-thinking | Not run | **34.67% strict / 39.67% loose** | Published model-card result is not locally protocol-equivalent | Not run |
| HumanEval raw completion, 164 | Not run | **81.10%** target-only Q8 PLE | **80.49%** | Not run |
| HumanEval complete-program chat, original tests, 164 | **147/164 target-only; 146/164 native MTP** | Not rescored | **150/164 historical rescore** | Not run |
| Historical EvalPlus no-thinking chat runner, 164 | See complete-program row | **149/164 (90.85%)** | **150/164 (91.46%)** | Not run |
| SWE-bench Verified, fixed three-task pilot | **2/3 v1; 1/3 v2** | Not run | Not run | Not run |

The final Splash controls scored **154/164 HumanEval and 30/30 thinking
IFBench** on dense 27B, versus **149/164 and 27/30** on 35B-A3B. HumanEval used
no-thinking while the matched IFBench run used thinking and a 16K output
budget. See the [paired external-runtime protocol](splash-comparison-2026-09-20.md).

The VQ 8/10 thinking pilot demonstrates a promising protocol, not that the VQ
checkpoint has completed or matched the full published IFBench benchmark.
See [public-quality-evaluation.md](public-quality-evaluation.md) for exact
prompts, stopping rules, failure analysis, and official-score limitations.

## Long-context and capacity lookup

| Model | Context | Result | Status |
|---|---:|---:|---|
| VQ 2.1bpw | 32,024 | 22.68 decode tok/s; 39.5 GiB | Qualified VQ row |
| Niwaki 99B | 16K | 946.24 PP / 36.36 decode tok/s | Directional rank-64-map run |
| Niwaki 99B | 65K | 757.55 PP / 31.07 decode tok/s | Directional rank-64-map run |
| Niwaki 99B | 128K | 671.82 PP / 29.38 decode tok/s | Research frontier; broad quality still blocks promotion |
| Niwaki 99B | 128K | 610.74 PP / 41.60 decode tok/s | Retained persistent-Metal needle run |
| REAP-288 | 128K | 550.92 median PP / 20.56 median decode tok/s | Three cold retrieval runs |
| REAP-288 | 192K | 281.17 PP / 4.19 decode tok/s | Needle recovered; only 0.25 GiB over safety floor |
| REAP-288 | 262K | no valid completion | Not claimed |
| Qwen3.8-27B Splash | 131,024 | **219.03 PP / 38.67 native decode tok/s** | 40 GiB hard ceiling; 32-token exact-replay decode; directional |
| Qwen3.6-35B-A3B Splash | 131,030 | **1,161.30 PP / 135.85 stream decode tok/s** | 40 GiB hard ceiling; 32-token exact-replay decode; directional |

The two Niwaki 128K rows use different runtime/numeric paths and are both kept
because one represents the low-rank MLX research frontier and the other the
persistent-Metal decode frontier. They must not be merged into a synthetic
single result.

## Throughput, caching, and specialized extremes

| Feature | Retained result | Interpretation |
|---|---:|---|
| REAP four-request continuous decode | 45.70 aggregate vs 41.5--41.7 serial | Exact outputs; aggregate throughput, not 45 tok/s per request |
| REAP rolling 12-request IFBench | 39.20 vs 38.35 aggregate | Only +2.23% decode; refill prefill limited end-to-end gain |
| Exact routed-MoE four-slot gate | 45.02 vs 38.74 aggregate (**1.162x**) | 12/12 byte-identical |
| Generated-turn RAM prefix reuse | 5,584.72 ms to 157.45 ms | About **35x** prompt reuse for the matched prefix |
| Prompt-copy speculation | first run **113.44 tok/s**; later 92.54/87.47 | Verbatim grounded re-emission only; all 110 copy proposals accepted |
| Mixed lifecycle soak | 256 valid + 64 malformed + SSE disconnect | Ready/idle at end; no monotonic footprint growth |

## Compression and architecture experiments

| Experiment | Best observed effect | Decision |
|---|---:|---|
| Packed Q4/g16 KV at 65K, isolated layer | 0.669 vs 0.784 ms Q8; 48 vs 68 MiB | Not promoted: 3.716% relative L2 vs 0.455% Q8 |
| Cross-layer QSA candidate reuse | 65.6--100% recall even with an 8x pool | Rejected: insufficient exact recall |
| Raw-margin MTP scheduling | apparent 2.1--3.7% mixed-prompt movement | Rejected: confounded by equal movement in an unchanged arm |
| Q4 dense GDN copies | up to 35.30 tok/s from about 33.6 | Rejected: changed trajectory or exceeded memory ceiling |
| Lossless UInt16 VQ indices | about 1% isolated gain | Rejected: about 2 GiB extra residency and no material benefit |

See [DeepSeek transfer probes](deepseek-v41-transfer-probes.md) for the packed
KV/QSA details and [prior-research ledger](prior-research-ledger.md) for the full
accepted/rejected experiment history.

The [September 19 follow-up](vq-throughput-round2-2026-09-19.md) also separates
new padded-tile and QSA component measurements from paired full-model results.
No component latency improvement should be added to this scorecard as an
end-to-end speed claim.

## VQ expert pruning

Both conservative VQ pruning paths were already tested with routing masks before
top-k. Mask tests keep all 512 physical experts resident, so they establish
model behavior but not RAM savings.

| Test | Unpruned VQ | Public REAP-384 map | VQ-aware REAP-448 v1 | VQ-aware REAP-448 v2 | VQ-aware HOPE-384 |
|---|---:|---:|---:|---:|---:|
| IFBench first 30, non-thinking strict | **14/30** | 13/30 | **15/30** | not rerun | 12/30 |
| IFBench first 30, non-thinking loose | **14/30** | **14/30** | **15/30** | not rerun | 12/30 |
| IFBench bounded-thinking keys 0,10,...,90 | **8/10** | 7/10 | 5/10 | 6/10 | skipped after first-30 gate |
| EvalPlus HumanEval chat, 164 | **145/164** | 140/164 | skipped after thinking gate | skipped | skipped after first-30 gate |
| Decode evidence | about 30.33--30.61 fixed-input tok/s | about 30.45 fixed-input | **30.61 fixed-input** | not rerun | 28.85 aggregate IFBench decode |
| Physical saving | none in mask test | theoretical 7.783 GiB checkpoint reduction | theoretical 3.892 GiB checkpoint reduction | same geometry | none; exporter rejected |

REAP-448 v1 used 16,526 effective calibration tokens from held-out IFBench
keys 100--299. V2 added 13,103 generated reasoning/final tokens and changed 517
removed layer-expert choices, but recovered only one of the three lost thinking
cases. Both had zero request errors. The larger 448 pool performing worse than
the public 384 map on thinking shows that calibration coverage, not retained
expert count alone, controls the loss.

The later HOPE experiment used 41,900 independent calibration tokens and
conditional pairwise expert contributions. HOPE-384 alone passed the 3-case
gate, but fell to 12/30 on the expanded first-30 set. HOPE-448 and HOPE-288
already failed the 3-case gate at 2/3. The runtime masks retained every weight,
and unchanged top-10 work produced no useful speed signal.

Decision: do not repeat these uniform pruning runs or export any map. Revisit
only with a substantially broader independent agentic/coding/reasoning
calibration corpus and layer-sensitive budgets. Pruning does not address
target-only decode while top-10 activated experts remain unchanged.

### September 20 layer-sensitive follow-up

The materially different allocation suggested above was tested: layers
`0,1,31,35,36,39` retain all 512 experts, while the other 42 use the VQ-aware
HOPE rankings at 288, 384, or 448 experts.

| Test | Unpruned v1 | Sensitive6 + 288 | Sensitive6 + 384 | Sensitive6 + 448 |
|---|---:|---:|---:|---:|
| IFBench first 30, strict/loose | 14/30 | 15/30 | **15/30** | **17/30** |
| Bounded-thinking 20/70/100 | not freshly rerun | not expanded | **3/3** | **3/3** |
| HumanEval original 164 | 145/164 historical | not run | **152/164** | **149/164** |
| Projected physical main weights | 45.780 GiB | 34.341 GiB | **39.243 GiB** | 42.512 GiB |
| Actual mask-test footprint | 36.7 GiB | 36.7 GiB | 37.0 GiB | 36.9 GiB |

Those rows are MTP-off diagnostics. Required MTP-on qualification subsequently
rejected the 384 allocation: v1 scored 14/30 versus its unpruned 18/30 control;
v2 scored 13/30 versus 14/30, emitted 17,547 tokens with two truncations, and
its fixed 64-token MTP fixture fell from a fresh paired 53.66 to 46.03 tok/s.
The mask kept every weight resident, so no memory saving was measured and no
physical export is retained. Full protocol and MTP telemetry are in the
[qualification report](vq-sensitive6-pruning-2026-09-20.md).

## Product targets

These are deferred research targets, not pending automatic work after closeout.
"Met" below only applies to the stated workload scope.

| Goal | Current evidence | Status |
|---|---:|---|
| PP >= 600 tok/s | VQ 519.91 warm | Open |
| Target-only decode >= 40 tok/s | VQ 30.59 median | Open |
| MTP decode >= 60 tok/s | VQ 57.94 retained fixture | Not met; mixed-workload target unqualified |
| Peak footprint <= 40 GiB | VQ 36.3--39.6 GiB qualified rows | Met only on qualified workloads; not an all-context cap |
| Stable automatic configuration | one normal serving path | Met |
| Public quality comparable with 27B | partial VQ IFBench only | Open |

## Document map

- [Detailed VQ evaluation](benchmark-history.md#current-vq-evaluation): full retained
  performance table and implementation notes.
- [Historical reference evaluation](benchmark-history.md#historical-reference-evaluation):
  REAP, 27B, Niwaki, long-context, concurrency, and soak rows.
- [Public quality evaluation](public-quality-evaluation.md): complete local
  quality protocol and comparison details.
- [Benchmark contract](benchmark-contract.md): what a reproducible claim must
  contain.
- [Prior-research ledger](prior-research-ledger.md): exhaustive optimization
  history and negative results.
- [Model capabilities](model-capabilities.md): supported tensors and checkpoint
  compatibility.
