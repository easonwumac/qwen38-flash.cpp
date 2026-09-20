# VQ layer-sensitive expert pruning, 2026-09-20

Status: rejected after MTP-on qualification. The MTP-off mask-only result was
promising, but the required daily-use path regressed quality on both v1 and v2,
regressed v2's fixed-fixture throughput, and saved no physical memory.

## Question and design

This experiment tested whether the VQ-aware HOPE rankings become useful when
six sensitive routed-MoE layers keep all 512 experts while the other 42 layers
retain 288, 384, or 448 experts. Layers `0,1,31,35,36,39` remain unmasked.
Routing still selects exactly top-10 from the surviving pool. The shared
experts, codebooks, dense backbone, PLE, and all retained expert weights are
unchanged.

The tested daily checkpoint is VQ 2.1bpw **v1**, revision
`64b0fb0f98a552d91fb9abd5531d547b2e78c8a8`. The separately downloaded v2
checkpoint is revision `87d89bc328f8226deb95e26d9a69c7cd1f353007`; it was
added to the later MTP-on qualification described below.

The runtime test uses an additive router-logit mask before softmax/top-k. Every
physical weight remains loaded, so the observed 36.6--37.0 GiB footprints are
mask-test overhead, not the memory result of pruning. The lightweight manifest
views reference the base checkpoint's shards without copies or symlinks.

## Results

Hardware: Apple M5 Pro, 18 CPU cores, 64 GiB, macOS 26.5, MLX 0.32.2. MTP and
prefix cache were off. Requests were serial. There was no active thermal
conditioning; throughput from the long serial sessions is descriptive and is
not a clean speed A/B.

| Candidate | IFBench first 30 | IF30 output tokens | Thinking gate 20/70/100 | HumanEval | HE output tokens / length-limited | Mask-test peak | Projected physical main weights |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unpruned v1 control | 14/30 | 23,089 | historical control only | 145/164 historical | 43,843 historical | 36.7 GiB | 45.780 GiB |
| Sensitive6 + 288 | 15/30 | 14,110 | not expanded | not run | not run | 36.7 GiB | 34.341 GiB |
| Sensitive6 + 384 | **15/30** | **11,957** | **3/3** | **152/164 (92.68%)** | 38,405 / 4 | 37.0 GiB | **39.243 GiB** |
| Sensitive6 + 448 | **17/30** | 20,541 | **3/3** | 149/164 (90.85%) | 40,109 / 8 | 36.9 GiB | 42.512 GiB |

IFBench used the public 300-row dataset's first 30 rows, greedy/non-thinking,
temperature zero, max 4,096, and the official strict/loose scorer; strict and
loose were equal for all four rows above. The fresh unpruned control reproduced
the historical 14/30 result exactly. The thinking gate used temperature 1,
top-p 0.95, top-k 20, seed 0, xhigh bounded thinking, and max 4,096.

HumanEval used all 164 original tasks, EvalPlus-style non-thinking complete-
program prompts, greedy decoding, max 768, EvalPlus 0.3.1 sanitization, and
network-denied three-second subprocess tests. These are original HumanEval
tests, not HumanEval+. The unpruned HumanEval row is the retained historical
control rather than a fresh paired rerun.

The main-weight projections are exact byte-accounting estimates for slicing
the router rows plus routed gate/up/down `codes` and `vq_scales` on the 42
masked layers. Codebooks and fixed tensors remain unchanged. They exclude
optional MTP and vision sidecars. Actual RSS/footprint must be measured only
after a physical export; allocator, mapped pages, state, and KV do not scale
directly with checkpoint bytes.

## Speed interpretation

Keeping top-10 means the active expert arithmetic is unchanged. The mask-only
IFBench aggregate decode rates were 28.12, 25.64, and 23.67 tok/s for 288, 384,
and 448; the fresh control was 26.46 tok/s. These runs had different output
lengths and thermal histories, so the movement is not attributed to expert
count. A later fixed-input probe was invalidated after the long thermal load
and is intentionally omitted.

A physical export could improve capacity and possibly locality, but it is not
expected to deliver a proportional decode gain. Prefill was not promoted from
these response-level measurements.

## MTP-on daily-use qualification

The first campaign intentionally disabled MTP to isolate target pruning. That
is not sufficient for the product path: automatic native Q6 MTP with the Q4
draft language head is the daily default. The 384 allocation was therefore
rerun on both checkpoint revisions with automatic depth, greedy non-thinking
generation, no prefix cache, serial requests, and the same public first-30
IFBench protocol. The lightweight manifests used each revision's matching
external MTP directory; no sidecar was copied or linked.

| Candidate | IFBench first 30 | Output tokens | Aggregate decode | Draft acceptance | Fixed 64-token MTP fixture | Peak footprint |
|---|---:|---:|---:|---:|---:|---:|
| Unpruned v1, retained control | **18/30** | 12,567 | 33.73 tok/s | 71.23% | 55.69 tok/s paired historical | 39.57 GiB |
| Sensitive6 + 384, v1 | 14/30 | 12,452 | 25.61 tok/s | 3,718/6,560 (56.68%) | 54.27 tok/s, 49/56 accepted | 40.09 GiB |
| Unpruned v2, retained control | **14/30** | 12,490 | 29.38 tok/s | recorded in the v2 report | **53.66 tok/s fresh paired**, 48/56 accepted | 39.3 GiB fresh fixture |
| Sensitive6 + 384, v2 | 13/30 | 17,547 | 31.11 tok/s | 6,496/9,345 (69.51%) | **46.03 tok/s**, 50/56 accepted | 40.25 GiB |

The fresh unpruned v2 fixture used the same binary and automatic settings as
the masked run. Its 53.66 tok/s median is close to the retained 54.86 result,
while the mask's 46.03 tok/s has a much slower verifier (1,125.14 versus
872.86 ms median). Higher acceptance did not compensate for the changed target
routing and verifier cost. The two v2 IFBench truncations also raised output to
17,547 tokens; its higher aggregate tok/s is therefore not a quality or
time-to-answer win.

The v1/v2 peaks slightly exceed the nominal 40-GiB goal and, more importantly,
the mask retains every weight. These measurements cannot be interpreted as
pruned-checkpoint memory savings. Full HumanEval was not rerun after both
first-30 MTP gates failed; the earlier MTP-off HumanEval values remain useful
diagnostics, not promotion evidence.

## Decision

- Reject **Sensitive6 + 384** for physical export. It failed the MTP-on
  first-30 quality gate on both checkpoint revisions, slowed v2's fixed MTP
  path by 14.2%, and the mask itself saved no memory.
- Do not promote 288: one 30-row score is insufficient for the aggressive
  43.75% pool reduction, and its quick 20/70/100 non-thinking gate was 2/3.
- Do not prefer 448: its IFBench score was two points higher than 384, but it
  uses 3.269 GiB more projected main weights, scored three fewer HumanEval
  tasks, and doubled the number of length-limited HumanEval generations.
- Future expert-pruning candidates must use MTP-on as the promotion default.
  MTP-off remains a diagnostic control, and no HumanEval or physical export is
  justified after a candidate fails the first-30 MTP gate.

## Agentic evaluation follow-up

HumanEval and IFBench are not agentic benchmarks. For coding-agent evidence,
use SWE-bench Verified under one fixed public scaffold; for broader terminal
work use Terminal-Bench. Both require a containerized evaluation environment,
which was not available on this Mac during this run. A small fixed subset is a
valid development gate only when clearly labeled; it must not be reported as a
full official score. GAIA targets general assistants, while tau-bench targets
multi-turn tool/policy interaction and is less direct for this coding-focused
engine.

## Drafter placement and v2 optimization boundary

CPU or ANE draft execution is not promoted from this result. On the fresh
unpruned-v2 fixture the complete native draft phase used 235.80 ms for 56
proposals, or about 4.21 ms per proposal, while target verification used
872.86 ms. A prior measured Core ML ANE probe needed 4.318 ms for the language
head alone, before the MTP block, recurrent state, conversion and GPU/ANE
handoff. The retained Metal head was faster. A CPU drafter would also contend
for unified-memory bandwidth and no compatible target-aware standalone CPU
draft model is present locally.

Single-request proposal and verification are causal: the next proposal block
depends on the accepted/rejected target result. Idle CPU cores do not make
those phases safely parallel. ANE remains potentially useful only if a complete
fixed-shape target-aware drafter, including handoff, measures below the current
end-to-end draft phase; moving only the language head has already failed that
gate. A generic small model would additionally consume memory/KV and must first
prove sufficient target acceptance.

v2 still has optimization room, but not an established v2-specific shortcut.
Its retained unpruned baselines are 535.59 warm PP, 30.23 target-only and 54.86
native-MTP tok/s. Layers 2--47 use the small d4 codebook for down; only nine of
48 routed layers use it for gate/up. The K256 tables already fit cache.
Precomputed codebook dots, packed-word reuse, `float4` lookup, prefill word
reuse, cross-row expert reuse and threadgroup staging were all measured and
rejected.

A fresh full-target S=5 profile measured v2 at 59.5445 ms versus v1 at
66.1002 ms, so mixed d4 is already about 9.9% faster in the verifier. Replacing
the retained `2+2+1` MoE split with `4+1` kept parity but measured 59.631 ms.
A native five-row MoE path improved the same-process verifier microbenchmark
from 61.1052 to 59.3785 ms (2.9%), but six interleaved HTTP passes over code,
JSON, explanation and creative prompts were noise-equivalent: output hashes,
proposal paths and acceptance were identical, while candidate/control rates
overlapped from 33.17 to 52.62 tok/s. It was removed.

The final v2-specific probe loaded the d4 BF16 activation and FP16 codebook in
four-element vectors while preserving the scalar accumulation order. It kept
exact output bits but moved the complete layer-27 S=5 MoE only from 1.8253 to
1.8103 ms (0.8%). It was also removed. The dominant remaining work is the
common target verifier/VQ trunk; future effort should require a measured
end-to-end reduction rather than another isolated d4 lookup micro-optimization.
