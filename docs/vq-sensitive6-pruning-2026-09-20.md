# VQ layer-sensitive expert pruning, 2026-09-20

Status: mask-only qualification complete. The 384-expert candidate is retained
for a future physical checkpoint export; it is not the daily model and no RAM
or speed saving is claimed yet.

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
not used here.

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

## Decision

- Keep **Sensitive6 + 384** as the sole physical-export candidate. It is the
  only tested allocation below 40 GiB of projected main weights while passing
  both the thinking gate and full HumanEval without an observed quality loss.
- Do not promote 288: one 30-row score is insufficient for the aggressive
  43.75% pool reduction, and its quick 20/70/100 non-thinking gate was 2/3.
- Do not prefer 448: its IFBench score was two points higher than 384, but it
  uses 3.269 GiB more projected main weights, scored three fewer HumanEval
  tasks, and doubled the number of length-limited HumanEval generations.
- Before daily use, physically export 384, verify mask/export token parity,
  rerun IFBench/HumanEval, then measure resident memory, PP, fixed-input decode,
  native-MTP acceptance/throughput, and long-context retrieval.

## Agentic evaluation follow-up

HumanEval and IFBench are not agentic benchmarks. For coding-agent evidence,
use SWE-bench Verified under one fixed public scaffold; for broader terminal
work use Terminal-Bench. Both require a containerized evaluation environment,
which was not available on this Mac during this run. A small fixed subset is a
valid development gate only when clearly labeled; it must not be reported as a
full official score. GAIA targets general assistants, while tau-bench targets
multi-turn tool/policy interaction and is less direct for this coding-focused
engine.
