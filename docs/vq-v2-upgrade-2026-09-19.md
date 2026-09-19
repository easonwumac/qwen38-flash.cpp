# VQ v2 checkpoint upgrade — 2026-09-19

Status: download, compatibility, HumanEval, IFBench 30 and paired short-context
performance checks complete. **This is a trade-off, not an unconditional
upgrade.** v2 improves the measured coding score, loses four net passes on the
IFBench pilot, and has essentially unchanged short decode and footprint.
Both revisions are retained; v1 remains the daily path pending the user's
selection. No model was deleted or replaced in place.

## Decision scorecard

Fresh paired runs unless marked historical. Rates are tok/s; distributions and
protocols are detailed below. [Machine-readable results and artifact hashes](results/vq-v2-upgrade-2026-09-19.json).

| Workload | v1 | v2 compact d8 / FP16 d4 |
|---|---:|---:|
| HumanEval original tests, native MTP | 146/164 (89.02%), historical | **153/164 (93.29%)** |
| IFBench first 30, native MTP | **18/30 (60%)** | 14/30 (46.67%) |
| 7,064-token PP, warmed median | 526.68 | 535.59 |
| Target-only 64-token fixture, median | 30.46 | 30.23 |
| Native MTP 64-token fixture, median | 55.69 | 54.86 |
| IFBench aggregate decode | 33.73 | 29.38 |
| HumanEval aggregate decode | 46.93, historical | 49.59 |
| Target-only PP/decode session peak footprint | 38.29 GiB | 38.26 GiB |
| Native-MTP evaluation session peak footprint | 39.57 GiB | 39.66 GiB |

The 1.7% warmed-PP difference is small, with only three within-session warmed
samples per model; independent repeats are needed to establish a robust gain.
No large speed or memory improvement is established. IFBench v2 gains key 4 but loses keys 1, 17,
19, 20 and 23; its three-case regression is 2/3 versus v1's 3/3. These small
pilots do not establish a universal model ranking. v2 has not been qualified at
32K/128K, with concurrent requests, or for multi-turn SSD state restoration.

## Attribution: quantization versus engine

**Not resolved by these scores.** The fresh paired measurements establish the
behavior of each checkpoint under this engine, not the accuracy of a separate
reference implementation. Kernel unit tests do not prove full-model parity.

All five newly failing IFBench prompts involve exact counts: key 1 misses a
keyword count; keys 17/19/20 require 2/13/3 numbers; key 23 requires seven
numbers plus four different coordinating conjunctions. For example, compact-v2
emits no numbers for keys 17 and 20, and repeats `meridian` twice instead of
once for key 1. These outputs finish normally, not by the token limit. This is
a specific instruction-following regression, not evidence of a global collapse
in intelligence; HumanEval improves in the measured protocol.

The original-FP16 and compact-d8 v2 executions pass exactly the same 14 IFBench
prompts. That argues against the new compact-codebook preparation being the
cause of this particular score difference, but does not establish numerical
identity or rule out a shared engine bug. Tokenizer/template and draft tensors
are identical across versions. d4 kernels pass local independent CPU/scalar
oracles; full routing, recurrent state and verifier parity remain separate
questions.

MTP is another confounder: the retained v1 target-only control scores 14/30,
while fresh v1 MTP scores 18/30. In exact arithmetic target verification should
preserve the target's greedy decisions; this engine's batched and serial paths
have different floating-point evaluation orders and are not token-identical.
Do not reinterpret MTP as a generally smarter model or assign the entire
18-to-14 difference to checkpoint quantization.

The follow-up MTP-off diagnostic selects three observed regressions, keys
17/19/20. Same final binary, identical prompts, greedy non-thinking, max 4,096,
all prefix caches off: **v1 passes 3/3; v2 passes 0/3**, strict and loose.
All six responses finish normally; telemetry confirms zero MTP proposals.
This is a targeted diagnostic, not an unbiased new quality score. It rules out
MTP alone as the cause of those three failures, but does not distinguish
checkpoint changes from the v2 execution path. [Responses and evidence](results/vq-v2-attribution-2026-09-19.json).

The remaining diagnostic is v2 original-FP16 execution against an independent
reference using identical token IDs, MTP off. Inspect the first differing
logits and layer/state values, rather than only final pass/fail; small
near-tie floating-point differences are not automatically kernel bugs.
Reference execution must respect the memory budget; do not load two full
engines together. This independent full-model comparison has **not** been
completed in this upgrade suite, nor has the full paired MTP-off IFBench 30.

## Measured candidates

Both candidates use the same pinned v2 weights, native Q6 MTP, Q4 draft LM head,
greedy non-thinking generation, no prefix cache, one request at a time. The
short fixture is `merge_sorted_unique`, max 64 tokens, one warmup followed by
three measured requests. These are same-session samples, not independent starts.

| Numeric policy | MTP fixture median [min, max] tok/s | IFBench first 30 | IFBench aggregate decode |
|---|---:|---:|---:|
| Original FP16 codebooks, `e358af4` | 47.66 [46.18, 47.84] | 14/30 strict and loose | 23.97 tok/s |
| Existing compact d8 policy, d2/d4 FP16 | 54.86 [54.84, 54.89] | 14/30 strict and loose | 29.38 tok/s |

The original-FP16 run produces 15,896 output tokens, with one 4,096-token
truncation and no HTTP errors. Its per-request decode median [P10, P90] is
24.17 [18.93, 34.18] tok/s. Both three-case regressions score 2/3; the counting
instruction fails. Both short MTP fixtures accept 48/56 proposed tokens.
These scores do not establish a quality improvement over v1.

The compact-d8 run produces 12,490 output tokens with no HTTP errors. Its
per-request decode median [P10, P90] is 27.79 [24.65, 40.07] tok/s. Aggregate
decode improves 22.5% over the FP16 control, but output lengths differ; this is
not a fixed-token latency comparison. The same 14 prompts pass in both policies.

Full original-test HumanEval, compact-d8 v2 with native MTP, scores **153/164
(93.29%)**, versus the retained historical v1 native-MTP result of 146/164
(89.02%). Use this as a protocol-matched historical comparison, not a fresh
paired v1 rerun. Eight former failures pass and one former pass fails. The new
run has zero HTTP errors, ten length-limited responses, 46,373 output tokens,
49.59 aggregate decode tok/s and per-request median [P10, P90] of
50.86 [46.95, 52.78] tok/s. Accepted drafts total 36,195/39,069 (92.64%).

HumanEval protocol: 164 original tasks, one greedy non-thinking complete-program
generation per task, max 768 output tokens, serial requests, EvalPlus 0.3.1
complete-program sanitizer, original tests in network-denied subprocesses with
a three-second timeout. This is not HumanEval+. The generation artifact records
the default `stop_profile=project`, but that option is ignored in
`evalplus-nonthinking` mode: no stop strings are sent, and grading consumes
`raw_completion`, not the post-extracted function body.

The compact-d8 session (smoke, IF3, MTP fixture, IF30 and HumanEval) peaks at
39.66 GiB physical footprint and 30.62 GiB RSS, with 8.08 GiB minimum estimated
reclaimable memory. Its guard exits 143 for the intentional shutdown, not a
memory-limit failure.

The original-FP16 session peaks at 39.44 GiB physical footprint, 24.80 GiB RSS,
with 7.33 GiB minimum estimated reclaimable memory. Guard exit 143 records the
intentional shutdown, not a memory-limit failure. The second candidate restores
the existing d8 cache **per projection family**, without requiring the down
projection to also be d8. This adds approximation relative to v2 FP16, so it
requires a separate quality gate; it is not a bitwise-only optimization.

The v1 layer-2 control keeps the exact output-bit hash
`18052187228802284419` across that constructor change. No d4 codebook is further
quantized. No per-user performance profiles are introduced.

The fresh v1 native-MTP control reproduces **18/30 strict and loose** on the
same IFBench inputs, versus v2's 14/30. v1 generates 12,567 output tokens with
zero errors and no length truncations: aggregate 33.73 tok/s, median [P10, P90]
28.44 [26.86, 45.94]. v2 has no truncations either. This is a small pilot, not
full IFBench or evidence that v2 is universally worse. The HumanEval improvement
and IFBench regression must both remain visible when choosing a daily model.

The fresh v1 MTP fixture is 55.69 [55.50, 55.85] tok/s, accepting 49/56 drafts,
versus v2's 54.86 [54.84, 54.89], accepting 48/56. The v1 control session peaks
at 39.57 GiB footprint, 37.02 GiB RSS, with 8.27 GiB minimum estimated
reclaimable memory. It does not include a fresh HumanEval run; session peaks
therefore have different workload coverage. RSS differences are not evidence
of comparable savings in unified-memory physical footprint.

## Paired performance protocol

The follow-up target-only comparison uses `long_context_benchmark.py --lines
440`: 7,064 chat-template tokens, 23,383 UTF-8 prompt bytes, numbered notes
containing `alpha beta gamma delta epsilon zeta eta theta`. Each fresh server
runs one first-request sample and three warmed samples, generating one token.
Neither RAM nor SSD prefix cache is enabled. This is not the historical
7,091-token repository corpus, and the one-token generation timing is not a
decode-speed measurement. Default Q8 KV admission starts at 8,192 tokens, so
this 7K prompt does not cross that threshold.

After the four PP requests, target-only decode uses the same
`merge_sorted_unique` 64-output-token fixture, one warmup and three measured
requests. This is real autoregressive output, not the historical fixed-input
64-step microbenchmark. Native-MTP comparisons use a separate fresh server:
`OK` smoke, IFBench keys 20/70/100 at max 512, the MTP fixture, then IFBench
first 30 at max 4,096. No concurrent inference or active thermal controller.

v2 target-only PP: first request 405.56 tok/s; warmed 530.10 / 537.01 / 535.59,
median **535.59 tok/s**. Target-only fixture: median **30.23 tok/s**, range
29.96–30.30. This session peaks at 38.26 GiB footprint and 26.24 GiB RSS,
with 9.64 GiB minimum estimated reclaimable memory.

v1 target-only PP: first request 440.94 tok/s; warmed 513.33 / 526.68 / 528.91,
median **526.68 tok/s**. Target-only fixture: median **30.46 tok/s**, range
30.27–30.48. This session peaks at 38.29 GiB footprint and 28.28 GiB RSS,
with 9.57 GiB minimum estimated reclaimable memory. All PP requests have zero
cached prompt tokens and identical one-token output hashes. Warmed decode
samples are within-session, not independent process restarts.

Prompt UTF-8 SHA-256:
`a0237643373b9a189e01ac862bb9401205651be76dc4242674129dd3729a2059`.
Use the script's fixed prompt and pinned tokenizer, not a rewritten README.

## Model identity

Both versions are `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`:

| Version | Pinned revision |
|---|---|
| Retained v1 control | `64b0fb0f98a552d91fb9abd5531d547b2e78c8a8` |
| v2 candidate | `87d89bc328f8226deb95e26d9a69c7cd1f353007` |

The candidate is the author's mixed-codebook release, not our own re-quantization
or expert pruning. It retains 512 routed experts, top-10 selection, the Q8 dense
backbone and VQ PLE. The native Q6 MTP sidecar remains separate from the trunk.

| Projection family | v1 | v2 |
|---|---|---|
| Layers 0–1, all three projections | d2/K256, direct U8 | Unchanged |
| Layers 2–47, down | d8/K16384, packed-14 | d4/K256, packed-8 |
| Gate/up in layers 27–33, 35, 47 | d8/K16384, packed-14 | d4/K256, packed-8 |
| Other gate/up | d8/K16384, packed-14 | Unchanged geometry |

The author's card reports better approximation to its BF16 teacher at essentially
the same disk size. Those KL/top-1 agreement results are not IFBench/HumanEval
scores, and disk bytes are not a prediction of this engine's physical footprint.
[Pinned model card](https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw/blob/87d89bc328f8226deb95e26d9a69c7cd1f353007/README.md).

## Engine changes

- Add d4/K256 packed-8 weight-tile reconstruction to segmented prefill, including
  fused gate/up and tail tiles. Permit gate/up and down to have different
  supported geometries; gate and up must still agree.
- Extend the existing ten-expert slot-packed down reduction to d4. Restrict the
  specialization to its actual 640-wide input; other widths use the generic
  reduction. The reduction order and dtype rounding remain unchanged.
- No routing, attention, KV, PLE or speculative acceptance policy changes.
- Fail closed if the old `QWEN38_PERSISTENT_VQ=1` experimental backend is
  requested for d4/mixed layers. Its legacy d2/d8-only kernels must not consume
  v2 data. The daily automatic MLX/Metal path supports the new geometry.
- The initial compatibility control uses original FP16 expert codebooks for
  v2 decode and wide prefill: the old compact-codebook predicate requires all
  three projections to be d8. The second candidate prepares compact d8 gate/up
  independently of down geometry. It retains FP16 d2/d4 and wide prefill;
  v1's numeric policy is unchanged. Keep both policies distinct in comparisons.
- Version the SSD state-cache numeric-policy key so the mixed-layer FP16 and
  compact-d8 executions cannot reuse each other's persisted states. Old cache
  files are not deleted; the new namespace simply does not consume them.

## Verification

- Clean Release build, MLX 0.32.2.
- Eight CTest targets pass with both v1 and v2 tokenizer assets.
- Nine segmented CPU-oracle cases: d2, d4 and d8, row tiles 8/16/24, shuffled
  source rows and incomplete tails; check both projection and fused SwiGLU.
- Twenty-four slot-packed/scalar down comparisons: d4/d8, batch 1/2/5,
  FP16/BF16, with/without fused shared-expert addition. Exact output bits match.
- Manifest fixture accepts a mixed d8 gate/d4 down layer. Packed-8 index
  round-trip and supported-geometry predicates have CPU regression coverage.
- Real v2 layers 2 (mixed d8/d4) and 27 (d4) complete S=1, S=5 and 128-row
  smoke checks with finite outputs. Batch/serial maximum absolute differences
  are 0.000488/0.056641 at layer 2 and 0.004425/0.005493 at layer 27 for S=5/128;
  these are **not** bitwise whole-forward parity claims. Wide GEMM and serial
  routing/reduction use different floating-point execution paths.
- The explicit persistent-backend opt-in is rejected on real v2 metadata before
  backend allocation, with the expected unsupported-geometry diagnostic.
- HumanEval runner/grader unit tests: 13 cases, one optional-dependency skip;
  IFBench runner CLI tests: five passed. These test the tools, not model quality.

Platform: M5 Pro, 18 CPU
cores, 64 GiB, macOS 26.5, MLX 0.32.2; AC power, no active thermal conditioning.
Only one full-model process may run at a time. Use the user-approved 40 GiB
startup admission threshold and 40 GiB physical-footprint stop threshold;
thresholds are not measurements. The first guarded start was refused at 41.1
GiB estimated reclaimable versus 42 GiB required; the user authorized lowering
only startup admission to 40 GiB. Runtime footprint/RSS/availability stops
remain 40/38/6 GiB. All completed sessions stop deliberately with signal 143;
none triggers a memory-limit stop. All test servers are stopped after the runs.

The final clean Release build passes 8/8 CTest. The v2 compact quality/MTP
binary and final paired-test binary differ only by the SSD numeric-policy
namespace marker; all measured runs disable SSD cache. Binary identities and
31 raw-artifact hashes are in the machine-readable record. The pre-existing
user edits to two evaluation scripts are preserved and excluded from this
milestone commit.

## Asset safety

The new revision is staged independently. Existing files reused across revisions
are checked against the pinned upstream SHA-256 before copying; no symlinks and
no deletion of the v1 checkpoint. Never reuse a derived Q4 draft LM head solely
because the model names match: verify its source tensors or regenerate it.

All 152 repository files are present with matching sizes; all 141 LFS files
pass pinned SHA-256 verification (47.913 GiB including MTP and auxiliary assets).
The locally derived Q4 draft head is additional, 357,581,270 bytes; its SHA-256 is
`e732a67b7eeb4d78dc47bebe2dec0851c6c6049275a7d7286c436f249833360d`.
It was regenerated from v2 `model-00019.safetensors`, source SHA-256
`7695d789f0d4f79eb09f58a1b88e2b32eb2d12575159a03949b31e025315a6e9`.
The conversion peaked at 2.25 GiB physical footprint, not a model-inference peak.
The regenerated v2 draft head and retained v1 draft head have different container
hashes, but all three tensor payloads, shapes and dtypes are identical. The Q6
MTP sidecar and target LM-head tensors are also unchanged. Thus the comparison
does not silently switch to a different learned draft model.

The SSD prefix-cache key incorporates model-directory identity, config/index
sizes and modification times. Keep revisions isolated during testing, and start
a fresh server after promotion; do not migrate live request state across them.
