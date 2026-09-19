# VQ quality-preserving optimization — 2026-09-19

Outcome: fix speculative QSA rollback without changing model weights, expert
selection, or codebook precision. None of this round's PP/decode prototypes
earned promotion. The existing 519.91 PP / 30.59 fixed-input target-only /
57.94 favorable-fixture MTP figures remain historical workload-specific
measurements, not new results from this change.

## What ships

- **Correct bounded QSA rollback.** The old checkpoint builder used the global
  2,048-token selection budget to decide whether to keep pools, even when the
  automatic decode budget was 512. It could discard valid pools while retaining
  only the last 64 raw rows. The next step then needed keys already discarded.
  The fix retains every completed causal pool in the accepted checkpoint and
  never trims raw keys beyond its pooling frontier. Before pooling starts, raw
  keys remain intact. This removes the need for an unbounded raw-window workaround.
- **Regression coverage.** Pure geometry sweeps cover acceptance lengths 0–5,
  windows 0/16/64, both selection budgets and pool boundaries. A real layer-3
  test compares bounded and unbounded state across BF16 and Q8 KV, including
  16 cases and 1,152 continuation steps. This is a layer-3 boundary test using
  repeated embedding input and 12 continuation steps alternating token-9419
  and token-11 input vectors, not a full-model long-context retrieval test.
  Ten previously failing HTTP HumanEval
  cases completed with byte-identical outputs to the unbounded workaround;
  nine passed the original tests and one still hit the 768-token limit.
- **Reliable full-program HumanEval scoring.** Preserve generated imports and
  helpers instead of stripping everything to a function body. Check that the
  selected Python interpreter can execute a known-good program inside the
  network-denied sandbox before scoring. Reject incomplete/duplicate task sets.
- **Reusable A/B tools.** MoE probes interleave control/candidate/ candidate/
  control, test exact output bits and report the separate timing arrays. A
  four-domain runtime probe measures warmed A/B/B/A requests and checks tokens
  against the original control. These are developer tools, not serving profiles.

The fix preserves the arithmetic policy already used by this VQ engine:
original FP16 codebooks for wide prefill, derived signed/affine 8-bit d8 tables
for decode, original d2 tables, exact top-10 routing. It does **not** establish
bitwise equivalence to unquantized Flash or between serial and batched GEMMs.

## Full quality qualification

The fresh, uninterrupted default-window native-MTP run completed **146/164
(89.02%)**, with **zero HTTP errors** and eight length-limited responses. All
164 raw outputs are byte-identical to the retained MTP generation artifact;
that older artifact mixed raw-window-64 successes with raw-window-0 retries
and is superseded by this uniform run for qualification.

| Original HumanEval tests | Passed | Aggregate decode | Per-request median [P10, P90] |
|---|---:|---:|---:|
| VQ target-only, fresh control | 147/164 (89.63%) | 25.35 tok/s | 25.48 [24.02, 28.42] |
| VQ native Q6 MTP, fixed rollback | 146/164 (89.02%) | 46.93 tok/s | 48.41 [42.87, 54.77] |
| 27B Q4, historical external control rescored | 150/164 (91.46%) | Not rerun | Not a paired timing measurement |

The one-task net MTP/control difference is not proof of statistical equivalence
or a general quality guarantee. Serial-only passes are tasks 10/91; MTP-only
is task 81. The MTP run emits 42,773 tokens and accepts 33,343/36,533 proposed
drafts (91.27%). Generation-phase time is 19.30% drafting, 79.09% verification
and 1.58% commit. These are measured phase timers, not hardware profiler
counters, and their accounting need not be disjoint at every operation.

The guarded session (full HumanEval plus the three-case IFBench regression)
peaks at **39.60 GiB physical footprint**, 34.21 GiB RSS, with 8.71 GiB minimum
reclaimable memory. It was stopped deliberately after testing; guard exit 130
records that shutdown, not a memory violation. The earlier serial-control
session peaked at 36.89 GiB. Cold first requests are included in the per-case
distributions; this is one serial sweep, not three independent benchmark runs.

IFBench keys 20/70/100, greedy non-thinking/max 512, also reproduce all three
retained native-MTP responses byte-for-byte and score **3/3 strict and loose**.
This is a small regression check,
not a new full IFBench score. Historical 17/30 and best-retained 18/30 results
must keep their respective configuration/protocol labels.

The target-only control has no HTTP errors and seven length-limited responses.
The 27B result is rescored with the same full-program grader but remains a
historical external-runner control, not a newly paired speed test.

Protocol: all 164 original OpenAI HumanEval tasks; EvalPlus-style non-thinking
chat prompts from `run_humaneval.py`; temperature zero, one sample per task,
768 output tokens, no custom stop strings, no prefix-cache hits. EvalPlus
0.3.1 sanitizes the **complete** generated program; the original HumanEval tests
run with a three-second sandbox timeout. These are **not HumanEval+ scores**.
The serial control uses MTP off and raw window 0. Final qualification uses the
automatic native Q6 MTP sidecar, Q4 drafter language head, adaptive depth up to
four and the corrected default raw window 64.

## Rejected speed probes

All component rows measure the **complete routed MoE**, including the overhead
introduced by the candidate, not just its changed kernel. Lower ms is better.
Median is the midpoint of the two central samples for even N; P10/P90 use the
sorted sample at `round((N-1)*p)`. Reported intervals describe samples, not
confidence intervals.

| Candidate | Scope; measured samples per arm | Control median ms [P10, P90] | Candidate median ms [P10, P90] | Decision |
|---|---|---:|---:|---|
| Next-group register lookahead | layer 2, 2,048 rows; 10 | 53.919 [53.360, 54.410] | 59.233 [58.652, 59.992] | 9.9% slower; removed |
| Double-buffer down GEMM, 2 producer SIMD groups | layer 2, 2,048 rows; 10 | 53.781 [53.364, 53.884] | 54.052 [53.927, 54.485] | No gain; removed |
| Same pipeline, 4 producer SIMD groups | layer 2, 2,048 rows; 10 | 54.335 [53.750, 54.567] | 55.038 [54.575, 55.684] | 1.3% slower; removed |
| Compact-codebook vector loads | layer 10, S=1; 120 | 0.3801 [0.3323, 0.6120] | 0.3818 [0.3342, 0.6189] | No gain; removed |
| Same vector loads | layer 10, S=5; 120 | 0.8157 [0.7706, 1.1231] | 0.8113 [0.7615, 1.2248] | 0.5% median movement, worse tail; removed |

Each candidate passed exact layer-bit comparisons. Two warmups per arm are
excluded. Input is the existing deterministic embedding fixture: token 9419 for
S=1, and `(9419 + row * 7919) % vocabulary` for multi-row probes. Component
tests have no generation, sampling, long KV context or MTP head. No extrapolated
whole-model PP or decode gain is claimed. PP probes peaked at about 1.59 GB
MLX allocation; small component footprints are **not** full-model memory.

An automatic **elapsed-ms-per-emitted-token MTP policy** was also tested. It
started at depth four, then probed depths three/two only when warranted, with
warmups and hysteresis. The same-process four-domain probe used max 256 tokens,
greedy non-thinking generation, cache off, an A/B warmup then A/B/B/A:

| Prompt | Control / candidate aggregate decode tok/s | Candidate output |
|---|---:|---|
| Merge sorted Python lists | 50.21 / 50.75 | Identical tokens |
| Local-server JSON | 46.31 / 46.81 | Identical tokens |
| Explain memory bandwidth | 34.38 / 36.42 | Identical tokens |
| Silent blue comet scene | 31.65 / 29.45 | Different trajectory |

Only two measured samples per arm/domain: directional evidence, not a broad
benchmark. The creative workload regressed 6.9%; the policy was removed.
Different tokens alone do not establish worse quality, but the candidate
failed the speed gate before a new quality campaign was justified. Peak guarded
footprint was 39.40 GiB. No speculative policy change ships.

## ANE: feasible research boundary, not a measured optimization

There is no ANE execution backend in this milestone. Core ML can assign model
subgraphs to ANE; a Metal kernel does not automatically run there. Apple's
documented palettization supports 1/2/3/4/6/8-bit indices and vector centroids,
not a direct import of our 14-bit/K16384 d8 format. This is a format/interface
mismatch, **not proof that all VQ execution on ANE is impossible**.

The bounded next experiment would be a fixed-shape dense projection or shared
expert subgraph, using the same effective weights and including GPU↔ANE
handoff, conversion, memory and synchronization in its timing. Confirm actual
ANE placement, then require task-quality and end-to-end gains before adding a
runtime dependency. Account for Core ML helper/driver allocations and system
memory pressure, not just the inference process's existing Metal footprint.
Moving only tiny routers or repeatedly crossing processors
at every layer is unlikely to pay; that is an architectural hypothesis, not an
ANE benchmark. MTP depends on target hidden states, so the two processors cannot
simply generate drafts and verify indefinitely in parallel.

Sources: [Apple's ANE transformer guidance](https://machinelearning.apple.com/research/neural-engine-transformers),
[Core ML palettization formats](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html),
[Core ML execution partitioning](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

## Reproduction and limits

Common platform: Apple M5 Pro, 18 CPU cores, 64 GiB unified memory, macOS 26.5,
MLX 0.32.2; AC power, no active thermal conditioning, no `pmset` thermal or
performance warning observed. Only one model process runs at a time, under
`memory_guard.py`. Guard limits are diagnostic ceilings, not observed memory.
Source base: `a8246e0406443982f2a2e17154b7ecd706afded4` plus this milestone.
Model: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, retained revision
`64b0fb0f98a552d91fb9abd5531d547b2e78c8a8`, unchanged bundled PLE and weights.

```sh
# QSA regression, real weights; MODEL is the local checkpoint directory.
python3 devtools/memory_guard.py --min-start-gib 10 --min-available-gib 5 \
  --max-rss-gib 4 --max-footprint-gib 4 -- \
  ./build-release/qwen38-self-attention-smoke "$MODEL" --qsa-rollback

# EvalPlus 0.3.1 must be installed in the selected interpreter.
python devtools/evaluate_humaneval.py --problems "$HUMANEVAL_JSONL_GZ" \
  --artifact "$GENERATION_JSON" --results "$SCORE_JSONL"
```

The A/B tools only test switches read **during execution**, not constructor-only
configuration. Confirm path engagement independently; an unknown switch could
produce a meaningless no-op comparison. Do not use the MoE probe's legacy
`warm_median_ms` as an A/B result: it mixes arms; use `ab.control_ms` and
`ab.candidate_ms`. Rejected prototypes are retained as evidence, not production
flags or extra worktrees.

No 128K VQ requalification, general chat 60 tok/s guarantee, ANE acceleration,
or achievement of 600 PP / 40 target-only tok/s is claimed. Keep quality-control
models until the outstanding long-context and broader-quality gates pass.

[Machine-readable evidence](vq-quality-preserving-2026-09-19.json) contains
model/tokenizer/template and artifact SHA-256 hashes, timing distributions,
all measured mixed-domain A/B samples, failed task IDs and guard results.
Local raw generation artifacts remain outside Git; no model weights, absolute
paths, credentials or large prompt/output dumps are committed.

Final verification: clean Release build; CTest **5/5**, including the real-model
tokenizer; scorer unit tests **6/6**, HumanEval runner tests **7/7**, IFBench CLI
tests **5/5**; final guarded real-weight QSA smoke **16/16 cases, 1,152/1,152
bit-exact continuations**; Python compilation and `git diff --check`. The linker
still emits the pre-existing duplicate `qwen38_core` library warning.
