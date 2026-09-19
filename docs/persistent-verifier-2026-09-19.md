# Persistent VQ verifier: rejected; native QSA fix retained

The user selected persistent MTP verification, without lowering drafter
precision. A complete 48-layer, layer-major target-verifier prototype was
implemented and measured. **It did not beat the production verifier and was
removed before MTP round/server integration.** This is not a completed native
MTP backend, a new quality result, or a production speed improvement.

## Same-session target-verifier results

Milliseconds per complete target verification; lower is better. Each cell has
12 measured observations, reported as median [P10, P90].

| Width | Existing MLX layer-major verifier | Persistent prototype | Decision |
|---|---:|---:|---|
| S=2 | 41.538 [41.000, 42.187] | 41.382 [40.836, 41.725] | Noise-equivalent, before native commit |
| S=5 | 62.843 [62.155, 63.719] | 84.567 [84.321, 84.898] | 34.6% higher latency; reject |

The timed native region includes origin checkpoint capture, embedding, PLE,
all target layers, the head and GPU top-2. State import is outside the timer;
full-accept commit is separately recorded. MLX timing includes target forward
and evaluated argmax rows, not a complete learned-MTP round. Native/MLX output
IDs differ on one of the five rows, so this is **not numeric equivalence** or
a quality-controlled product-speed comparison. Native serial is the exactness
oracle for the new scheduling/state changes.

Even five accepted/emitted tokens per 84.567 ms would be only 59.1 tok/s
before drafting and commit **on this fixed-input fixture**. This makes broader
MTP integration unjustified for this candidate; it is not a hardware ceiling.

The shared-weight A/B process peaked at **38.086 GiB footprint**, 24.852 GiB
RSS, with 9.416 GiB minimum reclaimable memory. This includes the target and
verifier, not an actively drafting MTP sidecar or full HTTP workload.

## What was actually implemented

- Reusable five-row activation arenas and 18 row-lifted Metal kernels. VQ
  codebooks, top-10 routing and each scalar kernel's reduction order remained
  unchanged; no FP32 reconstructed-codebook matrix representation was used.
- Layer-major batched HC, dense projections, router, routed/shared VQ MoE,
  final mixer and LM head. Stateful GDN, attention and PLE obeyed causal order.
- GPU-resident per-prefix GDN/convolution/PLE checkpoints; append-only hot KV
  and pooled QSA with frontier/pending-tail rollback. No full KV clone per row.
- GPU top-2 results and accepted-prefix commit, including prefix-zero abort.
  Allocations were reused between rounds; the trial retained full recurrent
  checkpoints, not a new compact transition-replay representation.
- Follow-up: adjacent-row interleaved dispatch and causal multi-row GDN
  prework/recurrence, writing checkpoints inside the kernels instead of separate
  copies. S=5 remained about 85 ms. Moving native waits alone did not solve the
  device-work cost. No per-stage hardware-counter attribution is claimed.

Correctness probes covered widths 1/2/5 at synthetic context states of 3,
511, and 2,051 tokens (the last with 2,048 Q8-cold tokens). Every prefix from
zero through full acceptance was compared with native serial:

- **33** exact checkpoint/abort/commit cases;
- **114** row comparisons of token, alternate token, logit and hidden stream;
- **33** exact next-step continuations, including full GDN, KV, QSA and PLE state.

These passed after fixing the QSA issue below. Synthetic tests are not needle
retrieval, IFBench, or HumanEval. The exploratory mmap-only process footprint
omits much of its weight residency and must not be presented as model RAM.

## Retained correctness fix

The pre-existing native selector wrote its first top-128 reduction to a
temporary ID buffer. When there were **129..256 candidate blocks**, the first
pass was also the last pass, so the merge loop never copied IDs to `selected`.
Attention could read stale indices from a previous step/request. With the
current four-token pooling this covers resulting context lengths 516..1,027.

The first pass now writes directly to the final ID buffer when no merge is
required. The dispatch helper is shared by production and a Metal regression
test. Poisoned-output tests compare with CPU top-128 at 129/255/256/257/511/
512/513/769/65,536 candidates. Restoring the old destination makes the test
fail at 129, confirming that it detects the original bug.

The retained full-model `qsa-reuse` probe also passes at starting contexts 515
and 1,023: importing the same state after different prior selector histories
produces identical token/top-2/logit, hidden stream and complete next state.
Run it under the memory guard:

```sh
qwen38-persistent-state-probe MODEL qsa-reuse
```

This affects the **experimental persistent backend**, not the default MLX
verifier. It does not revise historical production IFBench/HumanEval scores.

## Protocol and limits

- Model: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, revision
  `64b0fb0f98a552d91fb9abd5531d547b2e78c8a8`; packed-14 d8 / uint8 d2 routed
  VQ, existing INT8/U8 compact d8 codebooks, Q8 dense, exact top-10, checkpoint PLE.
- Apple M5 Pro, 18 CPU cores, 64 GiB unified memory, macOS 26.5, MLX 0.32.2.
  AC power, no active thermal/frequency controls; no recorded `pmset` warning.
- Prompt: exactly 127 repetitions of token 9419. Target inputs:
  `[9419, 11, 271, 40, 13]`, truncated to the tested width. Greedy argmax;
  no thinking, sampled generation, prefix-cache hit, or learned draft creation.
- A/B/B/A blocks of eight calls; the first two calls of each block are warmups.
  Earlier fine-grained alternating probes saw roughly 450 ms after backend
  switches in both arms. Their cause was not isolated; warm steady measurements
  do not erase that transition cost or qualify multi-round serving.
- Guard: 40 GiB footprint, 38 GiB RSS, 6 GiB minimum available, one GPU job at a
  time. No extra worktrees, model copies, or precision changes.
- Full learned-MTP acceptance, HTTP cancellation, fresh IFBench/HumanEval,
  PP and long-context throughput were **not** tested for the rejected candidate.
  The task's end-to-end native-MTP speed objective remains open.

Raw measured samples, warmups and guard summary:
[JSON evidence](results/persistent-verifier-2026-09-19.json).

Production runtime defaults remain unchanged. Keep the QSA fix/tests and this
written record; do not retain the slower verifier, enlarged scratch arenas,
or source-generation machinery in the serving engine.

## Milestone verification

- Clean Release build passed; only the pre-existing duplicate-library warning.
- CTest **7/7 passed**, including the real-model tokenizer fixture and the new
  Metal selector regression. The model-free initial run skipped the tokenizer;
  the final run supplied the existing local tokenizer assets.
- Negative control failed at 129 candidates, as intended; corrected selector
  passed all nine CPU-oracle sizes twice and rejected invalid padding.
- Real VQ `qsa-reuse` passed at both one-group boundaries under the unchanged
  guard. No new IFBench/HumanEval score or end-to-end MTP speed is claimed.
