# Prior-research ledger

This document is the optimization gate for `qwen38-flash.cpp`. Before starting
an experiment, its hypothesis must be checked against this ledger. A rejected
experiment is not repeated unless a materially different implementation or
measurement contract is stated first.

The historical measurements came from the same 64 GiB M5 Pro and the retained
Qwen3.8 Flash Next REAP-288 Q4 trunk plus Q8 REAP-288/L47 MTP sidecar. They are
reference evidence, not measurements of this C++ runtime.

## Established reference points

| Artifact | Controlled result | What it establishes | Limitation |
|---|---:|---|---|
| P124 stock split-K prefill | 759--766 tok/s at 256 tokens; 930--937 at 512; 1107--1113 at 1024 | The retained MLX/`mlx-serve` graph already proves PP 7xx locally; C++ must reproduce its layer-major whole-prompt path before inventing another prefill design | Q4 REAP-288 without MTP, one-token completion; the attempted direct-NAX replacement was rejected |
| P271 request-static MTP | 68.1--68.3 tok/s on short code, 39.5 prose, 42.2 JSON, and 46.75 at 7.3K context with 731.9 PP | Probe at depth 2, lock depth 3 only for high acceptance, and keep long prompts at depth 2; this is the retained MTP 6x-class reference | Workload-dependent; fixed depth 3 regressed prose, JSON, and long context |
| P73 serial | 35.028 decode tok/s; 479.357 prefill tok/s | The retained Zig/MLX target path can exceed the current C++ runtime | Fixed 186-token code prompt, 512 generated tokens, hot run |
| P73 Q8 MTP depth 2 | 59.866 decode tok/s; 463.316 prefill tok/s; 78.6% per-draft acceptance; 1.57 accepted drafts/round | Correct MTP can be strongly profitable on this machine | Workload-specific high-acceptance code result, not universal throughput |
| P60 corrected Q8 MTP | 44.36 decode tok/s versus 32.13 serial; about 60% acceptance | A broader workload can still benefit, but verifier cost and acceptance dominate | Different run and workload from P73 |
| P268 serial Q4 sweep | 34.359 all-control median; candidate 34.392 | Shape-specific QMV tuning alone does not move end-to-end throughput | One selected projection geometry |
| Current C++ runtime | about 31.0--31.3 warm decode tok/s; about 35.6 GiB peak | Correct full-model C++ baseline with bounded resident expert tier | No MTP, prefix cache, or complete long-context QSA yet |
| C++ fully batched-dispatch verifier | 59.24 ms for three rows versus 94.98 ms serial; 1.603x; exact token and rollback-continuation parity | Layer-major scheduling plus S-row Q4 routed-expert Metal dispatch improves verifier cost | Depth 2, empty origin, one interleaved A/B/B/A run; routed expert weights are still reread per row and round cost is above P73 |
| C++ multi-round Q8 MTP reference | Four consecutive tokens exactly matched serial across three rounds; two accepted drafts; 37.9 GiB peak | Transactional target rollback and committed-only head rebuild work across rounds | Tiny low-acceptance fixture, 256 MiB MLX cache, 12 resident expert layers; no throughput claim |
| C++ memory-bounded Q8 MTP timing | 13.82 tok/s for five emitted tokens versus 8.56 tok/s four-token serial timing; 1.61x ratio; 37.6 GiB peak | The complete MTP path is profitable even on a low-acceptance cold fixture | Not comparable to the 31 tok/s warm serial baseline: 12 resident layers, 256 MiB cache, per-round cache clearing, tiny sample |
| Reproduced P108 Q4/g64 MTP pack | Exact historical sidecar SHA-256 `9ac09cd2...10672a`; 848,139,296-byte file; 19 Q4 projections, 13 BF16 sensitive tensors; 0.79 GiB physical delta | The C++ runtime can evaluate the original P108 candidate without double-quantization or changing the Q4 target trunk | Proposal-only artifact; target verification still defines output quality |
| C++ P185 Q8-vs-Q4 proposal A/B | 78/98 top-1 agreement; mean/min logit cosine 0.9492/0.7852; adjacent target-token matches 27 Q8 versus 29 Q4; 3.6 GiB peak | Q4 cuts head storage while preserving comparable one-step proposal utility on this code trace | One 98-transition code trace, not a mixed corpus or depth-2 acceptance/throughput promotion |
| C++ Q8/Q4/Q8 full-round pilot | 13.86 / 16.25 / 14.09 emitted tok/s; all runs used three rounds, accepted two drafts, emitted the same five tokens, and matched the four-token serial oracle; Q4 peak 36.2 GiB versus Q8 36.9--37.4 GiB | Q4 was 16.3% faster than the mean of its two Q8 controls and reduced peak RSS without changing acceptance or committed tokens in this cell | Four-token cold, aggressively memory-bounded fixture with 12 resident layers, 256 MiB cache, and per-round cache clearing; requires warm mixed-corpus confirmation |
| Native HTTP Q4 MTP low-acceptance control | Auto MTP and serial emitted byte-identical eight-token text; MTP saw 0/4 accepted across two rounds, fell back, and reached 22.91 tok/s versus 29.27 serial; peaks 36.6 versus 34.9 GiB | The production API wiring is correct, and acceptance-only fallback works, but two losing rounds are too expensive | One 15-token code prompt; cold prompt/decode timing, not a throughput promotion |
| C++ layer-major prompt integration | Five chat prompts matched the retained oMLX first token exactly, including the 186-token P73 prompt (`I`, id 40). On that prompt, the internal batch path was 3.09x faster than the old serial path; the warmed HTTP path reached 99.2 prefill tok/s and peaked at 35.5 GiB | The old serial C++ prefill was both slower and a false numerical oracle. Batched target numerics are now the production path | Chunk size 64, Q4 REAP-288, no prefix reuse; still well below P73's 479.357 tok/s |
| C++ batched-prefill Q4 MTP check | The warmed 186-token P73 prompt accepted 18/28 drafts and emitted 32 tokens at 27.12 tok/s; prefill was 80.3 tok/s; peak 35.5 GiB | Batched target prefill correctly primes complete MTP state, but serial MTP-head prompt priming and current round cost remain unprofitable | One code prompt, depth 2, short 32-token generation; correctness check, not a speed promotion |
| C++ exact complete-state prefix cache | A repeated P73 request reused 185/186 prompt tokens with byte-identical output. No-MTP prompt time fell from 1955.67 ms after clear to 0.000042 ms on hit; MTP preserved 18/28 acceptance and identical 32-token output with 0 ms reported prompt time. Peaks were 34.5 GiB serial and 35.2 GiB MTP | One snapshot can safely reuse target KV, GDN, PLE, QSA, prior target stream, and the complete MTP head state together | One-entry RAM tier, capped at 8192 tokens by default; no partial rollback checkpoint or SSD tier yet |
| C++ batched verifier final head | A/B/A in guarded processes reduced the exact 3-row verifier by 3--5% to about 57 ms, versus about 94.5 ms serial; token parity held. Profiling attributed 42.59 ms to 36 GDN layers, 11.60 ms to 12 full-attention layers, and only 1.50 ms to the batched final head | The final mixer and LM head should consume `[1,S,H]` once, but 95% of remaining verifier cost is now the 48-layer trunk | A warmed 64-token HTTP run still reached just 31.98 tok/s with 22/38 accepted and one fallback; trunk/round cost remains dominant |
| C++ compiled-vs-batched numerical audit | On the 64-token P73 continuation, eager serial and the fast S=3 verifier emitted byte-identical text; compiled S=1 serial emitted a different valid greedy continuation. Warm decode was 29.87 / 31.98 / 32.07 tok/s for eager serial / fast MTP / compiled serial | The production MTP lifecycle is not the source of the observed text difference. MLX graph shape and whole-layer fusion change borderline argmax decisions, so a compiled S=1 run is not a valid bitwise oracle for an eager S=3 graph | Quality promotion still requires corpus-level scoring; exact-token parity must compare like-for-like numerical execution modes |
| C++ opt-in fused-GDN + sorted-MoE prefill | 185 prefill rows in 608.8 ms, about 304 tok/s and 9.10x over its row-serial diagnostic; 34.8 GiB peak; batched token matched that diagnostic | Cached Metal configurations remove repeated FFI setup and the recovered whole-prompt strategy is directionally correct | Still far below P124 and differs from the retained production token; both kernels remain opt-in |
| C++ read-only fused HC A/B/A | Control 29.84 tok/s, candidate 33.58, reverse control 32.12; candidate was +8.53% versus pooled controls and repeated at 33.95 tok/s. All four 16-token sequences were identical; peak was 33.7--34.8 GiB | The mlx-serve HC read design transfers to the independent C++/MLX engine and removes real decode overhead without extra resident memory | Synthetic position-0 decode trace, 14 steady steps per arm; still opt-in pending mixed-prompt quality and HTTP production cohorts |
| C++ fused GDN norm/gate epilogue | Candidate 33.89 and 34.00 tok/s versus counterposed read-only-HC control 33.80; all 16 tokens matched and the random BF16 kernel oracle stayed within 0.015625 max-abs | The epilogue is numerically bounded and ready to combine with GDN prework | Only about 0.3--0.6% alone, below the promotion threshold; remains opt-in |
| C++ fused GDN prework + epilogue | Candidate 34.96 and 34.79 tok/s versus counterposed HC-only control 33.66, about +3.4--3.9%; all 16 tokens matched. Layer-0 two-step oracle: first output max-abs 0.000488281, second output/convolution/recurrent max-abs exactly 0 | The recovered mlx-serve GDN fusion transfers cleanly and supplies the missing material gain behind the epilogue | Synthetic position-0 decode trace; remains opt-in until mixed-prompt HTTP cohorts |
| C++ P271 request-static policy port | Unit fixtures reproduce the eight-round depth-2 probe, 10-accept promotion threshold, 2,048-token prompt boundary, and permanent demotion after a 12-round window below 50% acceptance. Native API reports final depth, promotions, and demotions | The later 68 tok/s reference policy is now part of the real generation loop rather than an external benchmark note | End-to-end guarded mixed-prompt throughput is not yet measured; this is a lifecycle/correctness milestone, not a speed claim |

P73's 59.866 tok/s is explained by `(1 + 1.57) / 43.95 ms`, approximately
58.5 tok/s. It is not evidence of a universal 59 tok/s target path. The serial
and MTP acceptance gates remain the mixed-workload requirements in
`benchmark-contract.md`.

## Adopt

These findings are correctness requirements or already-supported performance
directions. New code should build on them.

1. **Qwen4 delta-norm decoding.** Every checkpoint tensor marked as a delta
   norm is used as `1 + weight`. This includes trunk HC norms, attention and
   indexer q/k norms, MTP pre-FC norms, and the MTP layer's matching norms.
   Historical no-fold captures are not valid oracles for the retained model.
2. **The MTP hidden seam is 10,240-wide.** For trunk row `r`, the head consumes
   the four-stream pre-mixer residual `H[r]` together with `token[r+1]`, at
   query position `r+1`, and predicts `token[r+2]`. The collapsed 2,560-wide
   logits input is not interchangeable.
3. **Only valid adjacent prompt pairs prime MTP.** A cold prompt never invents
   a zero-hidden row for its first token. Pending hidden state is cleared on a
   new request, copied on parent-to-child cloning, and accepted only for a
   contiguous next chunk.
4. **MTP history is committed-only.** Draft-head speculative rows are discarded
   after verification. Accepted history is rebuilt from target-captured
   pre-mixer streams, never from draft-head streams.
5. **Verification is one batched target pass.** The target input is
   `[t1, d0, ..., d(m-1)]`; acceptance is the longest matching target-argmax
   prefix for greedy generation. The final row supplies the bonus/correction
   distribution.
6. **Rollback covers every state family.** A partial accept restores target KV,
   GDN recurrent and convolution state, PLE window/token history, QSA raw and
   pooled indexer state, offsets, and the accepted target hidden stream.
   The MTP head similarly owns KV plus QSA raw/pooled state and cannot use a
   KV-only snapshot.
7. **Use the retained Q8 REAP-288/L47 sidecar first.** Its production behavior
   and acceptance are established. Model/head re-quantization is a separate,
   quality-gated project and is not a prerequisite for runtime parity.
8. **Use adaptive profitability controls.** Depth 2 is a strong seed for this
   pack. The runtime must fall back to serial when observed acceptance and round
   cost make MTP unprofitable, rather than claiming one prompt's best depth as a
   universal default.
9. **Keep full-model tests serialized and guarded.** No optimization evidence
   is accepted from an unguarded run on the 64 GiB machine.
10. **Use layer-major chunked target prefill.** The retained production runtime
    and Transformers agree on the 186-token template and first generated token;
    the former C++ token-at-a-time path does not. Added tokens such as `<think>`
    must remain atomic even when their tokenizer metadata says `special=false`.
11. **Cache complete state or do not cache.** The first C++ cache intentionally
    keeps one exact target/MTP snapshot rather than multiplying hybrid entries.
    Its public cached-token count and cache-clear behavior are part of the API
    contract; an MTP hit must preserve both committed tokens and acceptance.
12. **Treat P124/P271 as the implementation baseline.** Prefill must reproduce
    the stock split-K whole-prompt graph, while MTP starts with the eight-round
    depth-2 probe and request-static depth-3 promotion/demotion policy. Do not
    substitute the older P73 numbers for this later production lineage.

## Reject or do not repeat

| Direction | Historical evidence | Decision |
|---|---|---|
| Per-shape affine-Q4 autoregressive QMV retuning | P268's best isolated kernel was 1.082x, but full-engine ON was only 0.10% above pooled controls and slower than reverse controls | Do not continue tile/threadgroup/half-vector sweeps without a fused multi-consumer design |
| Grouped S=2..3 verify gate/up | Isolated 1.06x; full model 44.446 versus 44.897 tok/s, with worse measured acceptance | Rejected as a production optimization |
| Grouped verify down/reduce prototype | Isolated Q4 gain 1.05x; Q8 generalization produced incorrect values | Do not port as-is; require a new exact Q8 design and end-to-end cost case |
| Routed-expert union gate/up prototype | The guarded depth-2 full-model parity run reached 38.3 GiB RSS before completion and was terminated at the 38 GiB cap | Rejected for the 64 GiB product profile; do not raise the safety cap to promote it |
| Interleaving verifier rows by top-k slot | Exact 3-row A/B/A was 56.94 ms row-major versus 57.01 ms interleaved (0.999x) | Rejected; argpartition slots do not provide useful cross-row expert locality |
| Compiling three serial rows inside one fixed S=3 layer closure | Exact layer-major control was 78.368 ms and the compiled closure was 78.344 ms (1.0003x), with token parity and 35.4 GiB peak RSS | Rejected; enclosing three sequential decode graphs in one closure does not remove their weight reads or synchronization cost |
| Same-layer attention-write folded into MLP HC read | Two guarded runs were 33.65 and 33.34 tok/s versus a counterposed read-only HC control at 33.93 tok/s; all 16 tokens matched | Rejected and removed. Writing the complete 10,240-wide updated stream as a second kernel output costs more than the saved dispatch; only revisit as a truly cross-layer deferred value with explicit barriers |
| Q4 MTP depth 3/4 on P73 | Warm 64-token results: depth 2 = 31.98 tok/s (22/38 accepted), depth 3 = 30.96 (24/51), depth 4 = 29.92 (5/20 and early fallback). Depth 4 also diverged from the depth-2/3 greedy text | Keep depth 2 as the safe default; deeper drafting does not repay extra head/verifier rows on this runtime and 5-row numerical parity is not promoted |
| Qwen4 Fused-QKV flag used in P73 | The Qwen4 geometry declined the path, so apparent timing differences were environmental | No performance claim and no blind rerun |
| Decode async ladder used in P73 | The Qwen4 layer loop never engaged it | No performance claim and no blind rerun |
| KV-only MTP prefix snapshots | QSA raw/pooled state and offsets become stale | Correctness-invalid |
| Serial verification of each draft | Re-reads the target for every proposal | Architecturally rejected; verification must stay batched |
| Per-row evaluation inside a layer-major verifier | 125.12 ms versus 94.62 ms serial, or 0.756x | Rejected; S synchronization barriers per layer erase locality gains |
| Naive concatenated proposal tree | GDN, PLE, QSA, KV, and MTP-head state are branch-dependent | Not a small optimization; only revisit as a true branch-local runtime |
| Full target execution on ANE | Unsupported operators, conversion overhead, and memory movement dominate | Not on the critical path; ANE may be reconsidered only for a measured isolated subgraph |
| More aggressive model quantization without a quality gate | It changes the requested quality target and does not remove dispatch/state costs | Not a substitute for runtime optimization |

## High-upside work not yet ported

The implementation order is deliberately narrow:

1. **Exact MTP model and lifecycle.** Load the Q8 sidecar, apply delta norms,
   expose the 10,240-wide target stream, implement valid prompt pairing, and
   establish one-step/two-step head parity before adding speculation.
2. **Transactional batched verifier.** Add S=2..4 target capture and complete
   rollback for KV, GDN/conv, PLE, QSA, offsets, and committed hidden history.
   Greedy MTP must reproduce serial token IDs across zero, partial, and full
   acceptance fixtures.
3. **Compile-stable fixed-capacity state.** Replace growing per-step QSA/GDN
   graph shapes with fixed-capacity or paged state so the verified MTP pass can
   reuse compiled graphs and avoid allocator/dispatch churn.
4. **Bounded multi-entry/SSD prefix tier.** One exact complete-state RAM snapshot
   is integrated. Add byte accounting before expanding entry count, then a
   durable SSD tier for long prompts. MTP restoration must continue to include
   complete QSA state or rebuild from a freshly forwarded tail.
5. **Verifier-wide fusion only after profiling.** The remaining plausible
   kernel opportunity is a fused, exact S=2..4 verifier path that removes
    model-byte movement, intermediates, and dispatches across multiple
    consumers. Improving one projection in isolation is explicitly excluded.
6. **Complete fused HC promotion.** The read-only port has an A/B/A win. Add
   per-layer numerical oracles, mixed-prompt decode cohorts, then fold the
   preceding HC write into the next read with explicit PLE/capture/final-head
   flush barriers.
7. **Branch-local shallow tree only after linear MTP is profitable.** A B=2,
   depth-2 shadow implementation is justified only if measured extra verify-row
   cost is lower than its acceptance benefit and all recurrent/QSA state is
   branch-local.

## Promotion gates

An optimization is promoted only when all of the following hold:

- the intended path emits an engagement marker and its kill switch proves a
  clean control;
- deterministic output/token parity passes for the affected path;
- MTP tests cover zero, partial, and full acceptance plus rollback continuation;
- full-model A/B/A uses the benchmark contract and memory guard;
- the median improves beyond run-to-run noise without unacceptable p10,
  prefill, TTFT, memory, or quality regression;
- the result states whether it is a workload-specific result or a mixed-workload
  product claim.

## Provenance and licensing

The conclusions above were derived from the retained P60, P73, P169, P255,
P268, and oMLX-MTP lifecycle audit artifacts. Reference implementations remain
outside this repository. Code is reimplemented from the documented model and
state contracts unless a compatible license is reviewed; adapted code keeps
source attribution in the file and in `NOTICE`.
