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
| Historical P124 stock split-K prefill | 759--766 tok/s at 256 tokens; 930--937 at 512; 1107--1113 at 1024 | This originally motivated recovering the layer-major whole-prompt path | Historical worktree/result only: a September 1 same-model rerun on retained mlx-serve 26.8.10 reached 564.8 warm PP, so the old 759--1113 figures are not a currently reproducible product baseline |
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
| C++ whole-prompt GDN-only prefill | P73's 185 forwarded rows fell from 1726 ms at chunk 64 to 1158 ms in one 186-row chunk, about 160 tok/s and 1.49x, with the retained first token `I` (id 40) and 34.0 GiB peak. A five-prompt warm cohort improved from 92.7--106.6 to 145.2--159.3 PP at 33.1 GiB peak | The oMLX-derived recurrence can safely remove repeated trunk reads for explicit wider-chunk experiments without requiring sorted-MoE | Four of five 32-token continuations differed from the chunk-64 numerical mode; chunk 64 remains the production default pending a quality-scored cohort |
| C++ causal-SDPA fast prefill | Adding one causal SDPA call per full-attention layer to GDN+grouped prefill reduced P73 from 608 to 414 ms (about 447 PP) and restored first-token id 40. At 256 tokens it reached 510 ms/about 502 PP with serial first-token parity; at 512 it reached 953 ms/about 537 PP with id-40 parity and 34.5 GiB peak. The five-prompt warm HTTP cohort reached 186.6--221.6 PP at short 30--77-token widths; two continuations were byte-identical and the other three remained coherent | The former per-token full-attention loop was a major architecture gap from P124. MLX causal mode correctly offsets queries against prefix K/V, proven by a synthetic wrapper test and the two-chunk 512-token run | Still opt-in: 1024 tokens at chunk 256 crossed the 8 GiB availability guard; chunk 128 completed at 36.2 GiB but only 416 PP. Mixed-prompt quality scoring and temporary-graph release remain required before changing the default |
| C++ shared prefill RoPE tables | Reusing one cos/sin table across Q and K in all 12 full-attention layers reduced the fixed 256-token case from 509.9 ms to 504.9 and 503.4 ms (about 507--509 PP), and the 512-token case from 952.8 to 946.3 ms (about 541 PP). Both fixed prompts retained serial first-token parity; peak stayed at 35.6--35.7 GiB | The table depends only on chunk position, width, rotary dimension, theta, and dtype, so reconstructing it twice per layer was redundant | About a 1% controlled gain, not a path to the remaining 600 PP gap by itself |
| C++ sparse prefill barriers | Evaluating every eight layers reduced the fixed 256-token fast-prefill case from 503--504 ms to 485.4 ms (about 527 PP), and the two-chunk 512-token case from 946.3 to 915.3 ms (about 559 PP). At 1024 tokens with chunk 128 it improved 2462.6 to 2378.1 ms (about 431 PP). All fixed first tokens retained serial parity; the 1024-token run peaked at 34.7 GiB with 9.4 GiB minimum available. Strides 16 and 48 were no faster than 8 | The retained runtime's MoE cadence transfers once causal SDPA makes each layer's graph compact enough; draining every layer was unnecessary synchronization | Opt-in through `QWEN38_PREFILL_BARRIER_STRIDE=8`; the default remains 1 while wider chunk-256 long-context graphs remain above the memory guard |
| Rejected blocked-GDN port at chunk 256 | The oMLX blocked-sequence recurrence increased the fixed fast-prefill case from 485.4 to 569.3 ms, raised peak RSS to 37.2 GiB, and changed the first token | Its documented roughly 2x layer win is at 16K rows, where staging Q/K/V amortizes threadgroup barriers and redundant reads; the 128--256-row chunks required by this 64 GiB machine are below that payoff regime | Implementation removed; reconsider only if temporary-graph reductions make multi-thousand-row chunks safe |
| Current retained mlx-serve warm control | mlx-serve 26.8.10 on the identical 256-token prompt/model reached 252.4 PP cold and 564.8 PP on the second full prefill with prefix cache disabled; output was `我已`. The C++ fast smoke reached about 527 PP but warms pages with its serial reference first | The live reproducible prefill gap is about 7%, not the 45% implied by historical P124 | Same 256-token chat prompt, chunk 256, one generated token, no MTP/PLD, 2K context; server wall time also contains decode and HTTP overhead |
| Rejected exact fused SwiGLU prefill | Porting the retained BF16 sigmoid-table fusion preserved the fixed first token but increased C++ fast prefill from about 485 to 557 ms and raised RSS to 36.7 GiB | A fusion profitable in another full graph can lose when isolated; MLX's scheduled elementwise chain is better for this 1.63M-element prompt tensor | Implementation removed; do not retry without an in-graph phase profile showing elementwise dispatches on the critical path |
| Rejected fused router at prefill width | Porting the retained exact-reduction softmax/top-k router increased the 256-token case to 626 ms and changed the first token | One-threadgroup-per-row fusion removes launch latency at decode width, but loses to MLX's batch softmax/partition at 255 forwarded rows | Implementation removed; retain the device batch routing chain for prefill |
| C++ guarded 512-row prefill | Processing the fixed 512-token prompt as one chunk reduced prefill from 915.3 ms/about 559 PP at chunk 256 to 763.5 and 762.5 ms/about 671--672 PP. The first token retained serial parity; peaks were 35.1--35.5 GiB with at least 9.5 GiB available. A server configured for chunk 512 then accepted a 1013-token raw prompt, automatically used chunk 128, and completed the warm prefill in 2297.5 ms/about 441 PP at 36.2 GiB peak | Wider QMM/SDPA batches finally amortize the per-layer graph and cross the 600 PP target without a new numerical mode | Product policy uses chunk 512 only for prompts up to 512 tokens; longer prompts automatically fall back to the validated chunk-128 path to avoid the known long-context temporary-graph pressure |
| C++ read-only fused HC A/B/A | Control 29.84 tok/s, candidate 33.58, reverse control 32.12; candidate was +8.53% versus pooled controls and repeated at 33.95 tok/s. All four 16-token sequences were identical; peak was 33.7--34.8 GiB | The mlx-serve HC read design transfers to the independent C++/MLX engine and removes real decode overhead without extra resident memory | Synthetic position-0 decode trace, 14 steady steps per arm; still opt-in pending mixed-prompt quality and HTTP production cohorts |
| C++ fused HC dense injection | Completing the MIT mlx-serve HC read by materializing the four injection rows as a folded BF16 `[10240,4]` matrix and reducing them inside the normalize/down kernels reached 41.68 and 42.60 tok/s on repeated guarded 64-step runs versus a 33.33 tok/s read-only-HC control. Peak RSS was 34.9--35.0 GiB. A one-row fixture differed by one BF16 ULP in one of four gates; the complete greedy trajectory matched the control through a long common prefix before taking another numerical path. A real Q4 MTP round retained serial parity, accepted 2 drafts in 3 rounds, and peaked at 34.9 GiB | The previously omitted injection half of the known three-dispatch HC design removes enough decode dispatch/graph work for a repeatable roughly 25--28% gain without material resident growth | Q4 REAP-288 trunk, M5 Pro 64 GiB, fixed 64-step token-9419 synthetic trajectory, greedy S=1, no MTP for throughput; opt-in `QWEN38_HC_FUSED_INJECTION=1` pending mixed-corpus quality scoring and production HTTP cohorts |
| C++ fused GDN norm/gate epilogue | Candidate 33.89 and 34.00 tok/s versus counterposed read-only-HC control 33.80; all 16 tokens matched and the random BF16 kernel oracle stayed within 0.015625 max-abs | The epilogue is numerically bounded and ready to combine with GDN prework | Only about 0.3--0.6% alone, below the promotion threshold; remains opt-in |
| C++ fused GDN prework + epilogue | Candidate 34.96 and 34.79 tok/s versus counterposed HC-only control 33.66, about +3.4--3.9%; all 16 tokens matched. Layer-0 two-step oracle: first output max-abs 0.000488281, second output/convolution/recurrent max-abs exactly 0 | The recovered mlx-serve GDN fusion transfers cleanly and supplies the missing material gain behind the epilogue | Synthetic position-0 decode trace; remains opt-in until mixed-prompt HTTP cohorts |
| C++ P271 request-static policy port | Unit fixtures reproduce the eight-round depth-2 probe, 10-accept promotion threshold, 2,048-token prompt boundary, and permanent demotion after a 12-round window below 50% acceptance. On the frozen P73 text (175 raw completion tokens here), guarded warm MTP reached 44.48 tok/s versus 34.23 serial, +30.0%, with 156/190 drafts accepted and one depth-3 promotion; peak was 33.7 GiB | The later policy is now part of the real generation loop and produces a material gain on its intended high-acceptance workload | Still below the old engine's 68 tok/s because proposal plus verifier cost remains high. On a general Python prompt, 10/22 acceptance reached 34.20 tok/s versus 34.63 serial, so the zero-accept early fallback remains necessary |
| C++ 16-layer verifier barrier groups | S=3 verifier fell from 56.10 ms at one barrier per layer to 47.29/47.54 ms at stride 16, a repeatable 15.3--15.7% latency reduction with exact token parity. P73 warm MTP rose 40.62→44.48 tok/s; a low-acceptance Python prompt rose 26.89→34.20. Peak RSS stayed 33.7--34.5 GiB | Owning the full pass removes synchronization overhead that individual kernels cannot; stride 16 is the promoted verifier default | Stride 48 regressed slightly to 47.53 ms, and per-layer profiler attribution is not meaningful when multiple layers share one barrier |
| C++ BF16-tree fused GDN verifier recurrence | Initialized-state S=3 verifier improved 47.18→39.90 ms (15.4%) with greedy and partial-accept continuation parity at identical 34.3 GiB peak. Warm depth-2 HTTP improved 40.90→42.18 tok/s; 256-token output SHA-256 was identical, with 41/56 drafts accepted | MLX BF16 `sum_axis` uses BF16 accumulation over four consecutive values per virtual thread before `simd_sum`; reproducing that tree makes a single-dispatch recurrence acceptance-safe | Q4 REAP-288 trunk and Q4 L47 sidecar, M5 Pro 64 GiB; opt-in `QWEN38_GDN_METAL_VERIFY_BF16_SUM=1` pending mixed-corpus promotion |
| C++ 16-layer resident-expert tier | In a control/candidate/superset/reverse-control HTTP sweep, `12:28` versus `12:24` reduced the uncached 57-token prompt from 3379 to 3114 ms and the 32-token cold generation from 1484 to 1411 ms; cached 256-token generation moved from 43.31 to 43.53 tok/s. Sampled RSS rose from 33.9 to 34.8 GiB. `12:32` reached 2824 ms prompt time but only 43.34 tok/s warm | Four additional resident layers buy useful first-request latency for less than 1 GiB while preserving the identical 20/22 and 39/56 acceptance paths | Promoted as the 64 GiB MTP default; explicit `QWEN38_RESIDENT_EXPERT_RANGE` still overrides it, and `12:24` remains the lower-memory choice |
| C++ rolling-economic MTP fallback | The legacy two-empty-round rule disabled MTP despite profitable cumulative histories. On the fixed B31 code/creative/JSON suite, a 16-round window requiring at least one accepted token per round improved 37.40→39.54, 38.23→40.16, and 38.65→40.80 tok/s (+5.0--5.7%); all three later fell back when the local window became losing. A separate cached 256-token completion was neutral within counterposed noise (43.79 versus pooled legacy controls at 43.85). Peak sampled RSS after the suite was 35.9 GiB | Empty streaks are not an economic signal once a request has established useful acceptance. A rolling local window preserves profitable phases and still bounds losses | Promoted by default with `QWEN38_ECONOMIC_MTP_FALLBACK=0` as exact legacy rollback. Continuing S=3 longer can select a different valid greedy path than switching to S=1, so corpus-level quality scoring remains a release gate |
| C++ hybrid history/MTP drafting | An exact request-local 2--5-token suffix cache replaced learned proposals only while the MTP state remained synchronized. On a counterposed warm code/creative/JSON cohort it reached 43.53/44.48/44.16 tok/s versus 43.18/43.16/43.48 for learned MTP alone, about +0.8%/+3.1%/+1.6%. History rounds accepted 18/27, 25/36, and 28/48 proposed tokens and avoided roughly 127--225 ms of learned draft work per request. The cache adds only token IDs and hash indices | Repeated syntax and discourse fragments provide near-free proposals; exact target verification preserves model authority, while a miss falls through to the Q4 head | Promoted by default with `QWEN38_HISTORY_DRAFT=0` rollback. Extending history verification after learned-MTP fallback was rejected: even an independent rolling guard produced only 41.79/43.06/43.26 tok/s. `12:32` also remained rejected despite slightly faster prefill: about 35.9 GiB RSS and 42.97/43.83/43.65 tok/s versus the faster `12:28` decode tier |
| Rejected cross-layer static-hot expert mlock | Partially locking the 16 most frequent layer-23 experts in every non-`12:28` layer added about 1.4 GiB of locked expert pages. Across candidate/control/reverse-candidate warm serial code/creative/JSON runs, pooled candidate throughput was 42.49/42.21/42.38 tok/s versus control 42.17/42.23/42.26: roughly +0.75%/-0.06%/+0.29%, only +0.33% overall | Expert popularity from one layer does not transfer strongly enough across all 48 routers, and the already-resident middle layers absorb the known layer-23 hot set | Implementation removed. The gain is noise-sized and does not justify extra locked memory or a calibration-specific expert list; retry only with all-layer, multi-domain route traces and a clearly larger predicted hit benefit |
| Rejected whole-MoE two-dispatch fusion | Folding the shared expert's gate/up, down, scalar router, and final add into the existing two routed-expert Metal dispatches reached 1.289 ms on the layer-0 smoke versus 1.124 ms for the retained routed fusion plus MLX shared path, a 14.7% regression; the first prototype also had a checksum mismatch, so it failed both promotion gates | The supposedly separate MLX shared graph overlaps useful work with the routed graph. Putting all shared weight work behind the routed gate/down dependency serializes it, overwhelming dispatch savings | Implementation removed after the performance gate failed; numerical debugging was intentionally not pursued. A future fusion must prove overlap or parallel scheduling rather than merely reducing dispatch count |
| Rejected shared-only gate/up fusion | Keeping shared down independent but replacing its gate QMM, up QMM, SiLU, and multiply with one small Metal dispatch reached 1.313 ms for the layer-0 complete MoE versus 1.266 ms control, a 3.7% regression; the prototype checksum also differed | Even without whole-MoE serialization, MLX schedules the two shared QMM branches efficiently and its quantized kernels beat the custom scalar unpack loop | Implementation removed after the performance gate failed. Do not repeat shared Q4 fusion without a kernel that first beats both MLX QMM branches in an isolated, dependency-matched microbenchmark |
| Rejected depth-4 history proposals | Allowing every history-cache hit to propose four tokens produced 42.31/43.04/43.03 tok/s warm on code/creative/JSON versus the retained adaptive-depth hybrid's 43.53/44.48/44.16. History accepted 21/36, 29/56, and 33/64 drafts over only 9/14/16 rounds, but the wider verifier cost dominated | Near-free proposal generation does not make S4 target verification free; exact suffix continuation is not confident enough to justify four rows universally | Strategy removed. The experiment exposed and fixed an independent crash: a fully accepted depth-4 round has five head-commit rows, so it now uses the serial commit path rather than calling the S2--4 batched commit API |
| GPU-only MTP state commit | Replacing an unused 10,240-value FP32 host readback with device evaluation preserved the P73 156/190 acceptance and byte-identical output; warm MTP moved 44.48→44.61 tok/s and peak RSS was 33.4 GiB | Committed head state needs an evaluation barrier, not a CPU copy | Small isolated gain; accepted rows are still replayed serially and remain the larger opportunity |
| Bounded MLX cache without per-round clearing | With the allocator cache still capped at 256 MiB, removing 66 explicit round clears preserved byte-identical P73 output and 156/190 acceptance. Cold/warm MTP reached 45.18/45.45 tok/s versus 44.31/44.53 controls; peak RSS stayed 33.5 GiB | Cache byte limits provide the memory bound; clearing every round adds synchronization without improving the guarded footprint | About 2% end-to-end, smaller than raw unaccounted wall time suggested; verifier and accepted-row replay remain dominant |
| Batched accepted-row head commit | On P73, committing current+accepted rows through one S=2..4 head-state pass reduced commit time from about 200 to 122 ms per 256 generated tokens. Cold/warm throughput reached 45.67/46.05 tok/s versus 45.18/45.45, with byte-identical output, unchanged 156/190 acceptance, and 34.3 GiB peak RSS. The low-acceptance Python prompt also preserved output/10-of-22 acceptance and reached 34.83 tok/s versus the prior 34.20 | Accepted target-hidden rows can share the MTP layer pass; target verification remains authoritative, so proposal scheduling does not change generated quality | A tiny cold fixture was neutral/slightly slower (19.07 vs 19.21 tok/s); serial rollback remains available with `QWEN38_BATCH_MTP_COMMIT=0` |

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
13. **Group verifier graph barriers.** Preserve the S-row graph and its BF16
    reduction order, but evaluate at 16-layer boundaries instead of every
    layer. `QWEN38_VERIFY_BARRIER_STRIDE=1` remains the clean diagnostic
    rollback, and all larger-stride tests stay under the memory guard.

## Reject or do not repeat

| Direction | Historical evidence | Decision |
|---|---|---|
| Per-shape affine-Q4 autoregressive QMV retuning | P268's best isolated kernel was 1.082x, but full-engine ON was only 0.10% above pooled controls and slower than reverse controls | Do not continue tile/threadgroup/half-vector sweeps without a fused multi-consumer design |
| Grouped S=2..3 verify gate/up | Isolated 1.06x; full model 44.446 versus 44.897 tok/s, with worse measured acceptance | Rejected as a production optimization |
| Grouped verify down/reduce prototype | Isolated Q4 gain 1.05x; Q8 generalization produced incorrect values | Do not port as-is; require a new exact Q8 design and end-to-end cost case |
| Routed-expert union gate/up prototype | The guarded depth-2 full-model parity run reached 38.3 GiB RSS before completion and was terminated at the 38 GiB cap | Rejected for the 64 GiB product profile; do not raise the safety cap to promote it |
| Copy-free cross-row expert-reuse gate/up kernel | A materially different Metal kernel avoided gathered weight copies and kept exact verifier token parity at 36.0 GiB peak, but the seeded S=3 verifier regressed from about 40 to 55.61 ms | Rejected and removed. At draft width three, expert overlap is too sparse to repay on-device duplicate detection, divergent row scans, and four-lane accumulation |
| GPU-resident autoregressive draft-token chain | Keeping depth-2 argmax IDs on device until one final readback preserved four-token serial parity, but the guarded round smoke regressed from 215.89 to 235.52 ms. Warm HTTP fell from 43.79 to 37.67 tok/s; target verify time also rose from 1616 to 1971 ms because the larger lazy dependency graph degraded downstream scheduling | Rejected and removed. MLX's per-token evaluation boundary is beneficial for this autoregressive head; revisit only with a fused sampler/embedding operator that materializes the token without retaining the logits graph |
| Q8 L47 sidecar as the production drafter | With the identical Q4 target, rolling-economic policy, and B31 code/creative/JSON suite, Q8 reached 36.73/39.32/39.20 tok/s versus Q4's 39.54/40.16/40.80. Q8 draft time was about 6.4--8.3 ms per round versus roughly 4--5 ms for Q4, without a compensating acceptance gain on these trajectories | Keep the reproduced Q4/g64 sidecar as the product default. The older Q8 high-acceptance P73 result does not transfer to the current C++ numerical lineage; extra sidecar precision alone is not a general speed path |
| Interleaving verifier rows by top-k slot | Exact 3-row A/B/A was 56.94 ms row-major versus 57.01 ms interleaved (0.999x) | Rejected; argpartition slots do not provide useful cross-row expert locality |
| Compiling three serial rows inside one fixed S=3 layer closure | Exact layer-major control was 78.368 ms and the compiled closure was 78.344 ms (1.0003x), with token parity and 35.4 GiB peak RSS | Rejected; enclosing three sequential decode graphs in one closure does not remove their weight reads or synchronization cost |
| Same-layer attention-write folded into MLP HC read | Two guarded runs were 33.65 and 33.34 tok/s versus a counterposed read-only HC control at 33.93 tok/s; all 16 tokens matched | Rejected and removed. Writing the complete 10,240-wide updated stream as a second kernel output costs more than the saved dispatch; only revisit as a truly cross-layer deferred value with explicit barriers |
| Standalone single-token HC write Metal kernel | Exact component checksums and all 64 greedy tokens matched, but the guarded full model measured 42.53 tok/s versus 42.62 tok/s for MLX's broadcast multiply/add; both peaked at 35.3 GiB | Rejected and removed. MLX already fuses this elementwise chain effectively, so a custom dispatch adds no useful throughput |
| Compile the lone PLE linear-attention layer after host n-gram lookup | All 64 greedy tokens matched; guarded full-model throughput was 42.70 tok/s versus 42.55 tok/s control, within run noise, with 34.1 versus 34.5 GiB sampled peak RSS | Rejected and removed. This checkpoint has only one PLE layer, so extending the compiled closure cannot close a material decode gap |
| Standalone fused shared-expert Q4 gate/up | The layer-0 MoE probe slowed from 0.807 to 1.350 ms warm and its checksum moved from 0.09523 to 0.08305 | Rejected and removed before a full-model run. MLX's paired QMM graph schedules this small 640-row projection better than an isolated 1,024-thread custom dispatch |
| Smaller fused routed-MoE gate/up threadgroups | Exact layer-0 checksum held, but 256-thread probe timings oscillated between 0.79 and 1.25 ms and the guarded 64-step full model fell to 42.47 tok/s from the 42.62 tok/s control; 512 threads was slower in the isolated probe | Rejected and removed. The 1,024-thread/32-SIMD layout remains the better whole-model scheduler choice despite noisy microbenchmarks |
| Two-update deferred HC without materializing the intermediate stream | Layer-0 first/second checksums matched exactly, but warm latency rose from 1.990 to 2.145 ms before whole-model testing | Rejected and removed. Recomputing the pending BF16 update inside each stream's fused normalization costs more than the eliminated standalone materialization; the retained source's extra `xs` output is load-bearing |
| Custom S=3 GDN verifier recurrence with explicit BF16 boundaries | Convolution state matched, but recurrent max-abs error was 0.015625, layer output error reached 0.86--1.00, and the full verifier lost greedy-token parity | Rejected and removed. Matching elementwise rounding is insufficient because MLX's 128-wide reduction association is acceptance-sensitive |
| Compiled existing S=3 GDN verifier graph | The initialized-state fixture retained exact recurrent/convolution state and output max-abs `3.05e-05`, but guarded full verifier runs were inconsistent: 46.12 then 48.50 ms versus controls 47.14 then 46.81 ms | Rejected and removed. The custom MoE kernels remain scheduling barriers, so wrapping each GDN graph in a closure produces no repeatable whole-verifier gain |
| Extending the S=1 fused HC kernel directly to S=2..64 | On the guarded P73 workload, warm MTP fell from 40.62 to 33.32 tok/s, acceptance collapsed from 156/190 to 10/22, and output changed. Four-row HC smoke checksums also drifted | Rejected and removed. The BF16 reduction order is part of verifier acceptance behavior; a verifier fusion must preserve the retained S-row numerical mode rather than independently applying the S=1 kernel to each row |
| Exact rowwise S=3 fused HC verifier | A four-row component oracle matched mixed, injection, written, and final streams exactly. The seeded verifier improved from 39.73 to 38.25 ms, but a warm depth-2 HTTP cohort was flat-to-worse: verify time 1370.84 to 1371.85 ms and throughput 42.18 to 42.13 tok/s | Rejected and removed. The isolated scheduling gain is hidden by the live graph and does not improve product throughput |
| Batched causal-SDPA verifier | The seeded verifier improved only from 39.84 to 39.13 ms while peak RSS rose from 34.0 to 34.6 GiB. In the same 57-token prompt-cache protocol, warm depth-2 HTTP fell from 43.32 to 40.09 tok/s and took a different MTP acceptance path | Rejected and removed. The wider SDPA workspace and changed numerical mode do not repay eliminating the rowwise KV-prefix reads at this short draft width |
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
