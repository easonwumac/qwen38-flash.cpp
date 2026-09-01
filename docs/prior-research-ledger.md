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
| P73 serial | 35.028 decode tok/s; 479.357 prefill tok/s | The retained Zig/MLX target path can exceed the current C++ runtime | Fixed 186-token code prompt, 512 generated tokens, hot run |
| P73 Q8 MTP depth 2 | 59.866 decode tok/s; 463.316 prefill tok/s; 78.6% per-draft acceptance; 1.57 accepted drafts/round | Correct MTP can be strongly profitable on this machine | Workload-specific high-acceptance code result, not universal throughput |
| P60 corrected Q8 MTP | 44.36 decode tok/s versus 32.13 serial; about 60% acceptance | A broader workload can still benefit, but verifier cost and acceptance dominate | Different run and workload from P73 |
| P268 serial Q4 sweep | 34.359 all-control median; candidate 34.392 | Shape-specific QMV tuning alone does not move end-to-end throughput | One selected projection geometry |
| Current C++ runtime | about 31.0--31.3 warm decode tok/s; about 35.6 GiB peak | Correct full-model C++ baseline with bounded resident expert tier | No MTP, prefix cache, or complete long-context QSA yet |
| C++ dense/state/shared-MoE-batched verifier | 60.37 ms for three rows versus 94.90 ms serial; 1.572x; exact token and rollback-continuation parity | Layer-major scheduling plus batched state projections and shared expert improve verifier cost | Depth 2, empty origin, one interleaved A/B/B/A run; routed experts remain row-serial and round cost is still above P73 |

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

## Reject or do not repeat

| Direction | Historical evidence | Decision |
|---|---|---|
| Per-shape affine-Q4 autoregressive QMV retuning | P268's best isolated kernel was 1.082x, but full-engine ON was only 0.10% above pooled controls and slower than reverse controls | Do not continue tile/threadgroup/half-vector sweeps without a fused multi-consumer design |
| Grouped S=2..3 verify gate/up | Isolated 1.06x; full model 44.446 versus 44.897 tok/s, with worse measured acceptance | Rejected as a production optimization |
| Grouped verify down/reduce prototype | Isolated Q4 gain 1.05x; Q8 generalization produced incorrect values | Do not port as-is; require a new exact Q8 design and end-to-end cost case |
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
4. **Prefix/prefill cache parity.** Implement chunked prefill and exact prefix
   snapshots for the target. MTP restoration must include its complete QSA
   state or rebuild from a freshly forwarded tail; it must never silently
   restore KV alone.
5. **Verifier-wide fusion only after profiling.** The remaining plausible
   kernel opportunity is a fused, exact S=2..4 verifier path that removes
   model-byte movement, intermediates, and dispatches across multiple
   consumers. Improving one projection in isolation is explicitly excluded.
6. **Branch-local shallow tree only after linear MTP is profitable.** A B=2,
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
