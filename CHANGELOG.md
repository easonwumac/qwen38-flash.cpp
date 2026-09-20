# Changelog

## 2026-09-20 — External Splash 27B qualification

- Qualify Splash 1.0 with its Qwen3.8-27B Q4 target and five-layer DFlash 2
  draft on the same 64 GiB M5 Pro: HumanEval 154/164 and IFBench first-30
  19/30 strict/loose.
- Measure 80.00 tok/s median HumanEval decode and 95.37 native aggregate tok/s
  across the four-stream IFBench run. The default IFBench session peaked at
  42.11 GiB and therefore does not meet this project's 40 GiB product ceiling.
- Under an allocator-enforced 40 GiB ceiling, complete a 131,024-token cold
  prefill at 219.03 tok/s and a 32-token exact-replay decode at 38.67 native
  tok/s; session peak was 23.39 GiB and memory pressure stayed normal.
- Add chat-only complete-program HumanEval support and normalize Splash metrics
  in the long-context runner. Existing protocols and defaults are unchanged.

Details: [Splash 27B evaluation](docs/splash-27b-evaluation-2026-09-20.md).

## 2026-09-19 — VQ v2 paired evaluation

- Prepare existing compact d8 gate/up codebooks independently of down geometry;
  retain original FP16 d2/d4 and wide prefill. v1 arithmetic is unchanged.
- Invalidate the SSD state-cache namespace across this numeric-policy change.
- Complete v2 HumanEval: 153/164 (93.29%), no HTTP errors; historical v1 is
  146/164. Fresh paired IFBench first30 is v1 18/30 versus v2 14/30.
- Paired warm PP: v1 526.68 / v2 535.59 tok/s; target-only short decode:
  30.46 / 30.23; native-MTP fixture: 55.69 / 54.86. No large speed or memory win.
- Retain both revisions, with v1 still the daily default pending selection.
  Document the user-approved 40 GiB reclaimable-memory admission threshold;
  runtime footprint/RSS/availability stops remain 40/38/6 GiB.
- Diagnose three IFBench counting regressions with MTP off: v1 3/3, v2 0/3.
  This excludes MTP alone for those cases, not full-engine numerical issues;
  independent full-model reference parity remains unverified.
- Final clean build and 8/8 CTest pass. Full protocols, distributions, guard
  peaks and raw-artifact hashes are recorded; no VQ v2 long-context claim.

Details: [VQ v2 evaluation](docs/vq-v2-upgrade-2026-09-19.md).

## 2026-09-19 — VQ v2 compatibility

- Support the upstream mixed d8/packed-14 and d4/packed-8 checkpoint in segmented
  prefill; extend exact-order slot-packed down reduction to d4.
- Preserve checkpoint FP16 codebooks for v2. v1 retains its existing numeric
  policy. Reject d4/mixed weights in the opt-in legacy persistent backend before
  creating its GPU resources.
- Clean Release build and 8/8 CTest pass with both model tokenizers. Add nine
  segmented CPU-oracle cases and 24 exact slot-packed/scalar comparisons.
- Stage the pinned v2 assets independently, verify all 141 large-file hashes,
  and regenerate the optional Q4 draft head from v2's Q8 target head.
- This initial compatibility milestone preceded full-model evaluation above;
  **v1's retained scores are not v2 results**.

Details: [VQ v2 upgrade](docs/vq-v2-upgrade-2026-09-19.md).

## 2026-09-19 — Daily-use closeout

- Adopt the existing VQ 2.1bpw automatic configuration for daily use. Native Q6
  MTP remains automatic when compatible assets are present; no runtime or
  precision changes and no experimental backend enabled.
- Condense README into a scorecard, quick start, limitations and document map.
  Preserve detailed measurements in [benchmark history](docs/benchmark-history.md)
  and model comparisons in the [results guide](docs/results-guide.md).
- Align operations with VQ assets, current chunk sizing and one automatic
  configuration. Document a 40 GiB footprint stop threshold and its polling
  limitations; remove obsolete daily profile/REAP launch advice.
- Close the speed-research phase by user decision. 600 PP / 40 target-only /
  60 mixed-workload MTP and VQ 128K remain unachieved/unqualified, not silently
  reclassified as passed.
- Verification: clean Release rebuild and 7/7 CTest passed. Default real-model
  smoke was admission-refused at 40.4 GiB available / 42 GiB required; no guard
  bypass and no new inference score claimed.

Details and frozen runtime identity: [closeout report](docs/daily-use-2026-09-19.md).

## 2026-09-19 — Retained runtime milestones

- `502ef99`: fix native QSA final selection for single-group reductions and add
  GPU/CPU oracle plus state-reuse coverage. Reject the slower persistent target
  verifier; production MTP unchanged.
- `5c5a503`: harden experimental persistent state/cache handoff and honor native
  early EOS; retain recovery checks without promoting the experimental backend.
- `6af8216`: document throughput probes and strengthen paired comparisons;
  component wins are not promoted as whole-model speed records.
- `dc33cd2`: fix speculative QSA rollback and qualify native VQ MTP with a fresh
  164-task HumanEval run: 146/164, zero HTTP errors, original-test protocol.

Earlier accepted and rejected work remains in the
[research ledger](docs/prior-research-ledger.md) and
[public quality evaluation](docs/public-quality-evaluation.md).
