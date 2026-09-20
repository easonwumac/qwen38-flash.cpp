# Changelog

## 2026-09-20 — External Splash 27B/35B qualification

- Under the same thinking/16K first-30 IFBench protocol, qualify dense 27B at
  30/30 and 35B-A3B at 27/30 strict/loose. The corrected 27B result supersedes
  the earlier 19/30 run whose 4K output budget truncated 11 cases.
- Complete HumanEval and MBPP+ controls: 27B scores 154/164 and 294/378; 35B
  scores 149/164 and 284/378. These code controls disable reasoning.
- On a fixed 12-prompt thinking workload, measure 27B at 45.71 B=1 / 102.61
  B=4 aggregate tok/s and 35B at 139.56 / 255.19 tok/s.
- Under a 40 GiB ceiling, measure 35B at 2,410.55 PP tok/s at 32K and 1,161.30
  PP / 135.85 exact-replay stream decode tok/s at 131K. The 128K session peak
  was 21.78 GiB in Splash's Metal-allocation counter.
- Extend evaluation tools for Splash metrics, MBPP solution output, and partial
  GPQA scoring without changing the existing default protocols.

Details: [paired Splash evaluation](docs/splash-comparison-2026-09-20.md).

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
