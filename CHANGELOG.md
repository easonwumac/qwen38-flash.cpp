# Changelog

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
