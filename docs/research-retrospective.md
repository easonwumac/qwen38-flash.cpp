# Performance research retrospective

This file records measured dead ends so future optimization work starts from the
accepted production baseline instead of repeating expensive experiments. Raw
derived checkpoints and disposable build trees are intentionally not retained.
The Git history and the project workflow reports preserve the detailed evidence.

## Retained production baseline

- Serve the proven REAP-288 Q4 layout with exact top-10 routing.
- Keep serial decode as the daily default. It is stable at roughly 41 tok/s on
  the measured M5 Pro 64 GiB system.
- Use a 1024-row prefill chunk for normal `speed` requests. The fixed 8K corpus
  reached 756--757 PP/s warm; 32K remained near 571 PP/s.
- Treat 192K as the largest proven safe context for this checkpoint. The 192K
  retrieval run was correct; 262K did not complete and is unsupported.
- Prefer the AoS n-gram table through bounded `pread`; retain the original table
  only as a compatibility fallback.
- Retain RAM and SSD prefix reuse, memory admission guards, JSON guard reports,
  soak tests, and fail-closed capability checks.
- Keep MTP opt-in. Explicit Q8 depth 4 can exceed 65 tok/s on favorable prompts,
  but mixed and creative workloads do not justify enabling it globally.

## Directions not to repeat unchanged

- Expert paging and fixed-slot SSD streaming reduced residency but collapsed
  decode to roughly 1--7 tok/s. I/O batching alone did not remove the per-expert
  synchronization and compute overhead.
- Larger 28/36 GiB paging budgets did not produce a reliable speed win over the
  smaller paging experiments. Do not infer throughput from cache hit rate alone.
- Packed decode and parallel expert reads showed no reliable benefit and remain
  unsuitable as defaults.
- Persistent W8 sidecars conflict with the memory target. Direct Q4-to-MPP was
  only a bounded prototype and has no full-engine speed evidence.
- Selective Q8, Q2, Q3, hybrid attention/FC, and lossy qmeta checkpoints either
  changed exact output or failed the combined quality, memory, and speed gate.
  Do not retain derived model copies after their measurements are recorded.
- Packed QSA at 32K measured about 555 PP/s and changed the first-token hash.
  Fixed 1024-row chunks and broader qmeta caching also did not close the 32K
  600 PP/s gap.
- BF16 attention scores, synchronized QSA tiling, and post-GEMM reductions must
  not be retried without a design that preserves FP32 selection and accumulation.
- Q4 MTP and unconditional Q8 MTP did not protect every individual workload.
  Aggregate acceptance or median speed is insufficient for a daily default.
- Adapter sweeps and MTP calibration artifacts are research outputs, not runtime
  dependencies. Keep conclusions and gates, not every generated checkpoint.

## Gate for future experiments

A new optimization must be tested against the exact serial baseline and report
model/quantization, prompt corpus, context, sampling, MTP depth, hardware,
thermal controls, memory guard measurements, and distribution statistics. It is
promotable only if it preserves required hashes or explicitly passes a reviewed
quality gate, improves the target workload reproducibly, and does not trade away
the daily memory headroom. Stop early when the guard reaches the configured
available-memory floor.
