# MTP ownership and KV append experiments

## Retained change

`commit_mtp_target_verification` now clears the consumed verification rows
after moving the selected checkpoint. Previously its rvalue-reference argument
left rejected checkpoints alive in the caller until the whole round returned,
including target rollback materialization. No model arithmetic, acceptance
rule, head rebuild order, or runtime profile changed. Callers must share any
needed stream before commit; all existing callers were checked. Invalid commits
still validate before mutation.

Tests cover all acceptance counts 0..4, consumed rows, retained stream values,
and invalid commits. Fresh Release build and all five CTests passed.

The model-free `qwen38-state-lifetime-probe retain|release` constructs five
independent 32 MiB checkpoint arrays. After a GPU completion fence, old behavior
retains 160 MiB and the new behavior retains 32 MiB for every acceptance count.
This demonstrates ownership reclamation, **not 128 MiB saved in the model**.
Actual verifier full-attention checkpoints are slices sharing a complete KV
backing. That backing remains live through the selected slice. Physical memory
and full-model PP/decode improvements have not been demonstrated.

## Rejected simple KV preallocation

The same probe's `concat|reserve` modes compare a synthetic BF16 history of
32,768 rows x 256 elements extended by 32 chunks of 512 rows. Both retain an
origin snapshot. Reserve uses MLX slice_update into capacity for all final rows;
concat uses existing concatenation. Full-array equality and unmodified origin
are checked after timing. This is append only, not attention, QSA, or inference.

On the local M5 Pro / 64 GiB machine, four processes in A/B/B/A order each ran
seven repetitions (discard first). Warm allocator peak was 64 MiB for concat,
72.25 MiB for reserve. Timings overlap and are noisy (roughly 20–34 ms per
32 appends). No consistent speed improvement or memory improvement: rejected.
This does not reject truly segmented KV with a block-aware attention kernel;
it rejects assuming slice_update alone provides that design.

## Remaining work

- Direct compressed-qmeta decode kernels already exist. Wide PP still expands
  metadata for grouped matrix multiplication; a PP-specific tiled compressed
  kernel remains unimplemented/unbenchmarked. Reusing a decode kernel is not
  evidence of efficient PP.
- Earlier speculative-head cleanup needs separate exception/lifetime analysis.
- Full-model admission check refused 39.9 GiB available versus required 44 GiB.
  The threshold and existing server were not changed. End-to-end parity and
  PP/decode no-regression benchmarking remain pending.

## Safety and reproducibility

Build target `qwen38-state-lifetime-probe`; run each mode serially under
`devtools/memory_guard.py`: start 16 GiB, minimum available 12 GiB,
RSS/physical footprint caps 4 GiB, interval 0.1 seconds. Use the private strict
MLX runtime with `MLX_STRICT_MEMORY_LIMIT=1`. The probe sets a 1 GiB allocator
limit and 16 MiB cache. No model weights are loaded. Thermal conditions were
not controlled. Guard sampling can miss very short process peaks; allocator
measurements above are not RSS or system memory savings. Raw outputs are in
`state-lifetime-results.json`.
