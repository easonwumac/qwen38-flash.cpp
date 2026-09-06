# QSA peak-memory experiments: rejected for production

2026-09-06. None meets the combined lower-peak/no-PP-regression gate. Production
SelfAttention retains its original score path. Experimental branches compile only
into `qwen38-qsa-peak-probe` with `QWEN38_QSA_PEAK_RESEARCH`; no new server profile
or runtime environment check is introduced by these experiments.

## Bounded setup

M5 Pro64GiB; strict private pinned MLX runtime; actual layer3 Q4 attention and
embedding/HC weights from the retained REAP-288/Q8-MTP-L47 model pack. Attention
history is deliberately synthetic deterministic BF16 data, NOT computed from a
real32K/64K/128K prompt. Thus these are kernel/attention memory and timing probes,
not long-context quality, retrieval, whole-model PP, or decode benchmarks.

For each context32768/65536/131072, extend one512-row diverse embedding/HC input
through packed QSA. Fourteen iterations per process, first three excluded;
median of11 samples. Each output and complete KV/raw/pooled-state tensor is
hashed with finite checks; every paired hash matches. Hash conversion occurs
after timing/MLX peak sampling, so process-guard peaks may include conversion
buffers. MLX allocation limit3GiB, cache64MiB, process/RSS guard4GiB, startup16GiB
and ongoing12GiB availability. One GPU process at a time. No thermal or cold-disk
controls; timing results are diagnostic, not rigorous noninferiority estimates.

## Results

### Fuse ReLU and head sum

The logical ReLU bank is about128MiB at64K,512 rows,4 heads. Fusion preserves
the FP32 head-reduction order, confirmed by full-array unit tests. However actual
attention allocation peaks were effectively unchanged: approximately
852,425,872 /1,183,861,904 /1,846,709,392 bytes for32K/64K/128K. Fused variants
differ by only scalar/allocation noise, not a saved full bank. MLX can reuse
storage; logical temporary size is not proof of additional simultaneous residency.
Which buffer is donated in this exact graph has not been instrumented.

Warm timing changes were small/noisy (including a128K regression in some runs).
No peak-memory benefit, so no production promotion.

### Bound the score matrix with query-row tiles

Use original FP32 matmul, ReLU and head sum, evaluating one query tile before
moving on. This reduces live score-bank size without changing tested outputs.

| History | MLX peak saved, sync128/256 tiles | Sync128 latency change | Sync256 latency change |
| --- | ---: | ---: | ---: |
|32K|56.8MiB|+10.0%|+7.0%|
|64K|108.8MiB|+5.0%|+12.3%|
|128K|212.8MiB|+16.2%|+12.4%|

Each percentage uses its adjacent same-build control. These candidates are
rejected: lower memory but slower PP violates the user constraint.

### Asynchronous256-row tile submissions

Avoiding the CPU wait did not retain the memory bound: outstanding work overlaps
allocations. Peak grew by81.25/161.25/321.25MiB at32K/64K/128K relative to stock,
and measured latency grew by1.2%/7.0%/10.4%. Rejected, not shipped.

## Reproduction

Build target `qwen38-qsa-peak-probe`, run through the guard and strict runtime:
`qwen38-qsa-peak-probe MODEL CONTEXT MODE`.

- 0: stock control.
- 1: fused ReLU/head sum.
- 2: synchronous128-row tiles.
- 3: synchronous256-row tiles.
- 4: asynchronous256-row tiles.

The probe clears inherited tile selection so these modes are explicit. Earlier
raw results used environment-selected prototypes; their JSON wrapper labels
record the tile size/submission mode at the time. Raw timing arrays and negative
results are retained in `qsa-peak-results.json`.

## Conclusion

Fresh Release build and all5 CTests passed. The isolated final probe rechecked
all five modes at64K with identical output/state hashes; raw outputs are in
`qsa-peak-final-verification.json`. Inspection of the normal MLX archive found
neither experimental environment selector, confirming the probe-only compilation.

The tested buffer changes do not deliver a new deployable peak-memory win.
Keep the earlier PP route-fusion candidate and original QSA behavior. A future
QSA attempt would need to fuse score production/reduction inside the GEMM or
avoid the score bank without CPU-synchronized tiling. That is unimplemented;
no speed or memory payoff is claimed. Full-model and long-context validation
of the earlier candidates remains outstanding; guard settings were not lowered.
