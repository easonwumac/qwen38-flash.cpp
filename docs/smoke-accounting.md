# Short smoke ownership accounting

Developer smoke only: release bootstrap logits after selecting the token and
serial oracle state after extracting its four token IDs. Add active/cache/peak
MLX bytes and physical-footprint samples at phase boundaries. No additional GPU
fences; immediate frees may still be affected by outstanding work. The production
server and its kernels/defaults are unchanged.

Same model, strict runtime, speed/lossless16/resident 12:29 recipe and
40 GiB start / 8 GiB available floor / 42 GiB process ceilings as the previous
admission experiment. M5 Pro 64 GiB, one bootstrap token 9419, serial four-token
oracle, MTP depth 2, no prefix cache. One run, no thermal control.

This run completed: tokens [271,40,599,264], serial parity true, three MTP rounds,
two accepted drafts, five emitted tokens. Guard peak footprint 38.0 GiB,
peak RSS 34.5 GiB, minimum available 10.5 GiB. Initial system availability was
about 49.89 GiB, higher than earlier aborted runs; completion is NOT proof that
the cleanup alone made the difference, nor that a 40 GiB start is always safe.

Measured deltas:

- Bootstrap logits release: 507,908 active bytes (~0.48 MiB).
- Serial state release: 60,432,522 active bytes (~57.63 MiB). Physical footprint
  did not immediately drop by the same amount; freed allocation moved to cache.
- First MTP round: active 39,006,554,726 -> 40,614,199,910 bytes (~1.50 GiB more).
  Subsequent rounds remained near 40,614,3xx,xxx bytes. This points to first-use
  materialization/initialization as the next accounting boundary, not evidence
  of a growing per-round leak. Its weights-versus-temporary composition is not
  yet measured.

The logged serial/MTP rates are cold, four-token diagnostic timings with memory
logging and cache clears; they are not comparable to the historical warm
128-token throughput benchmark and do not demonstrate a speed improvement.
No long-context claim. Raw output is in smoke-accounting-output.txt.

Swap use before/after remained 1687.44 MiB; swapouts unchanged at 3,826,768.
Compressor occupancy increased 158,930 -> 204,369 pages (16 KiB pages), so
unchanged swap does not imply no pressure. Fresh Release build and all five
CTests passed. Full-model evidence is limited to the four-token parity above.
