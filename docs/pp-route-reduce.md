# PP gather/reduce fusion: guarded candidate

Status: developer opt-in only (`QWEN38_PP_ROUTE_REDUCE=1`), not enabled in any
product profile. No service replacement, quantization change, expert pruning,
or memory-guard relaxation. Full-model non-regression remains a promotion gate.

## Change

The grouped prefill MoE path previously materialized a route-reordered down
projection and then a weighted copy before summing. A single Metal kernel reads
the sorted down projection through the inverse permutation, rounds the weighted
products to BF16, and accumulates in the same eight-partial BF16 order as the
pinned MLX col_reduce_small implementation. This ordering was verified against
local MLX source; attribution and the MIT license are retained.

An initial FP32 accumulator candidate changed the layer-0 output hash and was
rejected despite faster timing. The retained candidate preserves the tested
BF16 output, not a mathematically similar higher-precision substitute.

Contract: BF16 down/weights, top10, hidden2560, at most1024 rows. Other model
geometries fall back to the original path. Decode and compact S<=8 verification
return before this path; there are no added persistent weights or caches.

## Test setup

2026-09-06, M5 Pro 64 GiB, private strict MLX runtime based on pinned 0.32.2,
REAP-288 affine Q4/group64 target from the retained Q8-MTP L47 model pack.
Lossless16 qmeta, top10, temporary qmeta with deferred reduction, no request-wide
decoded-metadata cache; speed kernels otherwise unchanged. MLX cache64MiB,
allocation limit3GiB, guard4GiB RSS/footprint and12GiB minimum available.
No concurrent GPU test, thermal pinning, or filesystem cold-cache controls.

`qwen38-sparse-moe-smoke`: layers0/47, rows16/128/512, A/B/B/A order,22 iterations
per process, first excluded. All24 process output hashes matched their paired
controls. Inputs are deterministic embedding/HC activations, not real layer47
hidden states from a full model. At512 rows, warm MoE time improved approximately
6–7%; MLX allocation peaks dropped by exactly26,214,400 bytes (25MiB) in these
paired runs. End-of-run active memory was unchanged: this saves temporaries,
not steady model weights. At128 rows the peak reduction was6.25–7.5MiB, with
small/noisy speed changes. At16 rows it was less than1MiB.

`qwen38-pp-reduce-layer-ab`: actual layer0, prepared embedding input, two
successive chunks, repeated and diverse IDs, complete output and GDN state
evaluation. Each process performs3 warmup pairs then21 alternating pairs;
three independent processes passed exact full-array output/convolution/recurrent
comparisons. Median of the three per-process medians:

| Chunk rows / inputs | Control, two chunks ms | Candidate ms | Throughput ratio |
| --- | ---: | ---: | ---: |
|128 / repeated|8.295|8.222|1.009x|
|128 / diverse|14.591|14.417|1.012x|
|512 / repeated|24.887|23.670|1.051x|
|512 / diverse|32.116|30.844|1.041x|

Layer A/B process peak was1.8GiB; it holds both candidates in one process, so it
does not measure a paired complete-layer footprint reduction. One128/repeated
process had a0.4% time regression; small-width non-regression is not established
statistically. Fresh-kernel compilation cost is not included in warm numbers.

Narrow-path checks at layers0/47 S1/S5 retained exact hashes. Longer layer0
S1/S5 A/B/B/A runs (3 cycles,202 iterations/process) also retained hashes and
allocation peaks. Timing varied substantially even though the candidate kernel
is not called in this path; do not claim a decode speedup or guaranteed
full-engine non-regression from these measurements.

Unit tests additionally compare every output element against stock operations
for rows1/16/128 with signed, varied-magnitude inputs, cancellation, nonuniform
BF16 weights, and an inverse permutation. Fresh Release build and5 CTest suites
passed. Raw results are in the adjacent JSON files; very short process guard
samples sometimes report zero because the process exited before a sample, not
because it consumed no memory.

## Next gates and remaining work

Full resident model: paired cold/warm PP, decode, MTP, quality, SSD cache extension
and long-context footprint. A non-mutating admission check refused37.6GiB
available against44GiB required. No full-model run was attempted or service
changed. Do not multiply the25MiB per-layer-test saving by48 or apply the4–5%
layer speedup to whole-model PP.

This is one bounded memory/speed candidate, not completion of all proposed work.
Lossless13 is not promoted because its historical decode regression conflicts
with the user's no-slowdown requirement. PP metadata tile fusion and blockwise KV
updates remain separate, unimplemented research directions. The existing
SSD-backed prefix ownership fix still needs full-model validation.
