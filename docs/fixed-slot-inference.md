# Fixed-slot full-model inference (research only)

This is NOT the production resident engine or a serving preset. The guarded
`qwen38-paged-model-probe --fixed-slots` integrates the fixed expert slots into
the complete target model for short decode and PP measurement. `--fixed-baseline`
uses the same arithmetic with all 288 experts resident. Neither flag enables MTP.

## Ownership and routing

Each layer owns nine aligned managed arrays with 256 fixed hot slots and eight
cold slots. The 48 cold regions total approximately 0.989 GiB, but are partitioned
per layer, not globally adaptive. Hot IDs 0..255 are a placement experiment,
NOT a verified REAP-256 subset or a measured least-frequency selection.

All 288 original experts remain available to the original CPU router. Cold
misses read all nine original weight/metadata rows before publishing the new ID.
Partial I/O failure poisons the layer and aborts the probe. Cache views hold
managed-array storage ownership; a leased slot cannot be replaced. Host writes
wait for GPU completion. Fresh fused graphs directly index hot/cold slots with
no per-hit concatenation. A top-10 route needing more than eight cold experts
uses ordered unfused projection instead; the all-resident control follows the
same rule to avoid comparing different arithmetic on overflow routes.

PP retains the previously validated grouped-QMV path, including ordered BF16
reduction and groups of eight leased experts. This does not recover the optimized
resident engine's batched PP implementation. Decode still includes CPU routing
readback and per-layer synchronization. These are important performance limits.

## Candidate measurement

Apple M5 Pro /64 GiB; retained Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47,
Q4/group64 experts with BF16 metadata; no additional pruning, no MTP, greedy
target generation. 64 decode steps from token 9419; timing excludes first step
but includes cold-cache warmup thereafter. PP uses the same three plain-text
66/68/68-token prompts in the probe (cache explanation, Python review, Chinese
meeting notes), chunk size at most 64. No long-context or prompt-cache test.
OS file caches and GPU clocks/thermals were not controlled; single runs are not
distribution-level speed claims.

| Candidate metric | Measured |
| --- | ---: |
| Model + hot expert preload | 48.79 s |
| Decode excluding first step | 6.55057 tok/s |
| Step median / p95 | 140.05 / 214.77 ms |
| Decode cold misses / evictions | 708 / 328 |
| Decode cold-load time, including first step | 3.454 s |
| PP 66 tokens | 10.8565 tok/s |
| PP 68 tokens, Python | 8.53056 tok/s |
| PP 68 tokens, Chinese | 17.3424 tok/s |
| PP misses | 638 / 831 / 627 |
| PP loader seconds | 1.564 / 1.197 / 0.713 |
| Peak / final process footprint | 35.8 / 35.665 GiB |

Full PP logit hashes were 2909080846528897829, 17655255725674242853 and
10959034544364495653, matching the prior rowwise/grouped controls. Decode hash
was 9300840567661459111. Layer0/47 rows1/4/32/64 candidate-vs-fixed288 comparison
passed exactly. This is numerical coverage, not a broad language quality test.

External guard: start42 GiB for candidate,44 GiB for baseline; minimum available
12 GiB, RSS/footprint42 GiB. Private strict MLX cap40 GiB; internal footprint
early stop40 GiB. Candidate minimum available12.6 GiB, no OOM. Production is
unchanged. The fixed-slot initializer explicitly requires the guarded strict
research environment, and its budget cannot be resized through the old cache API.

The candidate is still much slower than the existing production engine. Do not
promote it merely because it saves memory. In particular, PP time is dominated
by more than cold reads; changing SSD I/O alone cannot recover resident PP.

## Same-engine all-resident control

| Metric | 288 resident | 256 + cold slots |
| --- | ---: | ---: |
| Decode excluding first step, tok/s | 9.19905 | 6.55057 |
| Step median / p95, ms | 77.65 / 96.80 | 140.05 / 214.77 |
| PP cache-explanation, tok/s | 25.1697 | 10.8565 |
| PP Python, tok/s | 25.0873 | 8.53056 |
| PP Chinese, tok/s | 27.5356 | 17.3424 |
| Peak footprint, GiB | 38.8 | 35.8 |

Both decode hashes and all three full PP hashes matched exactly. The baseline
reported zero expert misses, load53.20s and final footprint38.6164GiB. Its guard
minimum available was12.7GiB. Its second decode step took1.895s despite no expert
misses, so excluding only step0 does not constitute a fully warmed benchmark.
This is one sequential pair, not a stable percentage-regression estimate.

The research baseline itself is far below historical optimized serving speeds.
The comparison measures the current synchronous integration, not the best possible
paging implementation. It neither validates a low-cost 3GiB saving in production
nor establishes that SSD paging can never be efficient. No serving changes.

## File mapping alternative

Metal [bytesNoCopy buffers](https://developer.apple.com/documentation/metal/mtlbuffer)
can reuse existing storage. That can avoid an extra staging copy, not the need
for physical memory when the GPU uses the data. Earlier local managed-region
tests found misaligned tensor payloads incompatible with typed QMM without aligned
staging; see [managed regions](managed-expert-region.md). A cold-expert aligned
file layout with bounded region lifetimes remains an alternative to test, not
a demonstrated SSD-to-GPU bypass or a guaranteed residency cap from `mmap` alone.

Apple also provides [Metal I/O queues](https://developer.apple.com/documentation/metal/mtliocommandqueue)
and file-to-buffer load commands. This is a candidate for asynchronous bounded
cold-slot fills, not yet implemented or benchmarked in this engine.

## Cold start versus recurring misses

64 target steps make30720 expert requests. Candidate708 misses are2.305% of
requests, about11.1 loads/step. Step0 accounts for55 misses; the remaining653
occur after it.328 evictions confirm recurring replacement, not just initial
filling. The final steps still include misses, although steps59/60 have none.
PP's distinct expert working set causes638/831/627 misses on the three prompts.

These counters mean expert-cache misses and file-load requests, NOT measured
physical SSD operations. macOS may satisfy reads from its page cache, and the
loader timer includes synchronization and host copying. No block-I/O attribution
was collected. The fixed eight-slot-per-layer partition and uncalibrated numeric
hot list are experimental constraints, not an optimal hot-set policy.
