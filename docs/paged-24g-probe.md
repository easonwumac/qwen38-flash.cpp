# Full-model paging under a 24 GiB test ceiling

## Scope

Research-only `qwen38-paged-model-probe`; the server/default loader is unchanged.
The retained model is `Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47`.
All selected experts use their original affine weights/metadata. MTP, compact
qmeta, reduced top-k and fused/grouped MoE are not used. No weights were converted
or removed. The fixed 16 GiB expert tier is a first full-model integration, not
the completed adaptive process-budget controller.

Hardware: Apple M5 Pro, 64 GiB, macOS 26.5. Private strict MLX build linked at run
time; 20 GiB MLX allocation cap, zero allocator free-block cache. The external
guard samples every 0.1 seconds, requires 40 GiB start / 12 GiB available system
memory and terminates at 24 GiB RSS or physical footprint. The probe additionally
checks footprint after loading and each token, stopping above 22 GiB. These are
layered experimental safeguards, not a mathematically hard process-allocation
limit or authorization for arbitrary long contexts.

## Implementation

An explicit nonzero expert budget enables the experimental `MlxTensorStore`
route. Only tensor metadata is mapped; requested bytes are read into bounded
host staging and copied into aligned MLX buffers. It does not retain whole MLX
shard maps. Expert bundles contain gate/up/down weights/scales/biases, charged
conservatively for 16 KiB allocator pages. Loads validate actual active-byte
growth against that charge. Each lease survives its GPU completion fence.

The first paged MoE uses the existing reference router, per-selected-expert QMM
and ordered BF16 reduction. Shared-expert computation retains its original
batch shape: computing it row by row caused a detected BF16 parity difference
and was corrected. Prefill currently loops routed rows, so it forfeits grouped
prefill throughput. Nine serial reads/copies/evaluations per cache miss and
per-expert GPU synchronization are substantial overheads.

## Initial measurements (2026-09-06)

This is a synthetic correctness/feasibility workload, not a conversation corpus:
greedy generation starts at token 9419, without chat template or EOS stopping.
The separate PP probe uses eight IDs: `9419,11,358,1440,264,1937,13,198` in a fresh
decode state after generation, but with the same expert cache. OS file caching
was not flushed, thermals were not controlled, and no energy claim is made.

| Measurement | First 8-token run | Subsequent 64-token run |
|---|---:|---:|
| Model load | 1.536 s | 0.647 s |
| Decode excluding first token | 0.801 tok/s | 1.723 tok/s |
| Median step excluding first | not recorded | 0.531 s |
| p95 step excluding first (index floor) | not recorded | 0.965 s |
| Eight-token PP | 1.255 tok/s | 3.755 tok/s |
| Process footprint peak (sampled) | 10.1 GiB | 19.3 GiB |
| Expert cache after decode | 4.580 GiB | 15.999 GiB |
| Decode hits / misses / evictions | 2115 / 1725 / 0 | 24555 / 6165 / 139 |

The 64-token run reached the expert cap at step 60. Footprint stayed about
19.202 GiB through steps 60–63, then 19.265 GiB after PP. Minimum available system
memory was 26.5 GiB. Cache-miss loading accumulated 21.437 seconds during decode;
this includes host reads/copies and evaluation, not isolated physical SSD time.
The later run benefits from earlier OS cache activity and generates different
later text; it is not a controlled speedup comparison. The short first run is
not a verified cold-storage benchmark either.

After adding allocator-delta validation, two retries stopped safely around
4.5 GiB: deferred releases from earlier GPU work made the global active-byte
delta negative (4,560,081,982 -> 4,559,571,726 bytes during a 2,850,816-byte load).
The accounting snapshot now fences previous GPU work first. The check remains
enabled; no cap was relaxed. A final 64-token rerun then completed with the same
hit/miss/eviction counts and a 19.3 GiB process peak. With warmer OS caching it
measured 3.386 tok/s excluding the first token, median 0.296 s, p95 0.384 s,
and PP 3.742 tok/s. Cumulative miss-loading time was 5.976 s; system headroom
never fell below 25.9 GiB. This is evidence of sensitivity to file-cache state,
not a demonstrated algorithmic 2x speedup or sustained SSD streaming rate.

The first eight output IDs agreed between both runs:
`11,271,40,599,264,3377,440,821`. This repeatability is not full-model parity with
the optimized server. Real layer-0 and layer-47 router/shared-branch parity passed exact BF16
comparison for 1/4 rows against the unfused reference with an 8 MiB expert cache
and forced evictions. Broader quality and long-context tests remain pending.

A clean Release build completed under a 4 GiB build-process guard (0.3 GiB
footprint / 0.4 GiB RSS peak). CTest: three suites passed; the tokenizer fixture
suite was skipped. The two-layer parity probe peaked around 0.86 GiB footprint.

## Reproduce

Use only with the known private strict MLX library, not the advisory stock cap:

```sh
python3 devtools/memory_guard.py --min-start-gib 40 --min-available-gib 12 \
  --max-rss-gib 24 --max-footprint-gib 24 --interval 0.1 -- \
  env MLX_STRICT_MEMORY_LIMIT=1 DYLD_LIBRARY_PATH=/path/to/strict-mlx/stage/lib \
  build-paged24-verify/qwen38-paged-model-probe /path/to/model --extended
```

`--layer-parity` instead exercises bounded reference comparison; use a 4 GiB
external process ceiling and 16 GiB start headroom for that mode.

## Conclusion and next gates

The full model can generate a short sequence below the requested 24 GiB ceiling
without keeping all experts resident. This initial serial implementation is
much too slow to replace the current engine. Next work is coalesced/asynchronous
expert reads, packed/batched selected-expert execution with fewer fences, and
grouped paged prefill, while retaining exact routing and accounting. Adaptive
expert allowance must include growing state, staging and host/framework memory;
long-context, MTP and service integration are not validated by these results.
