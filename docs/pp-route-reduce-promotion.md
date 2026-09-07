# PP route/reduce full-engine promotion attempt

2026-09-08: **not promoted**. No full-engine throughput or output-parity result
was obtained. The candidate remains developer opt-in; serving configuration,
model weights, and memory protection were not changed.

## Prepared validation

The existing `qwen38-mtp-cache-probe` now accepts optional depth (0 or 4) and
repeat count (2 through 5), with context 0/8192/32768. Original invocations keep
their depth4/two-repeat behavior. The allocator cache cap is explicitly 256 MiB,
alongside the existing 40 GiB allocation limit. This allows the same full-engine
probe to cover serial decode, MTP, repeated uncached PP, and SSD prefix reuse.

Planned comparison: separate sequential processes in A/B/B/A order, changing
only `QWEN38_PP_ROUTE_REDUCE=0/1`. Start at 8K, without prefix caching, three
requests per process. Treat the first as cold-process and later requests as
warm; do not mix them into one median. Compare every generated token and MTP
acceptance counts before interpreting timings. Then cover the four short
prompt cases, depth0, and a fresh SSD cache in each arm. Larger contexts require
adequate observed headroom; an admission estimate alone is not sufficient.

## Actual attempts

Hardware: M5 Pro, 64 GiB. Retained REAP-288 Q4/group64 target with full Q8 MTP
L47 pack; lossless16 metadata, resident expert range12:29, speed profile,
temporary metadata deferred reduction, prefill512, no decoded-metadata prompt
cache, depth4/fallback, thinking off. Synthetic numbered weather records with
a final short Python request, requested context8192. Strict MLX library used.
Neither attempt completed the first request, so actual prompt/output timing
was not emitted. No thermal pinning or filesystem-cache controls.

| Attempt | Admission threshold | Stop footprint | Stop RSS | Stop available | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control, first | 40 GiB | 38.8 GiB | 15.5 GiB | 8.0 GiB | 76 |
| Control, retry | 48 GiB | 38.8 GiB | 31.0 GiB | 7.7 GiB | 76 |

The first admission check reported41.1 GiB available. After that process exited,
a separate check reported49.8 GiB, motivating one retry with a higher admission
threshold. Both runs retained the8 GiB available-memory floor and42 GiB
RSS/physical-footprint ceilings. These are sampled stop readings, not complete
peak measurements. Available memory includes reclaimable file-backed pages;
it must not be interpreted as guaranteed additional allocation capacity.

The guard terminated both runs before results. No candidate run or larger
context was attempted afterward. Other running applications, including oMLX,
were left alone. Five CTest suites passed with the retained model supplied for
the tokenizer test; a fresh Release build passed (existing library deployment
target warnings:26.0 versus26.2). `git diff --check` passed.

## Remaining gate

Retry in a lower-memory-pressure session, keeping the same runtime/model and
protection settings. Do not reduce resident experts, change quantization, or
relax the stop floor to manufacture a successful comparison. The prior local
layer speedups remain local-layer evidence only. Full-model PP, decode, MTP,
quality and SSD-cache non-regression are still unverified.
