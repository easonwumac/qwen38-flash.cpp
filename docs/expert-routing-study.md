# Expert frequency calibration and held-out cache replay

This is a small routing study, not an end-to-end speed claim for the proposed
memory budgets. We captured actual model expert requests, then replayed those
same requests through different CPU-only cache policies. The new budgets are
expert payload capacity, excluding dense weights, KV, PLE, MTP and workspace.

## Collection and safety

Model: retained Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47,
Q4/group64 routed weights, BF16 affine metadata. Apple M5 Pro,64GiB. MTP off,
greedy decode. Workload:64 seed decode steps from token9419; three existing plain
text prompts66/68/68tokens (cache explanation, Python review, Chinese notes),
each followed by64 decode steps. No chat template, EOS stopping, long-context,
user workload or quality benchmark is implied.

The first256-hot capture was stopped by the unchanged12GiB available-memory
floor during the second prompt's decode, at35.7GiB footprint. No OOM. Its partial
file is not used. A224-hot+8-cold-per-layer capture completed at31.9GiB peak,
31.741GiB final footprint, minimum available13.6GiB. The guard required40GiB
starting availability and capped footprint/RSS36GiB. Private MLX cap34GiB and
internal34GiB early stop remained active. Production was not modified.

The complete trace contains159685 access/boundary events. A replay of its actual
224+8 layout matched **14688 misses and14304 evictions exactly**. This validates
the event/lease accounting before counterfactual policies are compared.

## Method

Rank experts using only the initial seed decode and prompt0 decode:61440 routed
selections total. Prompt1 (Python) and prompt2 (Chinese), including their PP and
decode, are held out:76389 cache requests, of which61440 are decode selections
and14949 are grouped PP requests. PP counts one request per expert group, not
each token's routing multiplicity. Rankings use decode counts only.

Caches respect GPU lease boundaries. Dynamic entries use LRU; hot entries are
preloaded and pinned in the replay. Global budgets borrow across layers;
layer8 reserves eight dynamic entries per layer. Each packed expert is2764800
bytes. Capacity rounds down to whole experts and does not include allocator or
mapping overhead. This simulates placement, not a functioning global GPU pool.

Two held-out starts are reported in the JSON: retaining cache state after replay
of calibration phases, and starting held-out phases with an empty dynamic cache.
The latter still assumes the hot set is already loaded: preload I/O is not free.
No held-out counts influence hot-set selection. Ties among unobserved experts
use deterministic key order; this tiny calibration is not a universally best list.

## Held-out results after preceding use

| Policy | Expert capacity GiB | PP misses | Decode misses | Total misses |
| --- | ---: | ---: | ---: | ---: |
| Numeric256 hot +8/layer |32.629|1398|1794|3192|
| Numeric256 hot +global384 |32.629|1387|1745|3132|
| Ranked256 hot +8/layer |32.629|695|499|1194|
| Ranked256 hot +global384 |32.629|617|472|1089|
| Ranked16GiB hot +8GiB dynamic |23.996|2919|1851|4770|
| Ranked16GiB hot +12GiB dynamic |27.997|1383|779|2162|
| Pure global LRU24GiB |23.998|2554|1555|4109|
| Pure global LRU28GiB |28.000|1384|687|2071|

Changing the hot list alone reduced misses62.6% at unchanged capacity. Simply
making the old tiny dynamic cache global improved only1.9%. Therefore the
uncalibrated numeric hot list was a much bigger issue on this trace.

16+12GiB used4.63GiB less expert capacity than numeric256+8 and reduced total
misses32.3%; its decode misses fell56.6%, while PP misses were almost unchanged.
16+8GiB substantially increased PP misses. The extra4GiB dynamic capacity matters.
Pure dynamic28GiB was slightly better than16+12 after warmup: do not assume a
permanently pinned16GiB set must be optimal.

## Empty dynamic-cache held-out start

| Policy (hot set already loaded) | Total misses |
| --- | ---: |
| Numeric256 +8/layer |3277|
| Ranked256 +8/layer |1267|
| Ranked16+8GiB |6010|
| Ranked16+12GiB |4881|
| Pure dynamic24GiB |10712|
| Pure dynamic28GiB |10415|

Warm-state gains must not be presented as cold-start gains. Pure dynamic pools
must initially load all used experts; the pinned schemes have already paid for
loading their hot set outside this miss count. Compare total startup time in a
later live experiment before selecting a production policy.

## Per-layer evidence and next gate

Per-expert training/test frequency and actual capture miss/reload counts are
saved for all48x288 expert IDs, along with reproducible training rankings. Even
after ranking, held-out requests to the coldest32 vary:22/1280 at layers14/30,
versus178/1280 at layer47. A fixed cold quota does not mean equally cold behavior
in every layer. Workload drift needs promotion/demotion, not permanent pinning
based on this small sample.

The most promising memory-focused live candidates are16+12GiB and a flexible
28GiB global pool; ranked256 is a useful lower-risk control. Global pool binding,
dynamic slot ownership and fast PP integration remain unimplemented. These
miss counts are not physical SSD reads, measured tok/s or measured process peaks
for those candidates. The operating system may serve a file read from RAM.

Tools: `--routing-study NEW_TRACE.tsv` on the guarded probe,
`devtools/expert_trace_study.py TRACE --output NEW_RESULT.json --expected-misses 14688 --expected-evictions 14304`.
The expected counts must come from the matching capture, not a different run.
The script refuses existing output paths and mismatched live replay counts.
Validation: fresh Release build; all five CTest suites passed (including four
replay unit cases); fixed-slot pin/refusal/reuse and layer0/47 rows1/4/32/64
parity passed. The complete trace passed phase/count checks and live-cache
reconciliation; the interrupted trace was explicitly rejected as incomplete.
