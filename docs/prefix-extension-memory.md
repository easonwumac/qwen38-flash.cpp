# SSD-backed prefix extension ownership

The fast resident engine remains the product path. No expert paging, new
quantization, routing change, compute kernel change, or new user setting is
introduced by this change.

## Gap and change

Newly computed prompts already released their two RAM checkpoint owners after
a successful SSD save. A matching checkpoint used to extend a conversation did
not: the live state was snapshotted but the old RAM cache continued pinning its
backing allocations throughout prefill.

Each RAM entry now records whether that exact checkpoint was successfully loaded
or saved to the SSD store. After copying all continuation state (including pending
MTP rows and the previous target stream), an extending request releases the old
RAM owner only when it is SSD-backed. It does not add a disk write or reload.
Exact hits and RAM-only caches are unchanged. New entries default to unbacked;
refused or throwing saves do not authorize release. The old SSD checkpoint is
available for retry after cancellation/failure, subject to the existing store's
eviction rules. This is not an fsync durability promise.

## Bounded evidence, 2026-09-06

M5 Pro, 64 GiB. Model-free synthetic complete target/MTP state: 48 target layers,
BF16 KV/QSA/GDN/PLE arrays, shared 32,768-token snapshot extended to 65,536 tokens
in 128 chunks of 256. Both arms perform the same save/load roundtrip before the
extension. MLX allocator cache is 256 MiB; private strict allocator limit 3.5 GiB;
guard process cap 4 GiB and available-memory floor 12 GiB. No other GPU job was
launched concurrently. No thermal pinning or SSD cold-cache claim.

| Metric | Retain checkpoint | Release checkpoint |
| --- | ---: | ---: |
| Final physical footprint, median of 3 | 3.248 GiB | 2.253 GiB |
| Sampled guard peak across runs | 3.3–3.4 GiB | 2.3–2.4 GiB |
| State checks | 3/3 passed | 3/3 passed |

Median final reduction is **0.995 GiB**. Checks include layer/timeline/shape/dtype
and exact BF16 prefix/tail samples; they are not exhaustive array comparisons
or full-model quality tests. This probe is not PP throughput: it advances state
arrays without model computation. Do not subtract this reduction from a prior
cache-disabled model footprint. It does not reduce the model weight floor.

Run the `qwen38-prefix-memory-probe` target with `extension-baseline` and
`extension-release` under `devtools/memory_guard.py` and the strict MLX runtime.
The earlier `shared-*` 16-token probe modes remain available. Probe-owned
temporary checkpoint files are automatically removed at process exit.

Fresh Release build passed. All 5 CTest suites passed, including existing store
success/refusal/failure and state serialization tests. These do not execute the
full NativeEngine request/cancellation sequence.

Full-model validation is pending: the non-mutating launch-headroom check refused
at 38.9 GiB available versus the 44 GiB start threshold. No guard thresholds were
relaxed, model loaded, service replaced, or full-model speed/quality claim made.
Next gate: same resident lossless16/top10/MTP recipe, paired 32K→64K request
extension, cached-token counts, output parity, TTFT/PP/decode, physical footprint,
and retry after cancellation. Also verify failed SSD writes preserve RAM hits.

## Other audited candidate

The loader retains safetensors shard owners, but many tensors remain lazy and
compact-qmeta projections still retain scale/bias handles for their geometry.
No physical-memory saving has been demonstrated from dropping shard maps, so no
loader lifetime change was promoted based on object counts alone.
