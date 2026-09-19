# VQ 3.2bpw hybrid and SSD-streaming qualification — 2026-09-20

Status: selective external-shard loading is retained as an opt-in checkpoint
capability. VQ 3.2bpw hybrid and SSD-streamed targets are rejected as the daily
model. VQ 2.1bpw v1 remains the default.

## Question and acceptance boundary

This experiment tested whether selected or fully streamed routed experts from
`TheDrainFlorist/Qwen3.8-Flash-Next-VQ-3.2bpw` could improve quality while
keeping the process footprint at or below 40 GiB. The base was VQ 2.1bpw v1;
the external native Q6 MTP/Q4 draft-head sidecar was also from v1. No REAP mask
was applied because capacity variants already fit and the retained REAP/HOPE
experiments had failed broader quality gates.

All measurements used an Apple M5 Pro with 18 CPU cores and 64 GiB unified
memory, macOS 26.5, MLX 0.32.2, AC power and no active thermal conditioning.
The short decode probe was greedy, fixed input token 9419, eight steps, first
two steps excluded from `steady_tps`, MTP off. HTTP quality used greedy,
no-thinking IFBench keys 20/70/100 and the official strict/loose scorer.

## Results

| Candidate | Decode / quality | Peak footprint | Decision |
|---|---:|---:|---|
| VQ 2.1bpw v1 control | 30.53 tok/s fixed-input | 36.3 GiB | Retain |
| One resident ordinary V3.2 d4/K2048/packed-11 layer (L30) | 29.27 tok/s | 36.5 GiB | No speed gain |
| The same L30 streamed through 128 MiB cache | 21.32 tok/s fixed-input | 35.7 GiB | Exact-capacity trade-off only |
| Six representative streamed layers, 1 GiB cache | 8.73 tok/s | 33.0 GiB | Scaling gate failed |
| Rejected fixed-slot pool for those six layers | 9.92 tok/s | 33.5 GiB | Only +13.7%; prototype removed |
| Six-layer stream plus depth-4 MTP | 6.80 tok/s rowwise verifier; 48/60 drafts accepted | 37.0 GiB | MTP could not amortize routing boundaries |
| Six-layer stream plus grouped depth-4 MTP | 5.23 tok/s; same 48/60 accepted | 37.1 GiB | Slower; prototype removed |
| One resident V3.2 down projection (L31) plus v1 MTP | IFBench 3/3; 24.54 aggregate decode tok/s | 39.9 GiB | Quality gate passed, but slower than control |
| VQ 2.1bpw paired MTP control | IFBench 3/3; 28.19 aggregate decode tok/s | 39.1 GiB | Better throughput and headroom |
| Four resident V3.2 d2 layers (31/35/36/39), target only | about 29 tok/s after warm-up | 38.8 GiB | Fits only without MTP |
| Four resident layers plus v1 MTP | stopped by guard | 40.8 GiB | Fails memory boundary |
| Four-layer target with L31/L35 streamed, 128 MiB cache, v1 MTP | 16.96 tok/s on the coding smoke; 34/40 drafts accepted | 39.7 GiB | Fits but is too slow |

The one-layer resident/streamed L30 runs produced the same eight greedy token
IDs under the paired non-fixed-input parity probe; steady throughput was 27.03
resident versus 19.44 tok/s streamed. Selective streaming therefore preserved
the tested trajectory; its rejection is performance-based, not a known
arithmetic error.

The three-case IFBench result is only the predeclared development signal. It is
not a full quality ranking. The down-only hybrid also changed completion length,
so its shorter wall time cannot be interpreted as a model-speed improvement;
aggregate generated-token throughput is the comparable number.

## Why full streaming does not scale in this backend

V3.2's ordinary routed layers use d4/K2048 with packed 11-bit indices; selected
late layers use d2/K256. Both geometries execute in the fused decode kernel.
The limiting operation is not just codebook lookup or SSD bandwidth.

For an exact streamed layer, the router must finish before the host knows which
expert rows to read. That inserts a GPU-to-CPU dependency at every streamed
layer. A bounded cache can remove some file reads, and a fixed slot pool can
remove per-token concatenation, but neither removes the sequential routing
boundary. Six streamed layers were already far below the resident control even
on a repeated fixed token with warm expert reuse. Expanding the same dataflow to
48 layers cannot plausibly recover daily-use target or MTP throughput.

Batching eight prompt rows reduced cold prompt time, but batching four MTP
verification rows made decode slower because the packed d4 target kernels lost
more efficiency than the shared routing boundary saved. Acceptance was
unchanged, isolating the regression to target execution cost.

## Retained implementation and stopping rule

The retained code can build lightweight manifests whose tensor map references
explicit relative external shards, optionally select whole layers or only
gate/up/down projection sets, and mark selected layers for a bounded expert
cache. Paged layers are excluded from MLX whole-layer compilation because host
I/O is illegal inside that transformation. Prompt rows are grouped in blocks of
up to eight; decode and learned-MTP verification remain rowwise because the
paired grouped verifier was slower.

The full V3.2 download was stopped after the six-layer scaling gate failed. Do
not resume it or add REAP to this path unless a new design removes the per-layer
host routing dependency—for example, a lower-level execution graph that can
perform routing, cache-hit resolution and indirect expert dispatch entirely on
the GPU. Merely increasing the cache, changing packing, or pruning the expert
pool does not address the measured critical path.

Machine-readable measurements: [results JSON](results/vq32-hybrid-streaming-2026-09-20.json).
