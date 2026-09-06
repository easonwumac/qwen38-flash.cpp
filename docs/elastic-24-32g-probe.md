# Elastic 24 / 32 GiB research probe

This follows [the 24 GiB paging probe](paged-24g-probe.md). The server/default
loader remains unchanged. The user-requested range is total process memory:
approximately 24 GiB steady, at most 32 GiB during work, not 24/32 GiB of experts.

## Changes

- The experimental batch path leases all ten selected experts before building
  their ordered BF16 computation, then fences once per row/layer instead of once
  per expert. It keeps the reference router and shared-expert batch geometry.
- The pinned MLX-C `new_data` path synchronously copies host data (verified in
  the local pinned MLX/MLX-C sources). Batch mode drops per-field GPU fences and
  retains pre-load/accounting and bundle-completion fences. No managed alias or
  async host buffer lifetime is introduced. Reads are still serial, not async I/O.
- A fenced `set_expert_budget()` allows safe growth/shrink at probe boundaries.
  The planner uses a conservative 4 GiB non-expert/state allowance for this short
  workload, yielding a 20 GiB steady expert cache. The burst cache is capped at
  24 GiB, leaving 4 GiB inside the 28 GiB strict MLX allocation cap for other MLX
  buffers and another 4 GiB outside it under the process ceiling.
- During generation, eviction plus more than 48 misses in one 480-expert step
  permits one growth to the burst cache. After generation and short PP, the
  cache shrinks back to 20 GiB and the probe asserts footprint below 24 GiB.
  This is a bounded experimental policy, not continuous system-pressure control
  or a validated budget estimate for long context. The external headroom guard
  remains mandatory, and oversized requests fail rather than relaxing caps.

## Conditions

Apple M5 Pro / 64 GiB, macOS 26.5; retained
`Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47`.
No MTP, no pruning changes, no compact-qmeta conversion, no reduced top-k.
Synthetic greedy generation begins at token 9419, without chat template/EOS
stopping. PP is only eight tokens in a fresh state after generation, with the
same expert cache. These are feasibility microbenchmarks, not broad chat or
long-context throughput. Thermals and OS file-cache state were not controlled.

Guard: start headroom 40 GiB, available system memory minimum 12 GiB, sample
interval 0.1 seconds. The 24/32 probe has an external 32 GiB RSS/footprint ceiling,
28 GiB private strict MLX cap, zero free-block cache and 30 GiB internal early
stop at phase/token boundaries. The original 24 GiB comparison instead uses
20 GiB MLX / 24 GiB process caps. Sampled process limits are not hard allocation
guarantees; the strict MLX cap covers only MLX-managed allocations.

## Measurements (2026-09-06)

Same 64-token workload and 16 GiB expert allowance:

| Path | Decode excluding first | Step median / p95 | Eight-token PP | Peak footprint |
|---|---:|---:|---:|---:|
| Serial reference | 3.488 tok/s | 0.274 / 0.392 s | 3.832 tok/s | 19.3 GiB |
| Batch leases/fences | 5.472 tok/s | 0.170 / 0.273 s | 6.233 tok/s | 19.3 GiB |

Observed decode +56.9%, PP +62.6% in this pair, not a general speedup guarantee.
Both have 24,555 hits, 6,165 misses and 139 evictions. Token hash matches
`2496730379222366046`; full F32-converted PP logit hash matches
`12037094211002573605`. Loader timing was 5.938 s / 4.791 s respectively and
includes reads/copies/fences, not just SSD service time. Some file-cache effects
remain, although per-expert synchronization is also materially reduced.

Same 256-token workload, both using batch leases:

| Policy | Decode | Median / p95 | PP | Peak / restored footprint | Misses / evictions |
|---|---:|---:|---:|---:|---:|
| Fixed 20 GiB expert cache | 2.783 tok/s | 0.310 / 0.771 s | 3.241 tok/s | 23.3 / 23.3 GiB | 12,249 / 4,717 |
| Elastic 20 -> 24 GiB experts | 3.962 tok/s | 0.236 / 0.429 s | 3.495 tok/s | 27.3 / 23.3 GiB | 10,918 / 1,879 |

The elastic run borrowed at step 108, ended with 23.999 GiB experts and returned
to 19.998 GiB after PP. Footprint fell from 27.311 to 23.309 GiB. System headroom
stayed above 19.9 GiB. Fixed-cache headroom stayed above 21.4 GiB.

Both 256-token hashes match `12818014199420254704` and PP logit hashes match the
64-token cases. The reliable cache-policy comparison is 10.9% fewer misses and
hit rate 90.03% -> 91.11%. Do not attribute the whole measured timing difference
to extra RAM: fixed-cache model loading took 1.587 s versus 0.670 s for elastic,
and its initial expert reads were already slower *before* either cache filled.
That demonstrates a file-cache/I/O confound. Miss-loading totals were 53.499 s
and 29.106 s, respectively. None of these are controlled cold-SSD measurements.

Clean-build elastic rerun: 3.008 tok/s, median 0.296 s, p95 0.647 s, PP 3.459
tok/s. Peak footprint again 27.3 GiB; after shrinking 23.309 GiB (the explicit
24 GiB steady assertion passed), minimum system headroom 19.7 GiB. Hashes and
hit/miss/eviction counts were identical. This rerun loaded the model in 1.595 s
and accumulated 46.492 s in expert loading, illustrating why the earlier 3.962
tok/s should not be treated as guaranteed. Against the fixed-cache 2.783 tok/s
run the observed throughput gain is about 8%, not an established 42% improvement.
The higher memory allowance helps, but does not remove the serial I/O bottleneck.

Real layers 0/47 pass exact BF16 router/shared-branch comparison for 1/4 rows;
batch parity uses a 64 MiB cache with evictions while ten leases remain pinned.
This is stronger than token repeatability, but not a complete quality evaluation.
Fresh Release full build passed under a 4 GiB build guard (0.3 GiB footprint /
0.4 GiB RSS peak). CTest: three suites passed; tokenizer fixture suite skipped.
The clean-build layer parity checks passed before the final full-model rerun.

## Reproduce

```sh
python3 devtools/memory_guard.py --min-start-gib 40 --min-available-gib 12 \
  --max-rss-gib 32 --max-footprint-gib 32 --interval 0.1 -- \
  env MLX_STRICT_MEMORY_LIMIT=1 DYLD_LIBRARY_PATH=/path/to/strict-mlx/stage/lib \
  build-elastic-verify/qwen38-paged-model-probe /path/to/model --long --elastic
```

Add `--steady-only` to keep the 20 GiB expert budget. Use `--extended` instead
of `--long` for 64 tokens. Without `--elastic`, use the original 24 GiB process
guard and a 16 GiB expert cache. `--serial` retains the original synchronization
path for comparison; `--layer-parity` runs bounded layer comparison with a 4 GiB
external guard and 16 GiB startup headroom.

## Status

Elastic memory is demonstrated for this short workload, including actual growth,
eviction and shrink. Throughput remains far below the existing resident engine.
Next useful steps are batched/async SSD reads and selected-expert kernel packing,
then grouped paged PP. Long context, MTP, continuous adaptive pressure handling
and production server integration are still unverified.
