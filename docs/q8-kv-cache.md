# Q8 KV cache

## Status

The production runtime has an opt-in affine Q8 KV store for QSA-active long
contexts. It quantizes historical K/V in 64-channel groups while retaining a
bounded BF16 hot tail. The mixed Metal attention kernel reads both stores and
dequantizes only QSA-selected cold values.

Use `--kv-cache q8`. The default activation threshold is 65,536 tokens and can
be changed with `--kv-q8-min-tokens`; BF16 remains the default. The threshold
cannot be below 2,049 because the packed path requires active QSA selection.
The hot tail flushes into Q8 in 8,192-token slabs by default; use
`--kv-q8-flush-tokens` to change the interval. This removes the old per-chunk
packed-array concatenation from the critical path.

For high-throughput long prefill, `--qsa-shared-rows 4` runs QSA selection from
one representative query per four adjacent rows in a fused Metal kernel. The
chosen blocks are expanded back to all rows before per-row causal validity,
local tail, and token expansion are applied. This approximates only the QSA
selector; it does not change model weights or top-10 expert routing. The default
is `1` (no row sharing). `--qsa-packed-min-tokens 32768` starts the packed
selector before Q8 KV itself activates.

## Retained measurements

Environment: Qwen3.8-Flash-Next-REAP-288-MTP-Q8 target, affine Q4/group-64
weights, exact top-10 routing, lossless13 qmeta, MTP and prefix cache off,
temperature 0, thinking off, MLX 0.32.2, Apple M5 Pro MacBook Pro with 18 CPU
cores and 64 GiB unified memory, macOS 26.5 (25F71), AC power, one guarded
server process, and no recorded thermal warning.

- At 16,451 prompt tokens with deliberately early 2,049-token activation,
  fixed 512-row chunks, and four layers of lossless qmeta cache, the expected
  `K7-MOON-491` needle was recovered at 500.47 PP/s and 23.89 decode tok/s.
  The same slabbed Q8 path without the four-layer qmeta cache reached 477.89
  PP/s. These are single cold policy diagnostics, not a distribution.
- At 131,140 prompt tokens, threshold 65,536, fixed 512-row chunks, an 8,192-token
  hot slab, eight cached lossless-qmeta layers, packed QSA from 32,768 tokens,
  and four-row selection recovered `V1-NEBULA-128` in all three independent cold
  runs. PP was 581.40/550.92/538.23 tok/s (median 550.92); decode was
  21.45/20.56/20.02 tok/s (median 20.56). Peak footprint was
  39.93/40.05/40.05 GiB (median 40.05), peak RSS 28.63/29.15/29.64 GiB, and
  minimum available memory 8.58/8.52/8.40 GiB.
- A same-build, same-prompt, same-selector BF16 KV control recovered the needle
  at 529.19 PP/s and 10.56 decode tok/s, with 41.51 GiB peak footprint and
  30.38 GiB peak RSS. Against that control, Q8 reduced median peak footprint by
  1.46 GiB (3.5%), raised median PP by 4.1%, and raised decode by 94.7%.
- A later bounded-raw-state development run retained only the QSA raw-key rows
  needed for pooling and rollback (`QWEN38_QSA_RAW_WINDOW=64`). It recovered
  `V1-NEBULA-128` at 590.69 PP/s and 26.11 serial decode tok/s, with 40.0 GiB
  peak footprint, 33.4 GiB peak RSS, and 8.4 GiB minimum available memory. The
  adjacent 65,601-token run reached 594.80 PP/s and 25.93 decode tok/s versus
  22.91 for its control. These are single cold development runs, not a new
  three-run product distribution.

Q8 therefore provides capacity headroom while preserving more than 500 PP/s at
128K on the test machine. It remains opt-in because four-row QSA selection is an
accuracy/throughput tradeoff and the faster policy has not yet been requalified
at 192K. A 1024-row outer batch regressed 128K to 277.55 PP/s; keep fixed 512.
Machine-readable results are in [q8-kv-128k-results.json](q8-kv-128k-results.json).

## Rejected implementation

Gathering selected packed rows, reconstructing BF16 tensors, and then calling
stock SDPA raised the 8K next-token latency to 6,855 ms. That prototype was
removed. The retained Metal kernel reads each packed word once, expands four
channels into threadgroup memory, and performs online softmax without creating
a complete BF16 KV cache.

Q8 prefix states retain the quantized cold arrays, BF16 hot tail, cold frontier,
and format marker in both RAM snapshots and SSD safetensors. The loader remains
backward-compatible with existing BF16 and all-cold Q8 prefix-state files.
