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
- At 131,140 prompt tokens, threshold 65,536, fixed 512-row chunks, the 8,192
  hot-tail slab, and four cached lossless-qmeta layers recovered the expected
  `V1-NEBULA-128` needle. The single guarded run reached 392.57 PP/s and 11.42
  decode tok/s, peaked at 40.4 GiB footprint, and retained 8.7 GiB minimum
  available memory. Relative to the predecessor's 293.62 PP/s and 6.68 decode
  tok/s, the complete policy improved PP by 33.7%; this is not an isolated
  format A/B because qmeta caching also changed.

Q8 therefore provides real capacity headroom and now preserves 500 PP/s at the
16K stress gate, but the 128K target remains unmet. A 1024-row outer batch
regressed the 128K run to 277.55 PP/s, and the speed profile reached only 365.71
PP/s; neither is retained as the recommended policy. Q8 remains opt-in until
128K decode and a clean 192K guarded run pass promotion gates.

## Rejected implementation

Gathering selected packed rows, reconstructing BF16 tensors, and then calling
stock SDPA raised the 8K next-token latency to 6,855 ms. That prototype was
removed. The retained Metal kernel reads each packed word once, expands four
channels into threadgroup memory, and performs online softmax without creating
a complete BF16 KV cache.

Q8 prefix states retain the quantized cold arrays, BF16 hot tail, cold frontier,
and format marker in both RAM snapshots and SSD safetensors. The loader remains
backward-compatible with existing BF16 and all-cold Q8 prefix-state files.
