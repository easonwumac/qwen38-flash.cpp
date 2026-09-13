# Q8 KV cache

## Status

The production runtime has an opt-in affine Q8 KV store for QSA-active long
contexts. It quantizes the complete historical K/V state in 64-channel groups,
discards the BF16 copy, appends later chunks directly in packed form, and
dequantizes only QSA-selected values inside the Metal attention kernel.

Use `--kv-cache q8`. The default activation threshold is 65,536 tokens and can
be changed with `--kv-q8-min-tokens`; BF16 remains the default. The threshold
cannot be below 2,049 because the packed path requires active QSA selection.

## Retained measurements

Environment: Qwen3.8-Flash-Next-REAP-288-MTP-Q8 target, affine Q4/group-64
weights, exact top-10 routing, lossless13 qmeta, MTP and prefix cache off,
temperature 0, thinking off, MLX 0.32.2, Apple M5 Pro MacBook Pro with 18 CPU
cores and 64 GiB unified memory, macOS 26.5 (25F71), AC power, one guarded
server process, and no recorded thermal warning.

- At 8,216 prompt tokens with forced 2,049-token activation and 512-row chunks,
  logical K/V storage fell from 201,850,880 bytes BF16 to 107,268,096 bytes Q8,
  exactly 46.875%. Both selected token 248046. Single cold diagnostic runs were
  302.04 PP/s and 98.68 ms decode for BF16 versus 269.42 PP/s and 116.54 ms for
  Q8. This deliberately early activation is a correctness/storage stress, not
  the recommended policy.
- At 131,140 prompt tokens, threshold 65,536 and fixed 512-row chunks recovered
  the expected `V1-NEBULA-128` needle. The single capacity run reached 293.62
  PP/s and 6.68 decode tok/s for 15 generated tokens. The earlier BF16 capacity
  result was 214.22 PP/s and 8.67 decode tok/s, but used a different conservative
  chunk policy, so the difference is an end-to-end policy result rather than an
  isolated KV-format A/B.

Q8 therefore provides real capacity headroom and permits a wider long-context
prefill batch, but it is not yet a universal speed win. It remains opt-in until
decode and a clean 192K guarded run pass promotion gates.

## Rejected implementation

Gathering selected packed rows, reconstructing BF16 tensors, and then calling
stock SDPA raised the 8K next-token latency to 6,855 ms. That prototype was
removed. The retained Metal kernel reads each packed word once, expands four
channels into threadgroup memory, and performs online softmax without creating
a complete BF16 KV cache.

Q8 prefix states retain the quantized arrays and format marker in both RAM
snapshots and SSD safetensors. The loader remains backward-compatible with
existing BF16 prefix-state files.
