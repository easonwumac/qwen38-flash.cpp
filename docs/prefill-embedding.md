# Batched prefill embedding

`QwenModel::prefill_chunk_batch` validates a chunk's token IDs, gathers its
affine-Q4 embedding rows once, dequantizes them as `[1,S,2560]`, and tiles the
four hyper-connection streams once. Decode retains its original single-token
embedding path. There is no profile switch or numerical-mode change.

The previous path independently gathered, dequantized, and tiled every token,
then built `[1,S,10240]` through a left-fold concatenate. At 512 rows the leaf
streams contain 10 MiB, while the 511 concatenate nodes cumulatively describe
2,689,576,960 bytes (2,564.98 MiB) of outputs. That number is logical cumulative
allocation traffic, not a claim that every output is physically live at once.
The batch graph describes a 0.625 MiB packed gather, 0.078125 MiB combined
scale/bias gathers, a 2.5 MiB embedding, and a 10 MiB stream.

## Controlled layer-0 result

The developer A/B used the real base affine-Q4/group-64 embedding and complete
layer 0 with exact top-10/full metadata on an Apple M5 Pro with 64 GiB RAM. The
256 MiB MLX allocator cap was set before any array or tensor-store allocation.
Each arm included input-graph construction through evaluation of the complete
layer output, convolution state, and recurrent state. Three warmups preceded
ten alternating samples.

| Rows and IDs | Token-loop median | Batch median | Ratio |
|---|---:|---:|---:|
| 128 repeated | 6.529 ms | 3.402 ms | 1.919x |
| 128 diverse deterministic | 9.496 ms | 6.337 ms | 1.499x |
| 512 repeated | 68.774 ms | 12.425 ms | 5.535x |
| 512 diverse deterministic | 72.515 ms | 15.790 ms | 4.592x |

All output and GDN-state comparisons were bit-exact (`max_abs=0`). The guarded
process peaked at 1.8 GiB physical footprint and 1.2 GiB RSS, with 32.4 GiB
minimum available memory. These are isolated layer-0 results using synthetic
token-ID cohorts, not a full-model, natural-prompt, 65K footprint, or prefill
throughput claim. Raw samples and machine-readable metadata are stored in
`../experiments/prefill-embedding-2026-09-06/result.json`.

Coordinator rerun using the production helper passed all exact comparisons.
Medians (loop/batch ms):128 repeated6.446/3.399,128 diverse9.509/6.416,
512 repeated68.200/12.309,512 diverse71.786/16.015. Guard peak remained
1.8GiB footprint/1.2GiB RSS with32.4GiB minimum available. Server and probe
builds passed; CTest passed three tests, with tokenizer fixture skipped.

## Developer reproduction

```sh
cmake -S . -B build-all
cmake --build build-all --target qwen38-prefill-embedding-ab --parallel 2
python3 devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 --interval .1 -- \
  build-all/qwen38-prefill-embedding-ab /path/to/model
```
