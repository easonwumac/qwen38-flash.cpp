# Model capability contract

The v1 serving contract is validated on REAP-288 Q4. Other Qwen3.8 Flash Next
checkpoints must be classified from configuration and tensor metadata before any
weights are executed. A shared architecture name is not sufficient evidence of
runtime compatibility.

## Capability record

A future loader revision should derive one immutable record containing:

- hidden width, layer count, HC stream count/rank, vocabulary and context limit;
- per-layer attention type and full QSA/GDN geometry;
- per-layer MoE kind: routed-plus-shared or shared-only;
- routed expert count, top-k and intermediate width for every routed layer;
- weight quantization per tensor family, including bit width, group size and
  affine/symmetric mode rather than one model-wide value;
- PLE layer IDs, n-gram geometry, physical storage layout, quantization, and any
  paired-table/healing-map metadata;
- MTP presence, layer/reap lineage, tensor layout and quantization;
- the exact optimized-kernel predicates and the generic fallback selected for
  decode, verifier, and prefill.

The record should be printed by `qwen38-inspect` and included in the SSD-cache
compatibility key and benchmark metadata.

## Known layouts

| Capability | REAP-288 v1 | Niwaki 99B | Niwaki 113B |
|---|---:|---:|---:|
| Layers | 48 | 48 | 48 |
| Routed experts | 288 | 512 | 512 |
| Top-k | 10 | 10 | 10 |
| Routed layers | 48 | 24 | 32 |
| Expert intermediate | 640 | 448 | 448 |
| Routed precision | affine Q4/g64 | Q3/g64 | Q3/g64 |
| Backbone | Q4 | Q4 | Q4 |
| PLE | external retained table | Q2/g128 | Q2/g128 |
| MTP | optional matching L47 sidecar | not claimed | not claimed |

The Niwaki dimensions and quality/size claims come from the checkpoint authors'
model cards. They remain external claims until reproduced by this project:

- <https://huggingface.co/neopolita/Qwen3.8-Flash-Next-99B-A5B-Niwaki-3bit-mlx>
- <https://huggingface.co/neopolita/Qwen3.8-Flash-Next-113B-A5B-Niwaki-3bit-mlx>

## Current incompatibilities

The manifest parser already reads many dimensions dynamically, and generic MLX
quantized matmul can represent several bit widths. Production acceleration is
more constrained:

- fixed expert regions and several caches assume 288 experts;
- the fused routed-MoE Metal kernels assume hidden 2560, intermediate 640,
  top-10, and Q4/g64;
- compact-qmeta kernels and sidecars encode the REAP projection geometry;
- the decoder currently constructs routed-plus-shared MoE tensors on every
  layer, so a shared-only layer cannot be inferred by missing tensors;
- PP route/reduce has an optimized hidden-2560/top-10 path;
- PLE loading assumes the v1 n-gram storage contract and has not validated
  Niwaki's Q2/g128 paired representation or healing metadata;
- an REAP-288 MTP companion must never be attached to a differently pruned
  target merely because hidden sizes match.

Consequently, neither Niwaki checkpoint is supported today. The loader must
reject it with a capability-specific explanation before model allocation rather
than reaching a missing-tensor error or an incompatible Metal kernel.

## Promotion sequence

1. Obtain an explicitly authorized local checkpoint or metadata-only fixture.
2. Add a manifest-only fixture that proves the full capability record and all
   fail-closed diagnostics without loading weights.
3. Implement shared-only decoder layers and dynamic per-layer tensor inventory.
4. Route Q3/g64 expert projections through a correctness-first generic MLX path;
   optimized REAP Q4 kernels must remain gated by their exact predicates.
5. Implement and independently verify Q2/g128 PLE lookup, paired storage and
   healing-map semantics against the checkpoint's own loader.
6. Pass component oracles, tokenizer/chat parity, full first-token parity,
   serial generation, tool calls, cancellation and long-context retrieval.
7. Measure cold/warm PP, decode distribution, footprint and page behavior under
   the same benchmark contract. Only then add a named optimized profile.

Niwaki support must not change the REAP-288 API semantics, stable defaults, cache
format, or release claims. A generic fallback is acceptable for correctness
bring-up but is not a performance promotion.
