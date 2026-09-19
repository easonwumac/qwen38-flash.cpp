# Model capability contract

The production serving contract is validated on VQ-2.1bpw. REAP-288 and
Niwaki remain implemented historical/research layouts, not automatic fallback
models. Every Qwen3.8 Flash Next checkpoint must be classified from configuration
and tensor metadata before any weights are executed; a shared architecture name
is not sufficient evidence of runtime compatibility.

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

| Capability | VQ-2.1bpw | REAP-288 v1 | Niwaki 99B | Niwaki 113B |
|---|---:|---:|---:|---:|
| Layers | 48 | 48 | 48 | 48 |
| Routed experts | 512 | 288 | 512 | 512 |
| Top-k | 10 | 10 | 10 | 10 |
| Routed layers | 48 | 48 | 24 | 32 |
| Expert intermediate | 640 | 640 | 448 | 448 |
| Routed precision | packed-14 VQ, d2/d8 | affine Q4/g64 | Q3/g64 | Q3/g64 |
| Backbone | Q8 | Q4 | Q4 | Q4 |
| PLE | native VQ d8/K256 | external retained table | native Q2/g128 or external table | native Q2/g128 or external table |
| MTP | optional native Q6 | optional matching L47 sidecar | compatible external drafter tested | no qualified sidecar |
| Project status | production target | historical reference | research only | rejected by quality gate |

VQ v2 also supports d4/K256 packed-8 and mixed d8 gate/up + d4 down in the
automatic MLX/Metal path. It keeps the table's dimensions, expert count, routing,
backbone and PLE, but has a separate, pending qualification record. The opt-in
legacy persistent backend rejects its mixed geometry.
[Exact revision and upgrade checks](vq-v2-upgrade-2026-09-19.md).

The Niwaki dimensions and quality/size claims come from the checkpoint authors'
model cards. They remain external claims until reproduced by this project:

- <https://huggingface.co/neopolita/Qwen3.8-Flash-Next-99B-A5B-Niwaki-3bit-mlx>
- <https://huggingface.co/neopolita/Qwen3.8-Flash-Next-113B-A5B-Niwaki-3bit-mlx>

## Compatibility boundaries

The manifest parser reads the required dimensions dynamically, and generic MLX
quantized matmul represents several bit widths. Production acceleration remains
more constrained:

- each fused Metal kernel is gated by the exact VQ, REAP, or Niwaki tensor
  geometry it implements;
- compact-qmeta kernels and sidecars encode one projection family and cannot be
  reused merely because hidden sizes match;
- shared-only layers, healing maps, and paired PLE are Niwaki-specific and must
  remain outside the VQ automatic path;
- PP route/reduce has an optimized hidden-2560/top-10 path;
- an REAP-288 MTP companion must never be attached to a differently pruned
  target merely because hidden sizes match.

Niwaki checkpoints have completed research bring-up and performance probes, but
their quality gates did not justify promotion. They must not be advertised as
production alternatives or selected automatically. Unknown layouts must fail
with a capability-specific explanation before an incompatible kernel runs.

## Promotion sequence for another checkpoint

1. Add a manifest-only fixture that proves the full capability record and all
   fail-closed diagnostics without loading weights.
2. Route new expert projections through a correctness-first generic MLX path;
   optimized REAP Q4 kernels must remain gated by their exact predicates.
3. Implement and independently verify its PLE, attention, healing, and MTP
   semantics against the checkpoint's own reference loader.
4. Pass component oracles, tokenizer/chat parity, full first-token parity,
   serial generation, tool calls, cancellation and long-context retrieval.
5. Measure cold/warm PP, decode distribution, footprint and page behavior under
   the same benchmark contract. Only then consider it for the single automatic
   production path.

Research-model support must not change VQ API semantics, stable defaults, cache
formats, or release claims. A generic fallback is acceptable for correctness
bring-up but is not a performance or product promotion.
