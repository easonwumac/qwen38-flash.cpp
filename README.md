# qwen38-flash.cpp

`qwen38-flash.cpp` is an independent C++20/Metal inference server for Qwen3.8
Flash Next on Apple Silicon. It owns model loading, tokenization, the complete
48-layer forward pass, hybrid attention state, speculative verification,
caching, HTTP serving, streaming, and runtime observability.

The default server configuration is intended for daily inference. It selects
the validated Q8 KV, sparse-attention, prefill, MTP, and persistent-Metal paths
from the supplied model assets; tuning profiles are not required.

## Final retained model

- Package: `Qwen3.8-Flash-Next-REAP-288-MTP-Q8`
- Target: REAP-288 affine Q4/group-64, exact top-10 routing
- Optional drafter: Q8/group-64 layer-47 MTP sidecar
- Context declared by the model: 262,144 tokens
- Largest safe retrieval run proven on the 64 GB test Mac: 196,675 tokens
- N-gram storage: 29.8 GiB row-major AoS file read from SSD on demand
- Quantization metadata: lossless16 normally; lossless13 capacity sidecar
- Compute runtime: MLX 0.32.2 through a pinned MLX-C ABI

The production package does not need the original 29.8 GiB safetensors n-gram
table. `ngram_table.bin.aos` stores each requested row contiguously, so one
bounded `pread` retrieves its packed weights, scales, and biases without loading
the table into RAM.

### Niwaki 99B fast path

The engine also loads `Qwen3.8-Flash-Next-99B-A5B-Niwaki-3bit-mlx` natively:
mixed Q3/Q4/Q8 quantization, 24 shared-only layers, BF16 output-healing maps,
and MLX/PyTorch PLE convolution layouts are detected from the checkpoint. It can
reuse the retained tokenizer and higher-precision SSD Q4 n-gram table without
copying or linking either asset:

```bash
./build/qwen38-server --model "$NIWAKI_MODEL_DIR" \
  --tokenizer-dir "$REAP_MODEL_DIR" --ngram-table-dir "$REAP_MODEL_DIR" \
  --max-generation-tokens 4096
```

Niwaki contains no MTP tensors. When the supplied n-gram/tokenizer package also
contains the compatible retained REAP Q8 drafter, the server detects it and
uses depth 4 automatically. Profitable requests stay on batched MTP; after two
zero-accept rounds an unprofitable request imports its committed state into the
persistent Metal backend and continues there. `--mtp-depth off` remains an
explicit resource-limit override when the drafter allocation is undesirable.

Niwaki's dense BF16 healing matrices are executed as deterministic rank-64
deltas. The persistent Metal path directly reuses MLX's tensor allocations, so
prefill and decode no longer maintain or page between duplicate weight
resources. Direct buffers are released during prefill, rebuilt from the already
materialized allocations at handoff, and warmed before streaming begins.

On the M5 Pro 64 GiB validation Mac, the 131,140-token needle recovered
`V1-NEBULA-128` at **610.74 PP tok/s and 41.60 decode tok/s** without MTP. The
observed peak was 42,204,886,920 bytes (**39.30 GiB**). This is the retained fast
configuration for Niwaki; the REAP-288 checkpoint remains the higher-quality
reference. Full conditions are in the
[bring-up](docs/niwaki-99b-bringup.md) and
[low-rank map report](docs/niwaki-lowrank-maps.md).

## What improved

- **Layer-major batched execution:** wide prompt and verifier rows pass through
  each layer together, removing token-by-token dispatch and synchronization.
- **Fused decode kernels:** Q4 routed MoE, device-side routing, Hyper-Connection,
  and Gated DeltaNet work are fused around the actual decode-width hot path.
- **Grouped prefill:** 1,024-row grouped QMM/SDPA execution raises exact 8K
  prefill above 750 tokens/s while adaptive chunks bound long-context memory.
- **Bounded graph lifetime:** temporary compact metadata survives only to the
  existing eight-layer barrier, improving scheduling without retaining every
  decoded metadata bank for the whole request.
- **Pageable long-context experts:** normal requests can keep a measured expert
  tier resident; capacity profiles let macOS reclaim those pages as QSA/KV state
  grows.
- **Exact Qwen Sparse Attention:** raw and pooled indexer state, causal top-block
  selection, snapshots, verifier checkpoints, and rollback remain native to the
  engine.
- **Fused long-context selection:** a Metal selector can reuse each representative
  query across four adjacent prefill rows, then restore per-row causal validity.
  Combined with slabbed Q8 KV, this keeps 128K prefill above 500 tok/s.
- **Transactional MTP:** batched verification and accepted-row commit are paired
  with adaptive depth and economic fallback. Serial remains the safe default.
- **Complete-state prefix caching:** target, recurrent, attention, QSA, and
  optional MTP state can be reused in RAM or restored from bounded SSD storage.
- **Thread-affine serving:** one accelerator executor owns MLX state while four
  bounded HTTP workers keep health and status endpoints responsive.

## Final evaluation

Common environment: Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GB unified
memory, macOS 26.5 (25F71), AC power, no recorded thermal or performance
warning, temperature 0, thinking off, one server process, and guarded memory
measurement. The checkpoint is the exact REAP-288 Q4/group-64 target above.
Long-context distributions below use independent cold server starts.

| Workload | Configuration | Result |
|---|---|---:|
| Serial decode, retained 128-token fixture | `speed`, MTP off; 1 warmup + 3 samples | 41.03 / 41.18 / 41.06 tok/s; median 41.06 |
| Serial decode, retained 256-token fixture | `speed`, MTP off; 1 warmup + 3 samples | 40.95 / 40.50 / 40.73 tok/s; median 40.73 |
| Exact 8K prefill, 8,216 tokens | `speed`, chunk 1024; cold + warm; fixed first-token hash | 608.50 cold, 757.18 warm PP tok/s |
| Exact 32K prefill, 32,792 tokens | `speed`, MTP off; chunk 512 A/B and fixed 1024 | 567.29 / 571.12 / 571.65 PP tok/s |
| Niwaki persistent Metal 128K needle, 131,140 tokens | Niwaki 99B Q3 routed/Q4 backbone, `speed`, Q8 KV at 8,192 with 2,048-token slabs, QSA budget 512, external REAP Q4 n-gram, greedy/no-thinking, MTP off, min output 8; one directional run | expected `V1-NEBULA-128` recovered; **610.74 PP tok/s; 41.60 decode tok/s; 39.30 GiB observed peak** |
| Niwaki + external REAP Q8 MTP, 16K coding fixture | `speed`, depth 4, Q8 KV, QSA decode budget 512, greedy/no-thinking, 128 output tokens; one directional run | 96/128 drafts accepted; **70.08 tok/s**; 872 PP tok/s |
| Zero-tuning automatic path, 16K favorable/losing pair | default CLI above, fresh server, greedy/no-thinking, 128 output tokens each; one directional pair | coding stayed on MTP at **62.63 tok/s** (96/128 accepted, 702 PP); repeat fell back after 0/8 accepted and completed on persistent Metal at **40.51 tok/s** (856 PP) |
| Niwaki + external REAP Q8 MTP, 128K coding fixture | same settings, 131,107 prompt tokens and 128 output tokens; one directional run | 96/128 accepted; **55.34 tok/s**; 587 PP tok/s; valid deterministic implementation |
| 128K needle retrieval, 131,140 tokens | `memory`, Q8 KV, shared-row QSA, MTP/prefix cache off; 3 cold runs; expected `V1-NEBULA-128` recovered | 581.40 / 550.92 / 538.23 PP tok/s; median **550.92**; median decode 20.56 tok/s; 40.0 GiB median peak footprint |
| 128K BF16 KV control, 131,140 tokens | same build, prompt and selector policy; one cold run; needle recovered | 529.19 PP tok/s; 10.56 decode tok/s; 41.5 GiB peak footprint |
| 192K needle retrieval, 196,675 tokens | `memory`, MTP/cache off; expected `S4-PULSAR-192` recovered | 281.17 PP tok/s; 4.19 decode tok/s; 42.14 GiB footprint, 26.93 GiB RSS |
| Generated-turn RAM prefix reuse | 3,923-token first turn, then 3,951-token appended turn | 3,925 cached; prompt 5,584.72 ms → 157.45 ms, about 35x |
| Q8 MTP, favorable 128-token fixture | explicit depth 4; 3 samples; 97/124 accepted | 65.48 / 65.65 / 65.78 tok/s |
| Q8 MTP, favorable 256-token fixture | explicit depth 4; 3 samples; 197/240 accepted | 62.32 / 66.39 / 66.55 tok/s |
| Mixed lifecycle soak | 256 valid requests, 64 malformed requests, SSE disconnect recovery | ended ready/idle; no monotonic footprint growth across 3 cleared cohorts |

The 128K and 192K results are retrieval/capacity runs, not short-context speed
claims. The 192K run left only 0.25 GiB above its 6 GiB safety floor. A 262K
attempt did not return a valid result, so 262K is not claimed as safe on this
checkpoint and machine.

The deterministic 30-request serial quality suite scored 22/30 with zero
request or parse errors, matching the retained model baseline. This measures the
checkpoint and request lifecycle; it is not a claim of universal model accuracy.

Full workload definitions, hashes, distributions, guard reports, and rejected
experiments remain in the [benchmark contract](docs/benchmark-contract.md) and
[prior-research ledger](docs/prior-research-ledger.md).

## Known limits

- The Niwaki persistent path reaches the 40 tok/s no-MTP long-context gate, but
  still needs a mixed-corpus quality comparison and repeated cold-run distribution.
- Exact 8K prefill exceeds 600 PP tok/s, but 32K remains around 572 PP tok/s.
- The validated 128K Q8 recipe exceeds 500 PP tok/s, but its four-row QSA
  selection is an explicit long-context approximation and remains opt-in.
- The 192K capacity result predates the shared-row QSA recipe; it has not yet
  been requalified with this faster policy.
- Automatic MTP can still lose on an individual prompt, so the runtime probes
  acceptance and switches losing requests to persistent Metal.
- Persistent Metal handles serial continuation after MTP fallback. Moving the
  profitable batched verifier itself onto that backend is still required for a
  stable 60 tok/s result at 128K.
- The external Q8 drafter now works with Niwaki and exceeds 60 tok/s at 16K,
  but the retained 128K directional result is 55.34 tok/s. A 60 tok/s 128K
  product claim is therefore not made.
- The Q8 drafter increases admission pressure. `--mtp-depth off` is retained as
  a resource-limit override for smaller machines.
- Multimodal input is not supported.
- The server currently exposes Chat Completions, not the Responses API required
  by current Codex custom providers.

## Build

Requirements: CMake 3.24+, a C++20 compiler, `utf8proc`, and matching MLX/MLX-C
headers and shared libraries.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQWEN38_ENABLE_TOKENIZER=ON \
  -DQWEN38_MLXC_INCLUDE_DIR="$MLXC_INCLUDE_DIR" \
  -DQWEN38_MLX_LIBRARY_DIR="$MLX_LIBRARY_DIR"
cmake --build build --parallel
QWEN38_TEST_MODEL="$MODEL_DIR" \
  ctest --test-dir build --output-on-failure
```

Python is used only for offline conversion, evaluation, and developer tools; it
is not part of the serving path.

## Run

Normal exact inference:

```bash
python3 devtools/memory_guard.py -- \
  ./build/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --profile speed --prefix-cache-tokens 8192 \
  --max-generation-tokens 32768 --mtp-depth off
```

For 128K+ contexts or a memory-constrained desktop, use pageable experts and no
RAM prefix cache:

```bash
python3 devtools/memory_guard.py --min-available-gib 8 -- \
  ./build/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --profile long-context --prefix-cache-tokens 0 \
  --max-generation-tokens 32768 --mtp-depth off
```

Use `--mtp-depth auto` only when sufficient memory is reclaimable. Explicit
depth 4 is reserved for a previously calibrated high-acceptance workload.

The validated 128K recipe is:

```bash
QWEN38_QSA_RAW_WINDOW=64 \
python3 devtools/memory_guard.py --min-available-gib 6 -- \
  ./build/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --profile memory --mtp-depth off --prefix-cache-tokens 0 \
  --qmeta-cache-max-prompt-tokens 262144 --qmeta-cache-layers 8 \
  --qsa-packed-min-tokens 32768 --qsa-shared-rows 4 \
  --kv-cache q8 --kv-q8-min-tokens 65536 --kv-q8-flush-tokens 8192 \
  --prefill-chunk 512 --prefill-chunk-fixed
```

See the [Q8 KV notes](docs/q8-kv-cache.md) for semantics and measurements.

See the [operations guide](docs/operations.md) for cache persistence, recovery,
benchmark commands, and guard exit codes.

## API

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /healthz` and `GET /readyz`
- `GET /v1/status`
- `GET /metrics`
- `POST /admin/cache/clear`

Chat completions support native SSE streaming, cancellation at committed-token
boundaries, OpenAI-style tool declarations/results, and usage/performance
telemetry. See [API details](docs/api.md).

## Architecture and verification

- C++20 owns the runtime, scheduling, caches, tokenizer, and HTTP server.
- Metal and MLX-C provide Apple Silicon execution.
- Model requests are serialized intentionally; HTTP health/status work remains
  concurrent and bounded.
- The release suite covers unit tests, real tokenizer fixtures, MLX ABI,
  component parity, full generation, API/tool/SSE behavior, cancellation,
  malformed requests, cache restart/clear, retrieval, and soak behavior.

Design details are in [architecture](docs/architecture.md). Capability gates for
other Flash Next weight layouts are in [model capabilities](docs/model-capabilities.md).

## License

Apache-2.0. Third-party notices are retained in `NOTICE` and `licenses/`.
