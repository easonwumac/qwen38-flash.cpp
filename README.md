# qwen38-flash.cpp

`qwen38-flash.cpp` is an independent C++20/Metal inference server for Qwen3.8
Flash Next on Apple Silicon. It owns model loading, tokenization, the complete
48-layer forward pass, hybrid attention state, speculative verification,
caching, HTTP serving, streaming, and runtime observability.

The server has one automatic production configuration. It selects the validated
Q8 KV, sparse-attention, prefill, MTP, continuous-batching, and persistent-Metal
paths from the supplied model assets; there are no tuning profiles to choose.
When an exact lossless16 metadata sidecar is present, it is selected
automatically to keep the target plus Q8 drafter below the memory ceiling.

## Final retained model

- Package: `Qwen3.8-Flash-Next-REAP-288-MTP-Q8`
- Target: REAP-288 affine Q4/group-64, exact top-10 routing
- Optional drafter: Q8/group-64 layer-47 MTP sidecar
- Context declared by the model: 262,144 tokens
- Largest safe retrieval run proven on the 64 GB test Mac: 196,675 tokens
- N-gram storage: Q4 29.8 GiB, affine Q8 53.64 GiB, or BF16 95.37 GiB
  row-major AoS file read from SSD on demand
- Quantization metadata: lossless16 normally; lossless13 capacity sidecar
- Compute runtime: MLX 0.32.2 through a pinned MLX-C ABI

The production package does not need the original safetensors n-gram table.
`ngram_table.bin.aos`, `ngram_table.q8.aos`, and `ngram_table.bf16.aos` store
each requested row contiguously, so one bounded `pread` retrieves it without
loading the table into RAM.

### Niwaki fast path

The engine also loads the 99B and 113B Niwaki MLX checkpoints natively:
mixed Q3/Q4/Q8 quantization, checkpoint-declared shared-only layers, BF16 output-healing maps,
MLX/PyTorch PLE convolution layouts, and Niwaki's paired Q2 PLE shards are
detected from the checkpoint. The paired table is dequantized directly from the
original memory-mapped model shards, without conversion or a second weight copy.
The retained tokenizer is still required because the Niwaki packages omit
`merges.txt`:

```bash
./build/qwen38-server --model "$NIWAKI_MODEL_DIR" \
  --tokenizer-dir "$REAP_MODEL_DIR" \
  --max-generation-tokens 4096
```

`--ngram-table-dir "$REAP_MODEL_DIR"` remains an explicit higher-precision Q4
PLE override for controlled comparisons.

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
- **Decode-priority refill:** a full-width refill prefill yields at existing
  eight-layer evaluation barriers, so it preserves prompt arithmetic while
  active decode rows continue between layer groups.
- **Fused decode kernels:** Q4 routed MoE, device-side routing, Hyper-Connection,
  and Gated DeltaNet work are fused around the actual decode-width hot path.
- **Grouped prefill:** 1,024-row grouped QMM/SDPA execution raises exact 8K
  prefill above 750 tokens/s while adaptive chunks bound long-context memory.
- **Bounded graph lifetime:** temporary compact metadata survives only to the
  existing eight-layer barrier, improving scheduling without retaining every
  decoded metadata bank for the whole request.
- **Bounded long-context state:** QSA activation, adaptive prefill chunks, and
  slabbed Q8 KV are selected from context growth rather than a user profile.
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
- **Correct thinking sampling:** OpenAI-compatible temperature, top-p, top-k,
  and seed controls use device-side top-k selection and a bounded CPU draw.
  Sampling automatically bypasses greedy-only speculative and persistent paths.
- **Bounded thinking lifecycle:** thinking requests automatically use the
  checkpoint's sampled defaults and reserve final-answer capacity. If the
  reasoning budget or a premature EOS is reached before `</think>`, the engine
  inserts Qwen's early-stop instruction and continues from the same KV state.

## Final evaluation

Common environment: Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GB unified
memory, macOS 26.5 (25F71), AC power, no recorded thermal or performance
warning, temperature 0, thinking off, one server process, and guarded memory
measurement. The checkpoint is the exact REAP-288 Q4/group-64 target above.
Long-context distributions below use independent cold server starts.

| Workload | Configuration | Result |
|---|---|---:|
| Public IFBench, 300 prompts | official loose/strict scorer; REAP-288 Q4 + Q8 MTP; temperature 0, thinking off, max 4,096; serial 55-minute thermal soak | **39.67% loose / 34.67% strict** prompt accuracy; 0 errors; aggregate decode **37.46 tok/s**; 40.8 GiB peak footprint |
| Native concurrent IFBench, first 30 prompts | REAP-288 Q4 + Q8 SSD PLE; MTP/thinking off, temperature 0, max 4,096; official scorer | **50.00% strict/loose**, 15/30; 17/33 instruction checks; 0 errors |
| Public IFBench bounded-thinking, first 30 prompts | REAP-288 Q4 target; temperature 1, top-p .95, top-k 20, seed 0, xhigh, max 4,096; MTP off; serial run on the validation Mac | Q4/Q8/BF16 SSD PLE: **63.33% / 66.67% / 70.00%** strict/loose; median decode 35.46 / 36.07 / 37.25 tok/s |
| Public IFBench bounded-thinking, 10 stratified prompts | original indices 0,10,...,90; temperature 1, top-p .95, top-k 20, seed 0, 2,730-token thinking budget within max 4,096; MTP off; final-answer-only official scoring | Qwen3.8-27B Q4: **70%** strict/loose, 16.79 tok/s, 16.75 GB MLX peak; REAP-288 Q4 target + Q8 SSD PLE: **50%**, 36.86 tok/s, 38.88 GiB peak footprint |
| OpenAI HumanEval raw completion, 164 problems | REAP-288 Q4 target; greedy, max 512, one sample; sandboxed tests | Q4/Q8/BF16 SSD PLE with verified MTP: **77.44% / 80.49% / 81.10%**; Q8 PLE with MTP off: **81.10%**, 38.13 tok/s; Qwen3.8-27B affine Q4/group-64, MTP off: **80.49%** |
| OpenAI HumanEval, EvalPlus no-thinking chat, 164 problems | self-contained-script prompt and assistant prefill; greedy, max 768, one sample; EvalPlus 0.3.1 sanitizer; sandboxed tests; MTP off | REAP-288 Q4 target + Q8 SSD PLE: **149/164 (90.85%)**, 38.94 tok/s; Qwen3.8-27B Q4: **150/164 (91.46%)**, 17.21 tok/s and 17.29 GB MLX peak |
| REAP-288 bounded-thinking IFBench pilot, 3 prompts | corrected sampled xhigh thinking; automatic 2,730-token reasoning budget within max 4,096; MTP off | **2/3 strict and loose**; all three forced a close and returned a final answer; **35.81 aggregate decode tok/s**; 39.0 GiB peak footprint |
| Niwaki 113B bounded-thinking IFBench pilot, 3 prompts | 113B routed/backbone weights + REAP tokenizer/Q4 SSD PLE; same protocol; MTP off | **0/3 strict and loose** despite three final answers; **40.07 aggregate decode tok/s**; 26.7 GiB peak footprint |
| Niwaki 113B native-PLE bounded-thinking pilot, 3 prompts | native paired Q2/group-128 PLE mmap + REAP tokenizer; temperature 1, top-p .95, top-k 20, seed 0, xhigh bounded thinking, 105--188 prompt tokens, max 4,096; MTP off | **0/3 explicit keyword gates**; **37.76 aggregate decode tok/s** (36.65--38.70); 26.6 GiB peak footprint |
| Niwaki 113B stock control, first pilot prompt | stock `mlx-vlm` 0.7.0/MLX 0.32.2 and native 2-bit PLE; sampled xhigh thinking, max 4,096 | no `</think>` or final answer; 29.13 tok/s; 43.58 GB MLX peak / 41.3 GiB guarded footprint |
| Serial decode, retained 128-token fixture | `speed`, MTP off; 1 warmup + 3 samples | 41.03 / 41.18 / 41.06 tok/s; median 41.06 |
| Four-request continuous decode, 4 x 128 tokens | REAP-288 Q4 + Q8 SSD PLE; `speed`, MTP/thinking off, greedy; warm HTTP A/B on four short prompts | **45.70 aggregate decode tok/s** vs 41.5--41.7 serial; exact per-request output parity; 38.9 GiB peak footprint |
| Rolling IFBench concurrency, first 12 prompts | same REAP/Q8 PLE; 512-token cap, MTP/thinking off; rolling four-slot vs serial A/B | **39.20 vs 38.35 aggregate decode tok/s**; end-to-end 36.10 vs 36.34 tok/s; 12/12 byte-identical responses; 39.0 GiB peak footprint |
| Layer-yield refill, first 12 IFBench prompts | Apple M5 Pro 64 GiB; REAP-288 Q4 + Q8 SSD PLE; automatic tuning, four rolling slots, MTP/thinking off, greedy, max 512; one run without active thermal control | **39.28 aggregate decode tok/s**, 36.17 end-to-end; 3,665 tokens, 0 errors; 12/12 responses byte-identical to the uninterrupted-prefill control; 39.1 GiB peak footprint |
| Automatic MTP, retained 128-token fixture | Apple M5 Pro 64 GiB; REAP-288 Q4 target + Q8 drafter/Q8 SSD PLE; automatically selected lossless16 qmeta and resident layers 12:29; greedy/no-thinking; 1 warmup + 2 samples; no active thermal control | **71.08 / 71.04 tok/s**, median 71.06; 92/100 drafts accepted per sample; **38.3 GiB peak footprint** |
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
The non-thinking IFBench result is not representative of this checkpoint's best
instruction-following path: bounded thinking recovered 63.33% on the first 30
public prompts. HumanEval is strongly protocol-sensitive: raw completion puts
REAP Q8 and Qwen3.8-27B Q4 at the same 80.49%, while the standard EvalPlus
no-thinking chat scaffold raises them to 90.85% and 91.46%. The one-problem gap,
plus the earlier 1.83-point stock-runtime control on the same REAP checkpoint,
excludes this backend as the source of the apparent large quality collapse. The
direct 27B reference runner supports continuous batching; a four-request,
64-token smoke test reached 50.22 aggregate tok/s at a 17.68 GB MLX peak, but is
not reported as single-stream decode or benchmark accuracy. The
public results and comparability limits are documented in the
[public quality evaluation](docs/public-quality-evaluation.md).

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
- The executor automatically coalesces up to four ordinary requests into one
  exact-arithmetic continuous decode batch and refills completed slots from the
  queue. Thinking requests and batches above the 131,072-token aggregate
  admission limit remain serial; MTP is bypassed while a cross-request batch is
  active. Mixed-length IFBench gained 2.23% in decode but not end-to-end
  throughput because refill prefill pauses active decode.
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

Normal automatic inference:

```bash
python3 devtools/memory_guard.py -- \
  ./build/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --prefix-cache-tokens 8192 \
  --max-generation-tokens 32768
```

The same automatic path handles long contexts. On a memory-constrained desktop,
the RAM prefix-cache limit can be set to zero:

```bash
python3 devtools/memory_guard.py --min-available-gib 8 -- \
  ./build/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --prefix-cache-tokens 0 \
  --max-generation-tokens 32768 --mtp-depth off
```

The automatic policy uses a compatible drafter only while acceptance repays
verification. `--mtp-depth off` is a resource-limit override for environments
that cannot afford its allocation.

The validated 128K recipe is:

```bash
QWEN38_QSA_RAW_WINDOW=64 QWEN38_RESIDENT_EXPERT_RANGE= \
python3 devtools/memory_guard.py --min-available-gib 6 -- \
  ./build/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR" \
  --mtp-depth off --prefix-cache-tokens 0 \
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
- One thread-affine model serves up to four independent request states through
  layer-major continuous decode; HTTP health/status work remains concurrent and
  bounded. Single requests retain the normal serial/MTP path.
- The release suite covers unit tests, real tokenizer fixtures, MLX ABI,
  component parity, full generation, API/tool/SSE behavior, cancellation,
  malformed requests, cache restart/clear, retrieval, and soak behavior.

Design details are in [architecture](docs/architecture.md). Capability gates for
other Flash Next weight layouts are in [model capabilities](docs/model-capabilities.md).

## License

Apache-2.0. Third-party notices are retained in `NOTICE` and `licenses/`.
