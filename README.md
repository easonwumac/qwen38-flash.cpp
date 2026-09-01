# qwen38-flash.cpp

`qwen38-flash.cpp` is an independent, server-only inference engine being built
specifically for Qwen3.8 Flash Next on Apple Silicon.

The product goal is to own the complete path from model weights to API tokens:
model loading, execution scheduling, quantized kernels, hybrid attention state,
KV/prefix caches, speculative verification, serving, and observability. It is not
a UI and is not a wrapper around `mlx-serve`.

> [!IMPORTANT]
> The current milestone performs real end-to-end greedy generation through all
> 48 layers, runs transactional depth-2 MTP when a companion is present, and
> serves OpenAI-style completion APIs. It is not yet the optimized release: the
> controlled decode gates remain unmet, streaming and chunked prefill are
> outstanding, and QSA contexts above the 2,048-token selection budget remain
> unsupported.

The execution path now includes a real-checkpoint Q4 affine projection smoke:
`qwen38-qmm-smoke MODEL_DIRECTORY` loads the retained `lm_head` through MLX's
lazy safetensors primitives and evaluates its 248,320 logits on the Apple GPU.
This proves weight-to-Metal plumbing, not a complete transformer forward.

`qwen38-hyper-connection-smoke MODEL_DIRECTORY` additionally executes the real
layer-0 read/injection gates, residual write-back, and final mixer. Its outputs
match the independent `devtools/hyper_connection_oracle.py` MLX-Python oracle.
`qwen38-sparse-moe-smoke MODEL_DIRECTORY` verifies router top-k, ten selected
Q4 experts, and the gated shared expert against a second independent oracle.
`qwen38-gated-delta-net-smoke MODEL_DIRECTORY` verifies two stateful decode steps,
including depthwise convolution and recurrent caches, forget/beta gates, Q/K
normalization, gated RMSNorm, and Q4 output projection against a third oracle.
`qwen38-self-attention-smoke MODEL_DIRECTORY` verifies two cached layer-3
full-attention steps, including partial RoPE, grouped-query expansion, and the
query output gate. QSA contexts above its selection budget are not implemented.
`qwen38-ngram-smoke MODEL_DIRECTORY` validates PLE hash rows and the low-memory
SSD AoS row gather against the original 30 GB safetensors table.

## Build

Requirements: CMake 3.24 or newer and a C++20 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The optional correctness/Metal backend uses the MLX C API but keeps the model
graph, decode loop, caches, scheduling, and server in this project. Point CMake
at a matching MLX C header and library installation:

```bash
cmake -S . -B build-mlx -DCMAKE_BUILD_TYPE=Release \
  -DQWEN38_MLXC_INCLUDE_DIR=/path/to/mlx-c/include \
  -DQWEN38_MLX_LIBRARY_DIR=/path/to/mlx/lib
cmake --build build-mlx --parallel
ctest --test-dir build-mlx --output-on-failure
./build-mlx/qwen38-mlx-smoke
```

The two paths must come from compatible `mlx`/`mlx-c` revisions. They are not
downloaded implicitly, so offline and reproducible builds do not silently change
the compute backend.

Build the native tokenizer with `utf8proc` for Unicode NFC/category parity:

```bash
cmake -S . -B build-tokenizer -DQWEN38_ENABLE_TOKENIZER=ON \
  -DQWEN38_UTF8PROC_INCLUDE_DIR=/path/to/include \
  -DQWEN38_UTF8PROC_LIBRARY_DIR=/path/to/lib
cmake --build build-tokenizer --parallel
./build-tokenizer/qwen38-tokenize /path/to/model "Hello, world!"
```

The tokenizer implements Qwen's Unicode-aware pre-split, byte-level mapping,
BPE merges, NFC normalization, special tokens, and decoding in C++. Real-model
parity fixtures cover English, CJK, Korean, code, whitespace, contractions,
emoji, CRLF, combining marks, and chat control tokens.

The core also implements the checkpoint's text chat-template behavior for system,
user, and assistant turns, including default/low/medium reasoning controls,
thinking preservation, and generation prompts. Tool and multimodal content remain
explicitly unsupported until their exact template branches land.

With both MLX and tokenizer options enabled, run the native inference server:

```bash
./build/qwen38-server --host 127.0.0.1 --port 11438 --model /path/to/model \
  --prefill-chunk 64 --prefix-cache-tokens 8192 --mtp-depth auto
curl http://127.0.0.1:11438/healthz
curl http://127.0.0.1:11438/v1/status
curl http://127.0.0.1:11438/metrics
curl -X POST http://127.0.0.1:11438/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":1}'
```

`--mtp-depth auto` is the default. When the model index has the Qwen3.8 MTP
companion, short prompts start at depth 2 for an eight-round probe and promote
to depth 3 only after at least 10 accepted drafts. A promoted request is checked
in 12-round windows and permanently demoted to depth 2 below 50% per-draft
acceptance. Prompts longer than 2,048 tokens stay at depth 2. Without an MTP
companion auto mode stays serial. `3` uses the same adaptive cap explicitly;
`off`, `2`, and `4` are fixed alternatives. On a 64 GB Mac, auto mode also
selects the verified
`12:24` resident-expert safety tier unless `QWEN38_RESIDENT_EXPERT_RANGE` is
already set, caps the MLX cache at 256 MiB, and falls back to serial after two
consecutive zero-accept rounds.

`--prefill-chunk 64` is the default layer-major prompt path. It bounds the
temporary prompt batch while preserving the retained production numerics.
Values through 256 are accepted when `QWEN38_GDN_METAL_PREFILL=1` enables the
oMLX-derived whole-sequence GDN recurrence; wider chunks otherwise fail at
startup instead of failing partway through a request.

The server retains one complete target/MTP prefix snapshot up to
`--prefix-cache-tokens` (default 8192). This deliberately caches every recurrent,
PLE, QSA, and KV state family together. Set the limit to `0` to disable it;
`POST /admin/cache/clear` releases the snapshot and the MLX allocator cache.

For Q4/gs64 REAP-288 developer testing, `QWEN38_FUSED_MOE=1` enables the
attributed two-dispatch selected-MoE Metal path. Add `QWEN38_DEVICE_ROUTER=1`
to keep top-k selection and routing weights on the GPU. `QWEN38_COMPILE_LAYER=1`
compiles each non-PLE linear decoder layer after its first stateful step. Together
with the resident tier below, these form the current fastest verified path.

MTP verification evaluates the 48-layer lazy graph in three 16-layer groups by
default. This preserves the retained S-row numerics while removing most
per-layer synchronization overhead. `QWEN38_VERIFY_BARRIER_STRIDE=1` is the
diagnostic rollback; accepted experimental values are 1 through 48.

Accepted MTP rows are committed back into the drafter with one S=2..4 batched
state pass instead of serially replaying each row. This changes proposal-state
scheduling only; target verification remains authoritative. Set
`QWEN38_BATCH_MTP_COMMIT=0` for the serial diagnostic rollback.

On unified-memory Macs, run full-model experiments through
`devtools/memory_guard.py -- COMMAND`. Full-model smoke and benchmark binaries
refuse unguarded execution. The default guard refuses to start below
42 GiB reclaimable memory and terminates the process group plus recursively
spawned descendants before they exceed 38 GiB aggregate RSS or leave less than
8 GiB reclaimable. A system-wide lock prevents overlapping guarded model runs.

For per-token warmup evidence rather than a two-token snapshot:

```bash
./devtools/memory_guard.py -- ./build/qwen38-model-bench /path/to/model 8
```

On the verified 64 GB configuration, `QWEN38_RESIDENT_EXPERT_RANGE=12:34`
pins a bounded 22-layer expert tier. It is capped at 22 layers in code and must
be used with the memory guard during experiments. With fused MoE and device
routing, the measured peak was 35.9 GiB RSS, outputs were unchanged, and warm
decode reached about 27 tok/s. The MLX-style Q4 down-projection kernel further
reduced the verified peak to 34.3 GiB and reached 27.4--28.9 tok/s warm.
Adding decoder-layer compilation reached 31.0--31.3 tok/s at 35.6 GiB peak RSS.

Validate a model manifest and lazily map one tensor without loading all weights:

```bash
./build/qwen38-inspect /path/to/Qwen3.8-Flash-Next \
  --tensor language_model.model.embed_tokens.weight
```

## API direction

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /healthz` and `GET /readyz`
- `GET /v1/status`
- `GET /metrics`
- `POST /admin/cache/clear`

See [docs/api.md](docs/api.md) for current behavior.

## Architecture

- C++20 owns the runtime, inference loop, scheduler, caches, and server.
- Objective-C++ and Metal implement Apple Silicon acceleration.
- Python is limited to offline weight conversion, evaluation, and development
  utilities; it is never on the serving path.
- Backends sit behind narrow interfaces so correctness can be established before
  specialized Metal kernels replace reference operations.

See [docs/architecture.md](docs/architecture.md) and the
[benchmark contract](docs/benchmark-contract.md). Historical optimization
results are classified in the
[prior-research ledger](docs/prior-research-ledger.md); rejected experiments
must not be repeated without a materially different hypothesis.

## Target gates

On the retained reference Mac and exact retained Qwen3.8 Flash Next checkpoint:

- at least 45 tok/s exact non-MTP decode;
- at least 65 tok/s MTP decode on controlled mixed workloads;
- no quality regression relative to the retained reference path;
- prefix cache and long-context operation up to 262,144 tokens where memory
  permits;
- matching or lower runtime memory under identical conditions.

These are acceptance targets, not current results. Every future performance
claim must include the complete benchmark controls described in the contract.

## License

Apache-2.0.
