# qwen38-flash.cpp

`qwen38-flash.cpp` is an independent, server-only inference engine being built
specifically for Qwen3.8 Flash Next on Apple Silicon.

The product goal is to own the complete path from model weights to API tokens:
model loading, execution scheduling, quantized kernels, hybrid attention state,
KV/prefix caches, speculative verification, serving, and observability. It is not
a UI and is not a wrapper around `mlx-serve`.

> [!IMPORTANT]
> The current milestone includes the tested C++20 server foundation plus a strict
> Qwen3.8 manifest parser and lazy, zero-copy safetensors shard mapping. Health,
> readiness, status, model-list, metrics, and placeholder inference routes work.
> Graph execution and inference are not implemented yet, so this version is not a
> usable LLM runtime and makes no performance claim.

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

Run the server:

```bash
./build/qwen38-server --host 127.0.0.1 --port 11438
curl http://127.0.0.1:11438/healthz
curl http://127.0.0.1:11438/v1/status
curl http://127.0.0.1:11438/metrics
```

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
[benchmark contract](docs/benchmark-contract.md).

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
