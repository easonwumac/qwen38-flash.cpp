# qwen38-flash.cpp

`qwen38-flash.cpp` is an independent, server-only inference engine being built
specifically for Qwen3.8 Flash Next on Apple Silicon.

The product goal is to own the complete path from model weights to API tokens:
model loading, execution scheduling, quantized kernels, hybrid attention state,
KV/prefix caches, speculative verification, serving, and observability. It is not
a UI and is not a wrapper around `mlx-serve`.

> [!IMPORTANT]
> The current milestone performs real end-to-end greedy generation through all
> 48 layers, runs transactional adaptive depth-2/3 MTP when a companion is present, and
> serves OpenAI-style completion APIs with live SSE token streaming and bounded
> chunked prefill. Qwen Sparse Attention now owns raw/pooled indexer state,
> causal top-block selection, snapshots, and verifier rollback beyond the
> 2,048-token selection budget. It is not yet the final optimized release: the
> non-MTP and mixed-workload decode targets remain unmet, and 128K/262K context
> operation is not yet validated.

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
query output gate. Adding `--qsa` forwards the real indexer through the
2,052-token engagement boundary and checks batched-verifier output against the
serial path together with every rollback checkpoint's raw/pooled cache frontier.
`qwen38-ngram-smoke MODEL_DIRECTORY` validates PLE hash rows and the low-memory
SSD AoS row gather against the original 30 GB safetensors table.

`qwen38-long-context-smoke MODEL_DIRECTORY PROMPT.txt [CHUNK_ROWS [MAX_TOKENS]]`
is the guarded full-model QSA check. It requires `devtools/memory_guard.py`,
defaults to conservative 64-row chunks, and reports linear/full-attention time,
actual pooled blocks, prefill throughput, and one-token decode latency.

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
thinking preservation, generation prompts, OpenAI function declarations,
assistant `tool_calls`, and grouped tool-result turns. Canonical Qwen3.8 XML is
converted back to OpenAI tool-call objects for streaming and non-streaming
responses without exposing the XML. Multimodal content remains unsupported.

With both MLX and tokenizer options enabled, run the native inference server:

```bash
./build/qwen38-server --host 127.0.0.1 --port 11438 --model /path/to/model \
  --profile speed --prefix-cache-tokens 8192 --max-generation-tokens 4096 \
  --mtp-depth auto
curl http://127.0.0.1:11438/healthz
curl http://127.0.0.1:11438/v1/status
curl http://127.0.0.1:11438/metrics
curl -X POST http://127.0.0.1:11438/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":1}'
curl -N -X POST http://127.0.0.1:11438/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

Streaming uses HTTP/1.1 chunked `text/event-stream`. Confirmed target tokens are
sent directly from the native decode/MTP path, followed by a finish chunk and
`data: [DONE]`; this is not post-generation response slicing. Set
`stream_options.include_usage` to `true` to include usage in the finish chunk.
If the client disconnects, the sink propagates cancellation into native decode
at the next committed-token boundary instead of occupying the executor for the
rest of `max_tokens`. `/v1/status` and `/metrics` expose cancelled requests.
The listener uses four bounded HTTP workers and a 128-connection queue, while a
dedicated executor owns the MLX engine for its entire lifetime. Health, status,
and metrics therefore remain responsive during generation without evaluating
the model from multiple threads or duplicating its weights.

`--profile speed` applies the complete verified Apple-Silicon configuration:
the fused Q4 MoE/device router, compiled linear layers, fused HC/GDN paths,
selected-expert decode softmax, grouped SDPA prefill, the `12:28` resident
expert tier, chunk 512 for short prompts, and the bounded tiered long-prompt
policy. It also extends a successful prefix-cache entry through generated
assistant tokens, including lazily batched MTP-state catch-up after a fallback.
For an exact cached prompt, it also keeps an MTP path that was cumulatively
profitable before a late local fallback instead of disabling all speculation
on the replay. Set `QWEN38_MTP_CUMULATIVE_PROFITABILITY_CACHE=0` for the legacy
all-or-nothing decision.
Set `QWEN38_EXTEND_PREFIX_CACHE=0` for the prompt-only cache. Existing
environment variables and an explicit `--prefill-chunk`
override the preset. It normally settles around 35--37 GiB on the validated
64 GiB M5 Pro and should be launched through the memory guard. `--profile safe`
is the default conservative graph with chunk 64.

`--profile latency` enables the same optimized path but expands the resident
expert tier from `12:28` to `12:34`. On the tested 64 GiB M5 Pro this used about
0.9 GiB more memory and reduced the first cold decode step by 12--17%; sustained
warm decode was unchanged. Use it when cold-start/TTFT matters more than the
small memory increase. `--profile speed` remains the balanced default.

`--mtp-depth auto` is the default. When the model index has the Qwen3.8 MTP
companion, short prompts start at depth 2 for an eight-round probe and promote
to depth 3 only after at least 10 accepted drafts. A promoted request is checked
in 12-round windows and permanently demoted to depth 2 after two consecutive
windows below 50% per-draft acceptance. A recovered window resets that
hysteresis. The `speed` profile also treats only the first four depth-3 rounds
after promotion as a probation window and demotes immediately when they are
already below 50%; set `QWEN38_MTP_EARLY_DEMOTION=0` to retain the full initial
12-round window. `QWEN38_MTP_DEMOTION=0` is a diagnostic rollback that disables
both demotion checks without changing the initial probe.
Prompts longer than 2,048 tokens stay at depth 2. Without an MTP
companion auto mode stays serial. `3` uses the same adaptive cap explicitly;
`off`, `2`, and `4` are fixed alternatives. On a 64 GB Mac, auto mode also
selects the verified
`12:28` resident-expert balanced tier unless `QWEN38_RESIDENT_EXPERT_RANGE` is
already set, caps the MLX cache at 256 MiB, and falls back to serial after two
early zero-accept rounds or when a full 16-round window averages less than one
accepted draft per round. This avoids disabling profitable MTP on a random
empty streak while still ending locally losing speculation. Set
`QWEN38_ECONOMIC_MTP_FALLBACK=0` to restore the legacy two-empty-round policy.
While learned MTP remains active, a request-local suffix cache can replace a
learned proposal when it has an exact prior continuation of at least two
tokens. Learned MTP runs first by default: history proposals become eligible
only after an eight-round learned probe remains below 50% acceptance, and are
disabled for the rest of that request when their own two-round trial stays below 50%.
This avoids replacing a high-quality native Qwen draft with a merely matching
but locally weak suffix, or repeatedly retrying a failed history source. When
the history trial misses that floor, learned MTP resumes under its independent
profitability guard instead of disabling speculation immediately.
If no exact history continuation is available, the normal learned-MTP
profitability fallback remains active while the cache waits for one.
The target verifier remains authoritative. Set `QWEN38_HISTORY_DRAFT=0` to
disable history or `QWEN38_HISTORY_DRAFT=1` to force the legacy eager policy.
The cache stores token IDs and suffix indices only, so its memory is tiny
relative to model and KV residency even at long context.

Extended cache hits preserve the continuous decode state instead of rebuilding
the old assistant response through wide prefill. This is token-exact when those
numerical paths agree; otherwise a later greedy trajectory may differ while the
target model remains authoritative. In the retained code/creative/JSON chat
cohort, all responses remained coherent and prompt time improved by about
1.6--3.1x at a 34.7 GiB peak. The `safe` profile does not enable extension.

Thinking chat responses stop cleanly at either Qwen EOS marker. When a generated
`</think>` delimiter is present, `/v1/chat/completions` returns the answer in
`content` and the preceding trace in `reasoning_content`; clients should send
both fields back on subsequent assistant messages to reproduce the chat prefix.
The reproduced Q4/g64 L47 sidecar is the recommended production companion. A
same-target B31 run found the retained Q8 sidecar slower on code, creative, and
JSON prompts because its higher per-round cost did not buy enough acceptance.
The runtime nevertheless supports mixed Q4/Q8 sidecars by detecting each affine
projection's packed geometry. For Q8 routed experts it uses a dedicated
mlx-serve-derived gather-QMV/BF16-SwiGLU Metal lane whose reduction order is
needed for useful draft acceptance; `QWEN38_Q8_EXACT_MOE=0` restores the generic
diagnostic path. On the retained 38-token deterministic Python prompt, the
exact lane plus corrected depth hysteresis completed 256 tokens in 75 rounds,
accepted 180/217 drafts, and repeated at 59.28--59.30 tok/s warm. The same model
with the earlier premature demotion repeated at about 55.86 tok/s. This is an
M5 Pro 64 GiB, Q4-trunk/Q8-head, temperature-zero, thinking-off, no-prefix-cache
high-predictability result—not a universal MTP floor. Peak guarded RSS was
36.1 GiB.

Draft tokens are chained as one lazy MLX graph and synchronized together after
the final proposal. This removes one host synchronization per draft without
changing the proposal tokens or authoritative target verification. On the
retained 128-token Python fixture, counterposed warm runs improved from
61.80--62.03 to 62.59--62.67 tok/s while draft time fell from 241--243 to
219--221 ms with identical output and 89/103 acceptance. Set
`QWEN38_LAZY_MTP_DRAFT_CHAIN=0` for the serial-sync diagnostic rollback.

Verifier row decisions likewise batch their argmax and selected-logit graphs
behind one synchronization instead of synchronizing each row separately. The
same fixture retained byte-identical output and 89/103 acceptance while warm
throughput improved from 62.34--62.73 to 63.27--63.36 tok/s and total verifier
time fell from 1686--1696 to 1666--1669 ms. Set
`QWEN38_BATCH_VERIFY_ARGMAX=0` for the rowwise diagnostic rollback.

Accepted-row MTP head commits remain lazy and are materialized by the next
proposal instead of forcing an otherwise unnecessary end-of-round barrier. On
the 128-token fixture this reduced total commit time from about 83 to 26--27 ms
and improved warm throughput from 63.33 to 64.72 tok/s. Over 256 generated
tokens, the same 71-round, 185/205 acceptance path improved from 65.14 to
66.83 tok/s with byte-identical output. Creative and JSON controls also
improved slightly rather than regressing. Sampled RSS remained about 35.5 GiB.
Set `QWEN38_DEFER_MTP_COMMIT_EVAL=0` to restore eager materialization.

The batched target LM head also stays lazy until the already-batched verifier
decision consumes it, eliminating a second barrier between logits and argmax.
This is a smaller scheduling win: the 128-token fixture improved from
64.68--64.77 to 64.87--65.03 tok/s, and the 256-token warm run improved from
66.78 to 66.97 tok/s with identical output and acceptance. Set
`QWEN38_DEFER_VERIFY_HEAD_EVAL=0` to restore the separate head barrier. Developer
benchmarks that request isolated head timing still materialize it explicitly.

`--prefill-chunk 64` is the default layer-major prompt path. It bounds the
temporary prompt batch while preserving the retained production numerics.
Values through 512 are accepted when `QWEN38_GDN_METAL_PREFILL=1` enables the
oMLX-derived whole-sequence GDN recurrence; wider chunks otherwise fail at
startup instead of failing partway through a request. A configured chunk 512
is used only when the complete prompt has at most 512 tokens. Prompts through
6,144 forwarded tokens automatically use chunk 384, prompts through 8,192 use
chunk 256, and longer prompts use chunk 128 to retain the validated 64 GiB
memory bound.

`QWEN38_SDPA_PREFILL=1` replaces the full-attention token loop with MLX causal
SDPA at prompt widths of 16 or more. It retains the correct causal offset when
the key/value cache contains earlier chunks. The current experimental fast
prompt profile combines it with `QWEN38_GDN_METAL_PREFILL=1` and
`QWEN38_GROUPED_PREFILL=1`. `QWEN38_PREFILL_BARRIER_STRIDE=8` evaluates this
lazy graph every eight layers instead of every layer; values from 1 through 48
are accepted. Keep `--prefill-chunk 64` and leave these unset for
the retained numerical mode. On the 64 GiB validation machine, a single
512-row chunk reached about 671 prompt tok/s at 35.5 GiB peak; longer prompts
take the automatic chunk-128 safety path.

Serial decode and each MTP verifier row automatically switch the 12
full-attention layers from expanded-KV matmul/softmax to MLX grouped-query SDPA
once their cache reaches 512 tokens. The expanded path remains faster below the
crossover, while SDPA avoids materializing 16 repeated KV heads at longer
contexts. Set
`QWEN38_SDPA_DECODE=0` to force the short-context path or `1` to force SDPA for
diagnostics. At the same threshold, an MTP verifier evaluates all S query rows
in one causal grouped-query SDPA call while retaining a sliced checkpoint after
each row for partial acceptance and rollback. Set `QWEN38_BATCH_SDPA_VERIFY=0`
to restore one SDPA call per verifier row.
With both long-context paths active, exact history-cache hits at 512 or more
cached tokens may propose four tokens at once and fall back to the adaptive
S2/S3 continuation when no four-token match exists. Set
`QWEN38_LONG_HISTORY_DEPTH4=0` to restore adaptive-depth history everywhere.

The server retains one complete target/MTP prefix snapshot up to
`--prefix-cache-tokens` (default 8192). This deliberately caches every recurrent,
PLE, QSA, and KV state family together. Set the limit to `0` to disable it;
`POST /admin/cache/clear` releases the snapshot and the MLX allocator cache.
For an exact repeated prompt, the snapshot also remembers whether the initial
MTP probe deterministically fell back. A previously losing probe is skipped on
the next cache hit, while a profitable probe remains enabled. The response
reports this as `performance.mtp.profitability_cache_skip`. Set
`QWEN38_MTP_TRACE=1` for per-round acceptance and timing diagnostics.

For Q4/gs64 REAP-288 developer testing, `QWEN38_FUSED_MOE=1` enables the
attributed two-dispatch selected-MoE Metal path. Add `QWEN38_DEVICE_ROUTER=1`
to keep top-k selection and routing weights on the GPU. `QWEN38_COMPILE_LAYER=1`
compiles each non-PLE linear decoder layer after its first stateful step. Together
with the resident tier below, these form the current fastest verified path.
`QWEN38_HC_FUSED_INJECTION=1` completes the attributed decode-width HC read by
folding each four-row injection projection into a small dense resident matrix
and reducing it inside the existing normalize/down dispatches. A guarded
64-step fixed-start benchmark repeated at 41.7--42.6 tok/s versus 33.3 tok/s
for the retained read-only HC control, at 34.9--35.0 GiB peak RSS. The fused
gate may differ from the generic Q4 chain by one BF16 ULP, so this remains an
explicit speed/quality mode while mixed-corpus scoring is completed.

MTP verification evaluates the 48-layer lazy graph in three 16-layer groups by
default. This preserves the retained S-row numerics while removing most
per-layer synchronization overhead. `QWEN38_VERIFY_BARRIER_STRIDE=1` is the
diagnostic rollback; accepted experimental values are 1 through 48.

`QWEN38_BATCH_KV_VERIFY=1` builds each short-context full-attention layer's new
K/V rows once, appends them to the cached prefix once, and gives each verifier
row a prefix slice. Queries and attention remain rowwise, preserving the
retained short-context numerical path while avoiding repeated growing K/V
concatenations. The `speed` profile enables it; set it to `0` for rollback.

`QWEN38_GDN_METAL_VERIFY_BF16_SUM=1` replaces the S=2..4 GDN recurrence loop
with one Metal dispatch while retaining MLX's BF16 reduction tree: 32 virtual
threads each fold four consecutive values before the SIMD reduction. On an
initialized-state S=3 verifier it reduced 47.18 ms to 39.90 ms with identical
greedy tokens and rollback continuation, at the same 34.3 GiB peak RSS. A warm
256-token depth-2 HTTP completion improved 40.90 to 42.18 tok/s and produced an
identical output hash. It remains opt-in pending a wider prompt corpus.

Accepted MTP rows are committed back into the drafter with one S=2..4 batched
state pass instead of serially replaying each row. This changes proposal-state
scheduling only; target verification remains authoritative. Set
`QWEN38_BATCH_MTP_COMMIT=0` for the serial diagnostic rollback.

`QWEN38_SELECTED_SOFTMAX_ROUTER=1` ranks the 288 router logits before applying
softmax to only the selected ten experts. This is algebraically equivalent to
full softmax followed by top-k renormalization, but uses a different BF16
reduction path. A fixed-input A/B/A improved about 2%, and a three-prompt HTTP
cohort remained coherent while reaching 35.17/38.15/41.38 tok/s. Keep it as an
explicit speed mode until a larger quality suite is complete.

On unified-memory Macs, run full-model experiments through
`devtools/memory_guard.py -- COMMAND`. Full-model smoke and benchmark binaries
refuse unguarded execution. The default guard refuses to start below
42 GiB reclaimable memory and terminates the process group plus recursively
spawned descendants before they exceed 38 GiB aggregate RSS or leave less than
8 GiB reclaimable. SIGINT, SIGTERM, and SIGHUP drain the isolated child process
group before the guard exits, including when a terminal wrapper sends multiple
shutdown signals. A system-wide lock prevents overlapping guarded model runs.

For per-token warmup evidence rather than a two-token snapshot:

```bash
./devtools/memory_guard.py -- ./build/qwen38-model-bench /path/to/model 8
```

Set `QWEN38_FIXED_INPUT=1` to hold the benchmark input token constant and
remove generated-content routing differences from an A/B. Set
`QWEN38_PROFILE_DECODE=1` to append a materializing per-layer timing sample
after the requested warmup steps.

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
