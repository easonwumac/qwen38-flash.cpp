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
> non-MTP and mixed-workload decode targets remain unmet. A guarded real-API
> request now validates 258,457 prompt tokens, within 3,687 tokens of the
> declared 262,144-token limit.

The execution path now includes a real-checkpoint Q4 affine projection smoke:
`qwen38-qmm-smoke MODEL_DIRECTORY` loads the retained `lm_head` through MLX's
lazy safetensors primitives and evaluates its 248,320 logits on the Apple GPU.
This proves weight-to-Metal plumbing, not a complete transformer forward.

`qwen38-hyper-connection-smoke MODEL_DIRECTORY` additionally executes the real
layer-0 read/injection gates, residual write-back, and final mixer. Its outputs
match the independent `devtools/hyper_connection_oracle.py` MLX-Python oracle.
`qwen38-sparse-moe-smoke MODEL_DIRECTORY [PREFILL_ROWS]` verifies router top-k,
ten selected Q4 experts, and the gated shared expert against a second independent
oracle. An optional width from 2 through 512 exercises and times the production
grouped-prefill path; append `components` to report router, gate/up, down/reduce,
shared-expert, and merge medians. Run either memory-heavy mode through
`devtools/memory_guard.py`.
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

`qwen38-long-context-smoke MODEL PROMPT [CHUNK [MAX_TOKENS [PROFILE]]]` is the
guarded full-model QSA check. It requires
`devtools/memory_guard.py`, defaults to the same adaptive chunk policy and
`speed` profile as the server, and reports linear/full-attention time, actual
pooled blocks, prefill throughput, and one-token decode latency. Pass `safe` as
the profile for a conservative unfused diagnostic control.

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
warm decode was unchanged. On a separate 3,506-token expert-diverse prompt it
also improved cold prefill from 310.7--311.4 to 332.5 tok/s (+6.9%) at 35.9 GiB
peak RSS, while the next token stayed identical. Use it when cold-start/TTFT and
uncached prompt speed matter more than the small memory increase. `--profile
speed` remains the balanced default.

`--profile long-context` keeps the optimized `speed` graph and adaptive
prefill chunks but leaves the resident-expert range empty, allowing macOS to
reclaim model pages as QSA/KV state grows. It is the capacity profile for
requests approaching 262K on the tested 64 GiB machine; existing
`QWEN38_RESIDENT_EXPERT_RANGE` settings still take precedence. For maximum
headroom, pair it with `--mtp-depth off --prefix-cache-tokens 0`. It trades
cold-weight residency and short-request latency for memory capacity, so keep
`speed` for normal prompts. When compact qmeta prefill caching is enabled, the
server automatically disables the decoded-bank cache above 32,768 prompt
tokens so it cannot grow alongside the long-context attention state. Override
that crossover with `--qmeta-cache-max-prompt-tokens`; zero disables decoded
qmeta prefill caching for every request.

`--profile memory` is the lowest-pressure exact-weight preset for the retained
Qwen3.8 Flash Next build. It uses the lossless13 compact metadata sidecar,
leaves expert pages pageable, disables decoded-qmeta and RAM prefix caches,
keeps MTP off unless explicitly requested, enables packed QSA above 65,536
tokens, and retains the validated 256 MiB MLX allocator cap. Temporary decoded
metadata stays in the lazy graph only through the normal eight-layer prefill
barrier, avoiding a synchronization at every layer without retaining all 48
layers for the request. Set `QWEN38_QMETA_PREFILL_DEFER_TEMPORARY=0` for the
old per-layer synchronization path. The lossless13 sidecar is
required; startup fails clearly instead of silently falling back to the larger
metadata representation. Use the memory guard for every long-context run.

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

The retained high-acceptance MTP path has a standalone regression client and a
memory-guarded launch recipe in [docs/mtp-benchmark.md](docs/mtp-benchmark.md).
It checks per-length warm throughput, proposal acceptance, completion length,
and whether MTP actually engaged; it does not use prefix-cache hits.

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

For a lower-resident-memory deployment, `devtools/build_lossless_qmeta.py`
re-encodes the routed experts' BF16 scale/bias pairs as exact bank-local
dictionaries without changing the Q4 weights. `--bits 16` is the recommended
aligned format: its retained sidecar is 2.0 GiB for all 144 projection banks.
Set `QWEN38_COMPACT_QMETA=1` (or `lossless16`) to use it; `lossless13` selects
the denser 1.6 GiB diagnostic format. The runtime never enables either format
implicitly and fails at startup if the requested sidecar is incomplete.

```bash
python3 devtools/build_lossless_qmeta.py /path/to/model --bits 16
QWEN38_COMPACT_QMETA=1 ./build-all/qwen38-server \
  --model /path/to/model --profile speed
```

On the 64 GiB M5 Pro, the same 1,152-token MTP workload measured 30.68 GiB RSS
with lossless16 versus 36.49 GiB for the full-metadata control. The exact
89/103 and 185/205 acceptance paths and generated output were preserved.
Lossless16 reached 64.56 tok/s at 128 tokens and a thermally falling
64.60/62.01 tok/s pair at 256, compared with control medians of 64.18 and
64.995 tok/s. It is therefore a memory profile, not a claimed decode-speed
promotion. A repeated 2,424-token prompt reached 335.4 PP tok/s warm versus
397.1 for control, a 15.5% prefill cost because wide prefill reconstructs
layer-local affine banks before the grouped QMM. Keep the normal `speed`
profile when prompt throughput matters more than roughly 5.8 GiB of RSS.
For multi-chunk prompts, `QWEN38_QMETA_PREFILL_CACHE=1` retains each decoded
bank only for the duration of prefill and releases all 48 layers before decode.
It raised the same warm 2,424-token prompt to 367.4 PP tok/s (+9.5% over the
uncached compact path), then preserved the 89/103 128-token MTP trajectory at
64.02 tok/s. Cached banks also let the routed reduction use the normal
eight-layer prefill barrier instead of synchronizing every layer. A same-binary
A/B/A raised warm 2,424-token PP from 362.5/369.8 controls to 384.1 tok/s; a
repeat reached 382.4. Code, JSON, and prose prompts spanning 2.5K--4.4K tokens
kept identical first-token hashes, and an 8,664-token request reached 294.85 PP
with a 24.84 GiB sampled peak. Set
`QWEN38_QMETA_PREFILL_DEFER_REDUCE=0` to restore the per-layer barrier. The
request cache remains opt-in because it has not yet been validated near the
64 GiB capacity boundary at the full 262K context limit. The server therefore
uses it only through 32,768 prompt tokens by default, even when the environment
flag is set.

`--prefill-chunk 64` is the default layer-major prompt path. It bounds the
temporary prompt batch while preserving the retained production numerics.
Values through 1024 are accepted when `QWEN38_GDN_METAL_PREFILL=1` enables the
oMLX-derived whole-sequence GDN recurrence; wider chunks otherwise fail at
startup instead of failing partway through a request. A 1024-row superchunk is
kept whole through GDN and MoE while full-attention layers process two ordered
512-row subchunks, retaining the attention-state contract without decoding MoE
metadata twice. Adaptive mode retains an explicit 1024-row chunk through
32,768 forwarded tokens and then reduces it to 512; `--prefill-chunk-fixed` is
required to exercise 1024 rows at longer contexts. The optimized profiles
retain their default chunk 512 through 32,768 forwarded tokens, then reduce to
chunk 128 for longer prompts. On the 64 GiB M5 Pro, an 8,216-token prompt rose
from 463.7 tok/s with the old chunk-128 crossover to 655.45/638.38 tok/s, while
a 32,760-token prompt reached 532.67 tok/s and ended at a 36.66 GiB physical
footprint. The first-token output hash was unchanged. `--prefill-chunk-fixed`
disables the long-prompt adaptive reduction for
guarded throughput experiments; it must remain behind the memory guard because
larger long-prompt batches increase the Metal working set.

With the REAP-288 MLX 4-bit repack, MTP and prefix caches disabled, a guarded
32,792-token numbered-lines prompt improved from 502.35 tok/s at chunk 512 to
517.16 tok/s with the split 1024-row superchunk; the greedy first-token hash was
unchanged. At 65,560 tokens, combining the superchunk with the experimental
`QWEN38_QSA_PACKED_MIN_TOKENS=32768` setting reached 433.55/435.62 tok/s versus
367.41/357.89 tok/s at the retained defaults, with the same first-token hash in
both candidate runs. Peak physical footprint was 39.8 GiB. The earlier packed
QSA threshold remains the default because one prompt hash is not sufficient to
establish broad numerical parity for the lower threshold.

Packed QSA uses a 32-token Metal KV tile by default. Compared with the original
16-token tile, a guarded one-layer 65,536-token QSA smoke reduced packed
attention from 66.37 ms to 60.05 ms with dense/packed cosine 1.0. On the full
65,560-token prompt above, the split superchunk plus the experimental 32K QSA
threshold produced 466.02/442.49 tok/s with the retained first-token hash and a
39.9 GiB peak physical footprint. Set `QWEN38_QSA_PACKED_TILE=16` for the
diagnostic rollback.

For a guarded 64K throughput experiment, retaining decoded qmeta for only the
first four layers is the measured memory/speed sweet spot:

```bash
QWEN38_QSA_PACKED_MIN_TOKENS=32768 \
QWEN38_QMETA_PREFILL_CACHE=1 \
QWEN38_QMETA_PREFILL_CACHE_LAYERS=4 \
./devtools/memory_guard.py \
  --min-start-gib 40 --min-available-gib 10 \
  --max-rss-gib 40 --max-footprint-gib 42 -- \
  ./build-all/qwen38-server --model /path/to/model --profile memory \
  --prefill-chunk 1024 --prefill-chunk-fixed \
  --qmeta-cache-max-prompt-tokens 262144 --prefix-cache-tokens 0 --mtp-depth 0
```

On the same 65,560-token numbered-lines prompt this reached 473.62/463.55
tok/s, a 468.59 tok/s median versus 454.26 without the cache (+3.2%). Both runs
retained the `b344d80e...` first-token hash; peak physical footprint was 40.2
GiB and minimum reclaimable memory was 11.0 GiB. Caching eight layers did not
improve the cold sample (460.08 tok/s), so four layers remains an experimental
recipe rather than a default. It is validated on one deterministic prompt, not
as a mixed-domain quality or universal throughput claim.

The native engine caps the MLX allocator cache at 256 MiB in serial as well as
MTP mode. Before this bound covered serial inference, a 23,555-token request was
stopped by the memory guard at 5.8 GiB available memory. With the common bound,
the identical uncached request completed at 269.2 prompt tok/s. A six-copy,
141,275-token request then completed at 172.3 prompt tok/s and generated the
next token in 114.4 ms with MTP and prefix caching disabled. The server stayed
at about 35.7 GiB sampled RSS and the lowest manual availability sample was
7.4 GiB on the 64 GiB M5 Pro.

With resident-expert locking disabled—the behavior now exposed as
`--profile long-context`—an uncached 258,457-token request completed in
2,443.71 seconds at 105.77 prompt tok/s and generated one token in 196.5 ms.
MTP, history drafting, and prefix caching were disabled. The earlier RSS-only
memory guard kept at least 6 GiB available; the final sample had 16.4 GiB
available and about 4.4 GiB process RSS after reclaim. Those RSS figures exclude
Metal allocations and are retained only as historical throughput evidence;
current runs enforce a 48 GiB physical-footprint limit. This validates the API
path within 3,687
tokens of the model limit on the 64 GiB M5 Pro. It does not establish an exact
262,144-token boundary test or acceptable sustained decode speed at that
context; QSA decode and long-context retrieval quality remain open gates.

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

Set `--ssd-prefix-cache-gib N` to persist those exact complete-state snapshots
across requests and server restarts. The store is disabled by default, uses a
model/runtime compatibility namespace, finds the longest cached prefix of an
append-only prompt, and restores target, MTP, KV/QSA, GDN, PLE/ngram, and pending
stream state before processing only the appended tokens. It is not a token-list
cache: the full recurrent checkpoint is stored as safetensors. Entries are
least-recently-used and bounded by the configured total size. The default path
is `~/Library/Caches/qwen38-flash.cpp/prefix/`; use
`--ssd-prefix-cache-dir PATH` to select another SSD location. Enabling the SSD
store also extends eligible checkpoints through generated output so the next
chat turn can reuse the prior assistant turn. The existing
`--prefix-cache-tokens` limit remains the maximum persisted checkpoint length;
raise it deliberately for longer conversations. `POST /admin/cache/clear`
clears both RAM and the active SSD namespace.

```bash
./devtools/memory_guard.py -- ./build-all/qwen38-server \
  --model /path/to/model --profile speed \
  --prefix-cache-tokens 32768 --ssd-prefix-cache-gib 8
```

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
spawned descendants before they exceed a 48 GiB aggregate macOS physical
footprint, exceed 38 GiB aggregate RSS, or leave less than 8 GiB reclaimable.
Physical footprint is the authoritative 64 GiB safety limit because it includes
Metal/IOAccelerator allocations that `ps` RSS omits. Override it only for a
controlled experiment with `--max-footprint-gib`. The 48 GiB default leaves an
8 GiB reclaimable floor on the tested 64 GiB Mac; 52 GiB is not a stable
default. SIGINT, SIGTERM, and SIGHUP
drain the isolated child process group before the guard exits, including when a
terminal wrapper sends multiple shutdown signals. A system-wide lock prevents
overlapping guarded model runs.

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
