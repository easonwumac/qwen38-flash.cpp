# qwen38-flash.cpp

An independent C++20/Metal inference server for Qwen3.8 Flash Next on Apple
Silicon. Model loading, the 48-layer forward pass, tokenization, state/cache
management, speculative verification, and streaming HTTP serving run locally.

**Daily-use baseline, 2026-09-19:** VQ 2.1bpw, one automatic configuration,
optional native Q6 MTP. The optimization phase is closed at the user's request;
this is the best retained tested configuration, not a proven hardware limit.
[Closeout and verification](docs/daily-use-2026-09-19.md) ·
[Changelog](CHANGELOG.md) · [All model comparisons](docs/results-guide.md)

The newer mixed-d4/d8 **VQ v2 has been evaluated, with quality trade-offs**:
HumanEval 153/164 versus v1's historical 146/164, but paired IFBench 14/30 versus
18/30. A fixed three-task SWE-bench Verified pilot also favored v1, 2/3 versus
v2's 1/3; this is directional evidence, not a full-suite score. Short decode
and footprint are essentially unchanged. v1 remains the daily default.
[Paired results](docs/vq-v2-upgrade-2026-09-19.md) ·
[SWE-bench pilot](docs/swebench-verified-pilot-2026-09-20.md)

An external Splash 1.0 control materially improved the dense 27B operating
point on the same 64 GiB M5 Pro: HumanEval **154/164**, IFBench first-30
**19/30**, HumanEval decode **80.00 tok/s median**, and a constrained 131K-token
run at **219.03 PP / 38.67 native decode tok/s** with a 23.39 GiB session peak.
Its model-specific DFlash 2 draft cannot be reused by Flash-Next, and its
chat-only HumanEval prompt is not byte-identical to the historical 27B control.
[External runtime report](docs/splash-27b-evaluation-2026-09-20.md)

VQ 3.2bpw hybrid and SSD expert streaming were also tested under the same
40 GiB ceiling. A one-layer stream preserved the paired token trajectory, but
fell to 21.32 tok/s; six streamed layers fell to 8.73 tok/s, and MTP did not
recover the loss. A down-only hybrid passed the 3-case IFBench gate but was
slower than VQ 2.1bpw. These remain opt-in checkpoint capabilities, not the
daily model. [Hybrid/streaming qualification](docs/vq32-hybrid-streaming-2026-09-20.md)

A layer-sensitive VQ pruning mask kept six sensitive layers at 512 experts and
retained 384 on the other 42. Its MTP-off result reached 15/30 IFBench and
152/164 HumanEval, but the required MTP-on follow-up fell to 14/30 on v1 and
13/30 on v2. The mask also retained all physical weights and v2's fixed MTP
fixture fell from a paired 53.66 to 46.03 tok/s. The candidate is therefore
rejected; no sub-40-GiB physical checkpoint or speed saving is claimed.
[Pruning qualification](docs/vq-sensitive6-pruning-2026-09-20.md)

## What this delivers

- Packed-VQ expert execution without expanding the whole model into dense weights.
  Segmented GEMM reuses decoded tiles across prompt rows; layer-major scheduling
  reduces weight churn and dispatch overhead.
- Exact top-10 routing, checkpoint PLE n-gram lookup backed by mmap, QSA and Q8
  KV, with bounded allocations. Qualified workloads used about 36–40 GiB.
- Adaptive native MTP with transactional state commit/rollback and fallback when
  drafting stops paying; complete prefix state for subsequent conversation turns.
- OpenAI-style Chat Completions, tools, SSE, cancellation, health/status metrics,
  and up to four independent request slots. No UI or separate tuning profiles.

## Current VQ evaluation

Model: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`, revision
`64b0fb0f98a552d91fb9abd5531d547b2e78c8a8`; original packed-14 VQ experts,
Q8 dense backbone and bundled VQ PLE. d8 decode uses approximate INT8/U8
codebooks; wide prefill keeps the checkpoint FP16 codebooks.

Retained measurements on **M5 Pro, 18 CPU cores, 64 GiB, macOS 26.5, MLX 0.32.2**,
AC power, no active thermal conditioning. Greedy/no-thinking, MTP and prefix
cache off unless stated. These are different workloads, not one universal rate.

| Workload | Retained result | Scope |
|---|---:|---|
| Prompt processing | **455.45 cold / 519.91 warm tok/s** | 7,091-token hashed repository corpus; one cold/warm pair |
| Target-only decode | **30.59 tok/s median** | Fixed-input 64-step probe; three starts: 30.55 / 30.72 / 30.59 |
| Native MTP decode | **57.94 tok/s median** | Favorable 64-token coding fixture; three samples, Q6 MTP + Q4 draft head |
| HumanEval original tests | **146/164 (89.02%)** | Native MTP; full-program chat, max 768; 46.93 aggregate decode tok/s across the serial run |
| HumanEval target-only control | **147/164 (89.63%)** | Same full-program protocol; not HumanEval+ |
| IFBench first 30 | **18/30 strict/loose (60%)** | Native MTP, max 4,096; 35.28 aggregate decode tok/s across serial requests |
| IFBench thinking pilot | **8/10 (80%)** | First ten, sampled xhigh, bounded 4,096; MTP bypassed; not full IFBench |
| SWE-bench Verified pilot | **2/3 resolved** | Three fixed tasks, mini-SWE-agent, automatic native MTP; directional subset, not a full-suite score |
| 32K context | **22.68 decode tok/s** | 32,024 input / 29 output tokens, Q8 KV, MTP off; 39.5 GiB peak |
| Two-stream model probe | **41.35 aggregate tok/s (1.351×)** | Two independent 64-step streams; HTTP gains are smaller |
| Memory | **36.3–39.6 GiB footprint** | Observed qualified runs; not an all-context guarantee |

Full corpus/protocols, raw distributions and comparison limits:
[benchmark detail](docs/benchmark-history.md#current-vq-evaluation),
[quality qualification](docs/vq-quality-preserving-2026-09-19.md#full-quality-qualification),
[results guide](docs/results-guide.md).
The rewritten README is **not** the old prefill corpus.

## Quick start

Requirements: Apple Silicon, CMake 3.24+, C++20 compiler, `utf8proc`, matching
MLX 0.32.2 / pinned MLX-C headers and libraries, and the complete checkpoint.
Set `MODEL_DIR`, `MLXC_INCLUDE_DIR`, and `MLX_LIBRARY_DIR` to your local assets.
Weights and sidecars are not included in this repository.

Run from the repository source root:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DQWEN38_ENABLE_TOKENIZER=ON \
  -DQWEN38_MLXC_INCLUDE_DIR="$MLXC_INCLUDE_DIR" \
  -DQWEN38_MLX_LIBRARY_DIR="$MLX_LIBRARY_DIR"
cmake --build build-release --parallel
QWEN38_TEST_MODEL="$MODEL_DIR" \
  ctest --test-dir build-release --output-on-failure

DYLD_LIBRARY_PATH="$MLX_LIBRARY_DIR" \
python3 devtools/memory_guard.py \
  --min-start-gib 42 --min-available-gib 6 \
  --max-rss-gib 38 --max-footprint-gib 40 -- \
  ./build-release/qwen38-server \
  --host 127.0.0.1 --port 11438 --model "$MODEL_DIR"
```

Use a clean shell without old `QWEN38_*` experiment overrides. In particular,
`QWEN38_PERSISTENT_VQ=1` is **not** the daily path. Defaults include an 8,192-token
RAM prefix-cache limit and a 4,096-token generation cap; these are resource
limits, not model context limits. Shutdown with Ctrl-C/SIGTERM.

The guard refuses a crowded machine and stops its server if memory limits are
crossed; it is a polling safety monitor, not an allocator-enforced hard cap.
It does not guarantee that every request fits. Python runs the safety wrapper
and offline tools, not the inference core.

Place compatible `mtp-head-q6.safetensors` beside the model shards to enable
automatic native MTP. Optional `mtp-lm-head-q4.safetensors` accelerates draft
proposals only; the target LM head stays Q8. Without the MTP sidecar, execution
is target-only. No performance profile selection is required.
[Startup, asset checks and recovery](docs/operations.md)

In another terminal, check `/readyz` and send a request:

```bash
curl -fsS http://127.0.0.1:11438/readyz
curl -fsS http://127.0.0.1:11438/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash","messages":[{"role":"user","content":"Reply with exactly OK."}],"temperature":0,"thinking":false,"max_tokens":16}'
```

## Boundaries

- **600 PP / 40 target-only / 60 mixed-workload MTP tok/s remain unmet goals.**
  A favorable MTP fixture near 60 is not ordinary-chat performance.
- **VQ 128K is not qualified.** The declared 262,144-token model context and
  historical REAP/Niwaki retrieval runs are not VQ capacity guarantees.
- MTP verification and serial decode follow different floating-point paths.
  MTP is quality-tested but not promised token-identical to target-only output.
- Thinking pilot scores cannot establish full-benchmark quality or superiority
  over 27B. Thinking/sample settings can bypass MTP and cost many more tokens.
- Batching improves aggregate throughput, not every request's latency. MTP is
  bypassed during cross-request batching; thinking and oversized batches remain
  serial. Four slots do not imply four times the speed.
- Text only. Chat Completions is supported, not the Responses API. Keep the
  unauthenticated service on loopback; remote use requires a secured gateway.

## Historical reference evaluation

REAP-288, Niwaki, 27B and other comparison runs remain searchable in the
[results guide](docs/results-guide.md), [historical benchmark tables](docs/benchmark-history.md#historical-reference-evaluation),
and [public quality evaluation](docs/public-quality-evaluation.md).
They are reference/research results, not alternative daily configurations.

## Documentation

| Need | Document |
|---|---|
| Accepted configuration, closure decision, deferred work | [Daily-use baseline](docs/daily-use-2026-09-19.md) |
| What's changed | [Changelog](CHANGELOG.md) |
| Scores, speed, context, RAM, pruning and rejected models | [Results guide](docs/results-guide.md) |
| External Splash 27B quality, throughput and 128K control | [Splash evaluation](docs/splash-27b-evaluation-2026-09-20.md) |
| Full protocols and implementation gains | [Benchmark detail](docs/benchmark-history.md) |
| Startup, resources, cache, shutdown and recovery | [Operations](docs/operations.md) |
| API / engine / supported tensor layouts | [API](docs/api.md) · [Architecture](docs/architecture.md) · [Capabilities](docs/model-capabilities.md) |
| Measurement rules and future promotion gates | [Benchmark contract](docs/benchmark-contract.md) · [Release checklist](docs/release-checklist.md) |
| Failed experiments and why not to repeat them | [Research ledger](docs/prior-research-ledger.md) · [Latest verifier report](docs/persistent-verifier-2026-09-19.md) |

## License

Apache-2.0. Third-party notices are in `NOTICE` and `licenses/`.
