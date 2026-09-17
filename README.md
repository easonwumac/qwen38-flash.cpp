# qwen38-flash.cpp

`qwen38-flash.cpp` is an independent C++20/Metal inference server for Qwen3.8
Flash Next on Apple Silicon. It owns model loading, tokenization, the complete
48-layer forward pass, hybrid attention state, speculative verification,
caching, HTTP serving, streaming, and runtime observability.

The server has one automatic production configuration. For the VQ target it
selects the validated sparse-attention, prefill, continuous-batching, and
Metal paths from the supplied assets; there are no tuning profiles to choose.
When the checkpoint's optional native Q6 MTP sidecar is present, the same
automatic configuration enables it; otherwise the target runs without MTP.

## Current target model

- Package: `TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw`
- Target: native packed-14 VQ routed weights, original d2/K256 and d8/K16384
  geometry, exact top-10 routing
- PLE: the checkpoint's bundled VQ n-gram shards, memory mapped on demand
- Context declared by the model: 262,144 tokens
- MTP: optional native `mtp-head-q6.safetensors` sidecar; automatically detected
- Automatic prefill: 2,048-row chunks through 32K, then smaller bounded chunks
- Runtime: C++20/Metal with MLX 0.32.2 through a pinned MLX-C ABI

REAP-288, Niwaki, and Qwen3.8-27B results below are retained only as historical
quality, capacity, and performance references. They are not production targets,
fallback checkpoints, or dependencies of the VQ path.

## What improved for VQ

- **Layer-major batched execution:** wide prompt and verifier rows pass through
  each layer together, removing token-by-token dispatch and synchronization.
- **Segmented packed-VQ GEMM:** the Metal kernel shares decoded VQ tiles across
  routed prompt rows instead of repeating scalar matrix-vector work.
- **Exact fused VQ intermediates:** gate/up performs MLX-equivalent FP16 SwiGLU
  in-kernel, while down stores compact FP16 projections and applies route
  weights during FP32 reduction; layer outputs remain bit-identical.
- **Automatic 2,048-row prefill:** packed-VQ checkpoints use the larger verified
  chunk through 32K; non-VQ checkpoints and explicit resource limits keep their
  prior bounds.
- **Bounded layer-major prefill:** up to 8,192 prompt rows advance through each
  eight-layer evaluation group before the next weight group, reducing mapped
  weight churn without retaining an unbounded prompt graph.
- **Compile-time tail specialization:** wide VQ batches send complete 24-route
  tiles and short expert tails through separate fixed Metal kernels, then restore
  the exact route order before down reduction.
- **Decode-priority refill:** a full-width refill prefill yields at existing
  eight-layer evaluation barriers, so it preserves prompt arithmetic while
  active decode rows continue between layer groups.
- **Fused decode kernels:** routed MoE, device-side routing, Hyper-Connection,
  and Gated DeltaNet work are fused around the actual decode-width hot path.
- **Native Q6 MTP:** the optional Qwen4-Exp sidecar is discovered without model
  repacking; fused Q6/group-32 gate/up/down kernels remove ten independent
  gather-QMM launches per routed head layer.
- **Exact down-slot parallelism:** packed d8 decode distributes three routed
  experts across 30 SIMD lanes, then restores the original group and route
  reduction order; this removes idle-lane work without changing layer bits.
- **Decode-only compact codebooks:** d8/K16384 layers derive a centered signed
  INT8 gate table plus affine U8 up/down tables at load time. Direct decode reads
  half the codebook bytes;
  d2/K256 and segmented prompt GEMM retain the original FP16 codebooks.
- **Bounded graph lifetime:** temporary compact metadata survives only to the
  existing eight-layer barrier, improving scheduling without retaining every
  decoded metadata bank for the whole request.
- **Bounded long-context state:** QSA activation, adaptive prefill chunks, and
  slabbed Q8 KV are selected from context growth rather than a user profile.
- **Exact QSA raw-state retirement:** index keys are discarded after their
  four-token block has been pooled, retaining only a 64-row construction window.
  This reduces long-context selector work without changing selected tokens.
- **Exact Qwen Sparse Attention:** raw and pooled indexer state, causal top-block
  selection, snapshots, verifier checkpoints, and rollback remain native to the
  engine.
- **Fused long-context selection:** a Metal selector can reuse each representative
  query across four adjacent prefill rows, then restore per-row causal validity.
  Combined with slabbed Q8 KV, it bounds selector and KV work; VQ has not yet
  been requalified at 128K.
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

## Current VQ evaluation

Common environment: Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GB unified
memory, macOS 26.5 (25F71), AC power, no recorded thermal or performance
warning, temperature 0, thinking off, one guarded server process, exact top-10,
and MTP off unless the row explicitly names the native sidecar.

| Workload | Configuration | Result |
|---|---|---:|
| Repository prefill, 7,091 tokens | VQ 2.1bpw; README corpus SHA-256 `633c8445...`; automatic 2,048-row chunks and 8,192-row layer-major window; same-process cold/warm server requests; MTP and prefix cache off; fixed first token | **455.45 cold / 519.91 warm PP tok/s**; **38.3 GiB** peak |
| Repeated repository prefill, 14,173 tokens | same corpus repeated twice; automatic 2,048-row chunks, two bounded windows; same-process cold/warm server requests; MTP and prefix cache off; fixed first text `Based` | **446.3 cold / 479.1 warm PP tok/s**; **39.1 GiB** peak |
| Short steady decode, fixed input, 64 steps | VQ 2.1bpw; exact top-10; signed gate plus affine up/down d8 decode codebooks; one automatic 48-layer evaluation barrier; three independent starts; first two compile/warmup steps excluded | 30.55 / 30.72 / 30.59 tok/s, median **30.59 tok/s**; paired stride-8/48 medians were 30.00/31.06 tok/s; **36.3 GiB** peak |
| 32K long-context decode, 32,024 prompt tokens | VQ 2.1bpw; automatic QSA budget 512 and exact 64-row raw-state window; Q8 KV; greedy, thinking/MTP/prefix cache off; 29 generated tokens | **22.68 tok/s** versus 21.09 without raw-state retirement (**+7.58%**); byte-identical response; **39.5 GiB** observed peak |
| Native MTP, 64 output tokens | VQ target plus native Q6 sidecar and automatic Q4-only drafter LM head; adaptive depth starts at 4; greedy/no-thinking; retained `merge_sorted_unique` coding fixture; three-sample confirmation | 57.953 / 57.938 / 57.509 tok/s, median **57.94 tok/s**; no demotion, unchanged 49/56 accepted in 14 rounds; **39.3 GiB** peak |
| External-drafter capacity probe, 128 output tokens | VQ target remains authoritative; compatible external Q8 drafter, depth 4; greedy/no-thinking; two warm samples on one retained high-acceptance fixture | **44.804 / 44.809 tok/s**; 95/128 drafts accepted in 32 rounds; **38.8 GiB** peak; not a mixed-workload or 60 tok/s result |
| IFBench first 30 prompts | native Q6 MTP plus Q4-only drafter LM head; automatic depth 4→3→2 when later draft positions stop paying; greedy, non-thinking, max 4,096; serial requests; official scorer | **18/30 strict and loose (60.00%)**, instruction-level 63.64%; 12,567 output tokens, 0 errors; **35.28 aggregate decode tok/s**, 71.23% draft acceptance; **39.4 GiB** peak |
| IFBench first 30 target-only control | same target, prompts and decoding protocol; MTP off | **14/30 strict and loose (46.67%)**, instruction-level 51.52%; 23,089 output tokens, 3 truncations; **28.55 aggregate decode tok/s**; **36.8 GiB** peak |
| IFBench bounded-thinking pilot, first 10 prompts | temperature 1, top-p .95, top-k 20, seed 0, xhigh with bounded close, max 4,096; sampled generation bypasses MTP although the sidecar remains resident; final-answer-only official scoring | **8/10 strict and loose (80%)** versus 5/10 for non-thinking native MTP on the same prompts; 27,939 completion tokens, 0 errors/truncations; **28.60 aggregate decode tok/s**; 1,622.7 s request wall time; **39.3 GiB** peak |
| Native-MTP IFBench development gate, keys 20/70/100 | native Q6 sidecar; greedy, non-thinking, max 512; official per-row loose/strict scoring | **3/3 loose and strict**; 0 errors; **38.9 GiB** peak; target-only control also 3/3 |
| IFBench development gate, keys 20/70/100 | VQ 2.1bpw; greedy, non-thinking, max 512; official per-row loose/strict scoring | **3/3 loose and strict**; **36.7 GiB** peak |
| IFBench stratified development set, keys 0,10,...,90 | VQ 2.1bpw; signed gate plus affine up/down d8 decode codebooks; greedy, non-thinking, max 4,096; official scorer | **5/10 loose and strict**; same passing keys 20/30/60/70/90 as the affine control; 0 errors; **36.9 GiB** peak |

The bounded layer-major schedule keeps the existing VQ arithmetic and state
ordering. In the 7,091-token developer A/B, warm layer-major runs reached
526.21/527.34 PP tok/s versus 497.81/505.90 for chunk-major, with the same next
token 27775 and logit 17.125. The product server reached 519.91 warm PP tok/s
at 38.3 GiB. These are current milestones, not claims that the 600 PP tok/s or
40 tok/s decode targets have been reached.

The route planner sends <=8-row expert tails to an RTILE8 kernel instead of
padding them to 16 rows. The same segmented matrix path now also handles the
checkpoint's first two d2/K256 layers, which previously used one direct VQ
matrix-vector operation per route. On the fixed pre-edit repository README
(`5e07210e...`, 6,469 tokens), this reduced paired prefill time by 11.3%; the
first token remained 27775. A 128-row layer-0 comparison against the direct
path had max absolute error 0.00293, and IFBench keys 20/70/100 remained 3/3
strict and loose at 36.6 GiB peak. This A/B does not replace the retained
7,454-token row until that exact corpus is available for a paired rerun.

For direct d8 decode, per-dimension affine U8 codebooks have about 0.54--0.55%
relative L2 error and roughly 0.999985 cosine against the checkpoint FP16
tables on representative gate/up/down banks. Gate vectors now use the same
per-dimension affine scale with signed, zero-centred INT8 codes; this removes
the bias add from the hottest gate loop. Three paired starts raised the affine
median from 29.25 to 30.33 tok/s (+3.7%) without increasing the 36.3 GiB peak.
Applying compressed codebooks to segmented prompt GEMM was 8.5% slower, so
wide prefill keeps FP16 and its exact output hash. The 10-case stratified
IFBench gate retained both the affine control's 5/10 aggregate and the exact
same passing cases. This is still approximate arithmetic and a development
gate, not evidence from the full 300-case benchmark.

Target-only decode now evaluates the 48-layer lazy graph once per token instead
of at eight-layer boundaries. A 64-token natural autoregressive paired run was
token-identical and improved from 28.03 to 30.45 tok/s; a three-pair fixed-input
sweep improved the median from 30.00 to 31.06 tok/s. The production confirmation
above was 30.59 tok/s at 36.3 GiB. IFBench keys 20/70/100 remained 3/3 strict
and loose. Native MTP target verification has its own retained stride and is
not changed by this setting.

Persistent Metal now shares only tensors used by target decode. It leaves the
8.97 GiB SSD-backed PLE n-gram bank and inactive MTP tensors out of the MLX-to-
Metal residency handoff, while missing bindings safely fall back to mmap. On the
VQ 2.1bpw target, MTP/thinking off, serial IFBench keys 20/70/100 at max 512,
this reduced guarded peak footprint from 49.1 to **37.7 GiB**. The selective and
full-residency controls produced byte-identical outputs on the paired keys; the
current main scored 2/3, so this is a memory/scheduling fix rather than a new
quality claim. The run used the M5 Pro 64 GiB validation Mac, AC power and no
active thermal control.

## Historical reference evaluation

Common environment: Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GB unified
memory, macOS 26.5 (25F71), AC power, no recorded thermal or performance
warning, temperature 0, thinking off, one server process, and guarded memory
measurement. Unless a row says otherwise, the checkpoint is the historical
REAP-288 Q4/group-64 reference.
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
| Exact routed-MoE continuous decode, first 12 IFBench prompts | Apple M5 Pro 64 GiB; REAP-288 Q4 + Q8 SSD PLE; automatic lossless16 metadata, four rolling slots, MTP/thinking off, greedy, max 384; adjacent serial/four-slot runs without active thermal control | **45.02 vs 38.74 aggregate decode tok/s (1.162x)**; 3,557 tokens, 0 errors; **12/12 responses byte-identical**; 37.1 GiB peak footprint |
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

- VQ currently reaches 519.91 warm PP tok/s on the retained repository prompt
  and 30.33 tok/s on the short decode fixture. The 600 PP and 40 decode goals
  remain open.
- VQ has not yet been requalified at 128K. Historical REAP/Niwaki long-context
  results must not be presented as VQ performance.
- With the optional native Q6 sidecar and Q4-only drafter head, VQ reaches
  57.94 tok/s on one retained high-acceptance coding fixture at 39.3 GiB. The
  automatic controller keeps depth 4 there, but demotes unprofitable later
  positions on mixed work. The first-30 IFBench run reaches 35.28 aggregate
  tok/s and 60.00% strict/loose at 39.4 GiB. Neither the mixed-workload 60 tok/s
  target nor full-benchmark quality is established.
- The MTP verifier and ordinary d8 decode both use the approximate INT8/U8
  codebooks, but the verifier evaluates several target rows through a different
  batched graph. It therefore does not promise byte or token parity with
  target-only decode.
  On the paired 30 prompts MTP had 14 both-pass, 3 MTP-only-pass, 0
  target-only-pass, and 13 both-fail cases. A serial target verifier restores
  exact target-only output on the isolated divergent case but falls to 6.1
  tok/s; it is a diagnostic oracle, not a product fallback.
- The official model card reports 81.3% IFBench, but does not publish enough
  generation details for a direct local reproduction. The bounded-thinking
  first-10 pilot reaches 80%, showing that the checkpoint/runtime can approach
  that quality under a reasoning protocol; it is not a substitute for the full
  300-prompt run and costs about 2,794 completion tokens per prompt.
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

If `mtp-head-q6.safetensors` is present beside the model shards, the automatic
path starts at depth 4 and demotes later draft positions only when their measured
acceptance stops paying. An optional `mtp-lm-head-q4.safetensors` is used only
for draft proposals; the authoritative target keeps its Q8 language head. The
offline converter `devtools/build_mtp_lm_head_q4.py` derives this sidecar from
the checkpoint's Q8 head. Without the native MTP file the server runs
target-only; `--mtp-depth off` remains an explicit resource-limit override.

The following 128K recipe is retained for historical REAP comparison only; it
has not been qualified for VQ:

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
