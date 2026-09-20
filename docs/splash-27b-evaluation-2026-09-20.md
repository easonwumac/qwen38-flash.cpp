# Qwen3.8-27B Splash evaluation

Date: 2026-09-20

Status: external runtime/control; not part of `qwen38-flash.cpp`.

## Conclusion

Splash 1.0 materially changes the local 27B comparison. Its specialized
Qwen3.8-27B Q4 target plus five-layer DFlash 2 draft preserved strong quality,
ran HumanEval at an 80.00 tok/s per-request median, and kept a constrained 128K
session's Metal allocations well below the project's 40 GiB ceiling. The
retained VQ engine still
demonstrates a different Flash-Next/VQ execution path, but it is no longer the
best measured local quality/speed/memory operating point on this Mac.

The DFlash 2 algorithm could be trained for Flash-Next, but the published 27B
draft is model-specific and cannot be attached to the VQ checkpoint. Inco AI
did not publish a Flash-Next DFlash 2 checkpoint at the time of this run.

## Artifacts and environment

- Hardware: Apple M5 Pro, 20-core GPU, 64 GiB unified memory (`Mac17,8`).
- OS: macOS 27.0 (`26A428`).
- Runtime: Splash 1.0, installed from `incoai/tap/splash`.
- Package: `incoai/Qwen3.8-27B-Splash`, installed with manifest/hash
  verification.
- Target lineage: `mlx-community/Qwen3.8-27B-4bit`, 14.1 GiB packed target.
- Draft: `incoai/Qwen3.8-27B-DFlash2`, five layers, 1.2 GiB, seven proposed
  tokens per verification block.
- Vision weights: 0.9 GiB BF16, loaded by the package even though these tests
  were text-only.
- Short/quality tests used Splash's automatic memory plan. The 128K tests used
  `--max-memory 40G --max-context 128K`.

Sources: [Splash package](https://huggingface.co/incoai/Qwen3.8-27B-Splash),
[Splash engine](https://github.com/incoai/splash), and
[DFlash 2 checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

## Results

| Workload | Result | Conditions and limits |
|---|---:|---|
| HumanEval original tests | **154/164 (93.90%)** | Chat-only self-contained-program prompt, greedy/no-thinking, max 768, EvalPlus 0.3.1 sanitizer, six length-limited, zero API errors |
| HumanEval throughput | **80.00 tok/s median** | 41,870 completion tokens; 63.69 aggregate completion tok/s including TTFT and serial request gaps |
| IFBench first 30 | **19/30 strict and loose (63.33%)** | Pinned public dataset, greedy/no-thinking, max 4,096, rolling concurrency four, 11 length-limited, zero API errors |
| IFBench four-stream throughput | **95.37 native / 93.57 end-to-end tok/s** | 75,154 completion tokens; DFlash accepted 57,629 / 122,528 drafts (47.03%) |
| Predictable short decode | **94.23 tok/s** | Three 256-token integer-list requests; 672 / 672 drafts accepted; deliberately favorable |
| Medium-reasoning samples | **46.78 tok/s median** | Three 512-token coding/analysis prompts: 52.67 / 37.01 / 46.78; all hit the reasoning-token cap |
| 32K cold PP | **435.85 tok/s** | 32,722 prompt tokens, no cache, 75.08 s TTFT, one generated token |
| 128K cold PP | **219.03 tok/s** | 131,024 prompt tokens, no cache, 598.20 s TTFT, one generated token |
| 128K exact-replay decode | **38.67 native / 39.14 stream tok/s** | 131,008 / 131,024 prompt tokens reused; 32 output tokens; 25 / 49 drafts accepted; 510.8 ms TTFT |
| Constrained 128K Metal allocations | **23.39 GiB session peak** | Includes an intentionally cancelled divergent-prefix request; hard Metal ceiling 40 GiB, memory pressure stayed normal |
| Default four-stream IFBench Metal allocations | **42.11 GiB peak** | Automatic memory plan, therefore above the project's 40 GiB product ceiling under this counter |

The 128K decode row is a directional 32-token sample, not a long-output
distribution. At 131,024 input tokens only 48 tokens remained in the configured
window, so a longer completion would require a slightly shorter prompt and a
second cold prefill. Exact replay reused 131,008 tokens. A prompt truncated in
the middle and given a different suffix did not hit Splash's cache and began a
full re-prefill; that request was cancelled after 47,104 rows.

## Comparison with retained VQ v1

| Metric | Splash 27B Q4 + DFlash 2 | Flash-Next VQ v1 |
|---|---:|---:|
| HumanEval original tests | **154/164** | 146/164 native MTP; 147/164 target-only |
| IFBench first 30, no-thinking | **19/30** | 18/30 native MTP |
| Ordinary measured decode | 80.00 median on HumanEval | 46.93 aggregate on HumanEval |
| Favorable speculative fixture | 94.23 | 57.94 |
| 32K context decode | not isolated in this run | 22.68 |
| 128K context | 219.03 PP / 38.67 native decode | unqualified |
| Memory | 17.86 GiB fixed Metal plan; 23.39 GiB constrained Metal peak | 36.3--39.6 GiB qualified process footprint |

The HumanEval prompt transport differs: Splash exposes chat completions but not
the raw completions endpoint used by the historical 27B assistant-prefill run.
Both rows use the same original tests and EvalPlus 0.3.1 complete-program
sanitizer, so the result establishes strong quality and no speculative loss;
the four-pass improvement over the historical 150/164 27B row must not be
attributed solely to weights or runtime.

Splash's `/status` memory values are Metal allocator counters, whereas the VQ
row uses macOS task footprint. They are not a like-for-like RAM measurement.
An idle post-start sample reported 1.45 GB system footprint for the native
engine process (2.53 GB process-lifetime peak), while Splash reported a much
larger fixed Metal plan; file-backed zero-copy weights explain part of the
difference. No macOS process-footprint sampler ran throughout the 128K request,
so 23.39 GiB must not be relabelled as total process RAM.

IFBench used four-way rolling admission instead of the VQ row's serial native
MTP run. Temperature zero makes the quality comparison useful, but it is not a
byte-for-byte scheduler A/B. The aggregate speed difference is nevertheless
large enough that scheduling detail cannot explain it away.

## Engineering consequences

1. The useful part of Splash is not a generic configuration preset. It combines
   exact-shape Metal kernels, a model-specific DFlash 2 draft, fused
   draft/verify/accept/state updates, Q8 paged KV, rolling admission, and a
   machine-specific memory plan.
2. The existing 27B draft cannot serve Flash-Next. A Flash-Next DFlash 2 path
   requires training a new target-aware draft and adding compatible hidden-state
   taps. Runtime fusion can still inform the existing native-MTP path.
3. GSQ-RCO Q2_0 remains a useful Flash-Next experiment only if retaining the
   Flash-Next architecture is a product requirement. Splash now supplies a much
   smaller, locally verified daily-use alternative, so a 66.4 GB GSQ download is
   no longer needed merely to obtain a high-quality fast local model.
4. If GSQ-RCO is tested, test only Q2_0 first: it is the only published variant
   plausibly near the 40 GiB process ceiling and it directly tests whether
   avoiding lookup-table quantizers removes the VQ decode bottleneck.

## Tooling change

`run_humaneval.py` now supports a chat-only complete-program mode and maps
Splash request metrics into the existing generation-rate field.
`long_context_benchmark.py` accepts an explicit model ID and normalizes Splash
latency metrics. Existing protocols and defaults are unchanged.
