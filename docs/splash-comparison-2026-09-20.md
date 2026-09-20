# Splash 27B and 35B local evaluation

Date: 2026-09-20

Status: external daily-use runtimes, not part of `qwen38-flash.cpp`.

## Decision

Keep both Splash packages. Use `Qwen3.6-35B-A3B-Splash` for routine agent work
and long contexts: it is about 3--5x faster in the paired local tests. Use
`Qwen3.8-27B-Splash` when maximum local instruction-following or coding-test
accuracy matters. The 35B MoE does not replace the dense 27B on quality, but it
is the better default for interactive throughput.

Both models use a model-specific DFlash 2 draft. Draft tokens are verified by
the target, so speculation changes performance rather than the target's output
distribution. The published packages are fixed-layout Splash artifacts, not
general MLX checkpoints.

## Test environment and protocol

- Hardware: Apple M5 Pro, 20-core GPU, 64 GiB unified memory (`Mac17,8`).
- OS: macOS 27.0 (`26A428`); Splash 1.0.
- Server ceiling: `--max-memory 40G --max-context 128K`.
- IFBench: pinned first 30 OOD rows, dataset SHA-256
  `7f184d58ae7f957da3d81f4255ce2e98911e6f7047803f2ea8de3e3f83d5a81d`,
  temperature 0, seed 0, reasoning enabled, 16,384 output-token limit, rolling
  concurrency four, official strict and loose scorer.
- Fixed throughput matrix: the same first 12 IFBench prompts, reasoning enabled,
  temperature 0, 1,024 output-token limit, concurrency 1/2/4.
- HumanEval and EvalPlus MBPP+: temperature 0, reasoning disabled, 768 output
  tokens, one sample per problem, EvalPlus 0.3.1 execution and sanitization.
- PP: one cold deterministic prompt. Long-context decode is an exact-prefix
  replay with 32 output tokens; it is directional, not a long-output distribution.

The model cards report 35B reasoning as an on/off switch and 27B reasoning as
`low`, `medium`, `xhigh`, or `none`. Local thinking runs used `xhigh`, which
enables reasoning on both packages.

## Quality

| Benchmark | 27B dense Splash | 35B-A3B MoE Splash | Winner |
|---|---:|---:|---|
| IFBench first 30, thinking, strict | **30/30 (100%)** | 27/30 (90%) | 27B |
| IFBench first 30, thinking, loose | **30/30 (100%)** | 27/30 (90%) | 27B |
| HumanEval original tests | **154/164 (93.90%)** | 149/164 (90.85%) | 27B |
| MBPP base tests | **340/378 (89.95%)** | 335/378 (88.62%) | 27B |
| MBPP+ tests | **294/378 (77.78%)** | 284/378 (75.13%) | 27B |

The corrected 27B thinking result supersedes the earlier 19/30 row, which used
only 4,096 output tokens and truncated 11 cases. In the paired 16K run neither
model had an API error or a length-limited response. The 35B IFBench failures
were rows 0, 2, and 10. HumanEval had six length-limited generations on 27B and
nine on 35B; MBPP had ten and eleven respectively.

## Decode and concurrency

The fixed matrix below avoids comparing a favorable prompt from one model with
a difficult prompt from the other.

| Fixed 12-prompt thinking run | 27B aggregate tok/s | 35B aggregate tok/s | 35B / 27B |
|---|---:|---:|---:|
| B=1 | 45.71 | **139.56** | **3.05x** |
| B=2 | 81.12 | **188.74** | **2.33x** |
| B=4 | 102.61 | **255.19** | **2.49x** |
| B=4 / B=1 scaling | 2.24x | 1.83x | -- |

B=4 per-request median decode was 26.67 tok/s on 27B and 68.51 tok/s on 35B.
Aggregate throughput improves substantially, but each individual stream slows
under contention.

The full first-30 quality run produced much longer and model-dependent outputs,
so it is reported separately: 27B delivered 85.98 aggregate tok/s over 122,909
completion tokens; 35B delivered 326.42 aggregate tok/s over 132,716 completion
tokens. HumanEval median decode was 80.00 versus 271.87 tok/s, and MBPP median
decode was 60.38 versus 230.98 tok/s. These task rates agree with the fixed
matrix: 35B's speed advantage is large and repeatable.

## Prefill, long context, and memory

| Measurement | 27B dense Splash | 35B-A3B MoE Splash | 35B / 27B |
|---|---:|---:|---:|
| 32K cold PP | 435.85 tok/s | **2,410.55 tok/s** | **5.53x** |
| 131K cold PP | 219.03 tok/s | **1,161.30 tok/s** | **5.30x** |
| 131K exact-replay stream decode | 39.14 tok/s | **135.85 tok/s** | **3.47x** |
| Fixed runtime plan | 17.86 GiB | 20.99 GiB | -- |
| Constrained 128K session peak | 23.39 GiB | **21.78 GiB** | -- |

The 35B replay reused 131,008 of 131,030 input tokens and regenerated only the
22-token suffix. Splash's cumulative native decode counter for that two-request
session was 152.32 tok/s with 61.9% draft acceptance. The 35B extended quality
campaign eventually reached a 39.22 GiB Metal-allocation peak while remaining
inside the 40 GiB ceiling; it reported zero capacity and Metal failures. The
27B paired thinking and speed campaign peaked at 28.14 GiB under the same
ceiling. These are Splash allocator counters, not macOS task-footprint samples.

## Installed size and package geometry

| Package | Local cache | Target | DFlash 2 draft | Vision |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Splash | about 16 GiB | 14.1 GiB | 1.2 GiB, 5 layers | 0.9 GiB |
| Qwen3.6-35B-A3B Splash | about 20 GiB | 18.2 GiB | 0.5 GiB, 6 layers | 0.8 GiB |

Sources: [35B package](https://huggingface.co/incoai/Qwen3.6-35B-A3B-Splash),
[27B package](https://huggingface.co/incoai/Qwen3.8-27B-Splash), and
[Splash engine](https://github.com/incoai/splash).

## Pi configuration and operation

Pi uses the OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`. Splash
1.0 serves one loaded model on a fixed port, so selecting a model in Pi does not
hot-swap the server. Start the desired backend first:

```bash
# Fast daily default
splash serve --model incoai/Qwen3.6-35B-A3B-Splash \
  --max-memory 40G --max-context 128K --no-webui

# Or stop it with Ctrl-C and start the quality-oriented model
splash serve --model incoai/Qwen3.8-27B-Splash \
  --max-memory 40G --max-context 128K --no-webui
```

Then launch Pi with the matching model:

```bash
pi --provider splash --model incoai/Qwen3.6-35B-A3B-Splash --thinking high
pi --provider splash --model incoai/Qwen3.8-27B-Splash --thinking high
```

Use `--thinking off` for short code-generation controls. A non-interactive Pi
smoke request against the 27B server returned the requested `PI_OK`, confirming
the custom-provider route.
