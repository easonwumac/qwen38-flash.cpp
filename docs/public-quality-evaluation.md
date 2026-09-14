# Public quality evaluation

## IFBench

The first public quality gate uses the official single-turn IFBench OOD test
and its unmodified strict/loose verifiers. This is the most directly runnable
benchmark in the published Qwen3.8-27B text table: it is machine graded, public,
and its authors specify temperature zero and prompt-level loose accuracy.

The Qwen3.8-27B model card reports **79.5** on IFBench. That number is useful as
a benchmark target, not as a checkpoint-parity claim: this engine run used the
text-only Qwen3.8-Flash-Next REAP-288 MoE checkpoint rather than the dense,
multimodal 27B checkpoint. The Qwen card also does not publish the IFBench
thinking mode or generation limit used for its table.

Sources:

- <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>
- <https://github.com/allenai/IFBench>

### Reproduction contract

- IFBench commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`
- 300-row `IFBench_test.jsonl`, SHA-256
  `d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a`
- official `run_eval` strict and loose verifiers, single-turn test
- REAP-288 affine Q4/group-64 target and its Q8 MTP head
- target config SHA-256
  `bd5d90bb451466fa3812b71a27717ba7d928de1ffae2b0bce061ce6ddb270a0f`
- engine commit `a912439954fd2f6f917d6c9a6dfd56c9d28d4d38`
- temperature 0, thinking off, maximum 4,096 generated tokens, serial requests
- default automatic `speed` runtime, Q8 KV at 8,192/2,048, automatic MTP,
  no prefix-cache hits
- Apple M5 Pro MacBook Pro, 18 CPU cores, 64 GB unified memory, macOS 26.5
  (25F71), AC power, no active thermal control

Run generation with:

```bash
python3 devtools/run_ifbench.py \
  --input /path/to/IFBench/data/IFBench_test.jsonl \
  --responses /path/to/results/reap-responses.jsonl \
  --artifact /path/to/results/reap-telemetry.json \
  --no-thinking --max-tokens 4096
```

Then run the official scorer:

```bash
python -m run_eval \
  --input_data=data/IFBench_test.jsonl \
  --input_response_data=/path/to/results/reap-responses.jsonl \
  --output_dir=/path/to/results/eval
```

### Result

| Metric | Result |
|---|---:|
| Prompt-level loose accuracy | **39.67% (119/300)** |
| Instruction-level loose accuracy | **43.02% (148/344)** |
| Prompt-level strict accuracy | **34.67% (104/300)** |
| Instruction-level strict accuracy | **38.37% (132/344)** |
| API/request errors | **0/300** |
| EOS / length-limited responses | 291 / 9 |
| Generated tokens | 116,990 |
| Aggregate decode | **37.46 tok/s** |
| Per-request decode | median 35.21, p10 29.24, p90 40.18 tok/s |
| Aggregate prefill | 193.50 prompt tok/s |
| Per-request prefill | median 166.77, p10 147.00, p90 277.43 tok/s |
| MTP acceptance | 15,339 / 21,932 proposals, **69.94%** |
| Guarded memory | 40.8 GiB peak footprint, 33.9 GiB peak RSS, 8.7 GiB minimum available |

The per-request prefill rates are dominated by very short prompts and include
fixed request/graph overhead; they are not comparable to the 8K/128K PP gates.
Nine responses used the complete 4,096-token allowance; two nevertheless passed
the loose verifier. The 39.67 result is 39.83 points below Qwen3.8-27B's
published 79.5, but the gap cannot be assigned to the runtime without a
same-checkpoint stock-runtime run. It currently gates any claim that REAP-288 is
a high-quality general instruction-following replacement for Qwen3.8-27B.

Niwaki 99B is not promoted as the quality default. A three-prompt pilot repeated
or exhausted its allowance in both this engine and a stock `mlx-vlm` generation
on the same checkpoint, consistent with the checkpoint author's documented
long/repetitive reasoning limitation.

### Thinking-mode qualification

Thinking results must not reuse the temperature-zero contract above. Qwen's
published guidance warns that greedy thinking can degrade quality and repeat
indefinitely. The server now implements request-level temperature, top-p,
top-k, and deterministic seed controls. Sampling selects the candidate set on
the accelerator and transfers only the bounded top-k logits for the seeded CPU
draw. Because the retained MTP and persistent backends are greedy, the runtime
automatically bypasses them for sampled requests rather than applying invalid
speculative acceptance semantics.

A three-prompt bring-up used REAP-288, xhigh thinking, temperature 1.0, top-p
0.95, top-k 20, seed 0, maximum 4,096 generated tokens, and the same machine
and official scorer as above. Strict and loose prompt accuracy were both 2/3.
Two responses closed normally at 3,982 and 2,060 tokens; the third reached the
4,096-token limit. Aggregate decode was 35.09 tok/s (per-request median 35.89,
range 33.79--36.20). This pilot validates the sampling and response-splitting
path; three prompts are not an estimate of full-suite accuracy.

### Niwaki 113B pilot

`Qwen3.8-Flash-Next-113B-A5B-Niwaki-3bit-mlx` was tested on the same first
three IFBench prompts. The engine used the 113B routed/backbone weights with
the retained REAP tokenizer and Q4 SSD PLE because the checkpoint has no
`merges.txt` and its native 2-bit PLE is stored as safetensors shards rather
than the engine's row-major SSD format. This is therefore a clearly labelled
hybrid, not checkpoint parity. MTP was off.

The non-thinking pilot scored 0/3 strict and loose. Warm decode was 41.49 and
42.04 tok/s after a 32.98 tok/s cold request. Sampled xhigh thinking used
temperature 1.0, top-p 0.95, top-k 20, seed 0, and a 4,096-token maximum. All
three requests reached EOS in 302--636 tokens and aggregate decode was 40.25
tok/s, but their final answers omitted the requested exact keyword counts and
also scored 0/3. The combined guarded engine session peaked at 26.6 GiB
footprint and 26.0 GiB RSS, with 23.2 GiB minimum available memory.

A stock `mlx-vlm` 0.7.0 / MLX 0.32.2 control used the checkpoint's native
2-bit PLE on the first prompt. Sampled xhigh thinking did not emit
`</think>` or a final answer within 4,096 tokens; it decoded at 29.13 tok/s and
reported 43.58 GB peak memory, while the external guard measured 41.3 GiB peak
footprint. A stock non-thinking 512-token control also repeated and failed the
keyword constraint. These controls do not prove hybrid numerical parity, but
they show that the checkpoint itself also fails this pilot rather than exposing
a quality result hidden by the custom engine. A full 300-prompt run is not
justified until a larger pilot clears this gate.

## Other published Qwen3.8-27B rows

GPQA Diamond and LiveCodeBench v6 remain pending until their full harness and
prompt contracts are frozen. The engine now supports their required sampling
controls, but sampling support alone is not benchmark parity.
Terminal-Bench, SWE-bench Pro, NL2Repo, and DeepSWE require an agent harness and
tool loop; HLE additionally uses an external GPT-4o judge. None should be
reported as same-protocol results until those dependencies and sampling
semantics are implemented and frozen.

## Concurrency finding

The HTTP layer already accepts four connections, but inference remains one
thread-affine MLX queue. A four-engine prototype was rejected: idle startup
already reached 39.1 GiB footprint / 38.7 GiB RSS, and four simultaneous short
requests reached 49.6 / 48.1 GiB with only 3.5 GiB available before the guard
terminated the server. Four-way inference under the 40 GiB product ceiling must
therefore use single-model continuous batching with shared weights/qmeta, not
four replicated engine objects.
