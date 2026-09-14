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

A three-prompt bring-up initially appeared to score 2/3, but that result is
invalid: nucleus filtering had been applied after renormalizing the top-20
candidates. The corrected implementation matches `mlx-vlm`'s operation order:
full-vocabulary untempered probabilities select the nucleus, top-k bounds the
remaining candidates, and temperature applies only to the categorical draw.

Repeating the same REAP-288 xhigh pilot with temperature 1.0, top-p 0.95,
top-k 20, seed 0, and a 4,096-token maximum scored **0/3 strict and loose**.
Every response exhausted all 4,096 tokens without closing `</think>`; aggregate
decode was 34.84 tok/s (per-request median 34.66, range 33.86--36.06), with a
39.1 GiB peak footprint. The earlier 2/3 result and its throughput must not be
used as quality evidence. The corrected run instead exposes the need for an
explicit thinking budget and final-answer continuation.

That lifecycle is now implemented inside the engine. For a 4,096-token xhigh
request it reserves the final third and allows 2,730 generated reasoning tokens.
If the budget is reached, or EOS is sampled while reasoning is still open, the
engine replaces the stop with Qwen's documented early-stop phrase and closing
marker, then continues from the same KV state. The injected tokens do not
consume RNG draws. Thinking requests that omit sampling fields now inherit the
checkpoint defaults: temperature 1.0, top-p 0.95, and top-k 20.

On the same REAP-288 three-prompt pilot, all three requests required the
intervention and returned final answers. Strict and loose accuracy recovered to
**2/3**, aggregate decode was 35.81 tok/s (range 35.35--36.27), and the guarded
peak footprint was 39.0 GiB. This validates the lifecycle mechanism; three
prompts remain too small for a model-quality estimate.

### Niwaki 113B pilot

`Qwen3.8-Flash-Next-113B-A5B-Niwaki-3bit-mlx` was tested on the same first
three IFBench prompts. The engine used the 113B routed/backbone weights with
the retained REAP tokenizer and Q4 SSD PLE because the checkpoint has no
`merges.txt` and its native 2-bit PLE is stored as safetensors shards rather
than the engine's row-major SSD format. This is therefore a clearly labelled
hybrid, not checkpoint parity. MTP was off.

The non-thinking pilot scored 0/3 strict and loose. Warm decode was 41.49 and
42.04 tok/s after a 32.98 tok/s cold request. The original sampled result is
invalid because it used the incorrect top-p ordering described above. With
corrected sampling, xhigh thinking at temperature 1.0, top-p 0.95, top-k 20,
seed 0, and a 4,096-token maximum also scored 0/3. One request reached the
length limit, one emitted EOS before `</think>`, and one produced a final answer
that missed the exact constraints. Aggregate decode was 36.72 tok/s. The
corrected-sampling session peaked at 26.8 GiB footprint and 25.2 GiB RSS, with
16.9 GiB minimum available memory.

A low-effort control closed all three thinking blocks in 335--521 tokens but
still scored 0/3; lowering temperature to 0.6 made all three requests exhaust
4,096 tokens. These tiny controls establish lifecycle sensitivity, not an
accuracy estimate.

The bounded lifecycle made all three Niwaki requests return final answers, but
the official strict and loose scores remained 0/3. Aggregate decode was 40.07
tok/s and guarded peak footprint was 26.7 GiB. A separate 0.3 frequency-penalty
probe shortened the three completions from 7,793 to 1,919 total tokens, but also
remained 0/3. Frequency penalty is therefore exposed as a request option but is
not an automatic default, especially because IFBench deliberately requires
exact keyword repetition.

A stock `mlx-vlm` 0.7.0 / MLX 0.32.2 control used the checkpoint's native
2-bit PLE on the first prompt. Sampled xhigh thinking did not emit
`</think>` or a final answer within 4,096 tokens; it decoded at 29.13 tok/s and
reported 43.58 GB peak memory, while the external guard measured 41.3 GiB peak
footprint. A stock non-thinking 512-token control also repeated and failed the
keyword constraint. These controls do not prove hybrid numerical parity, but
they show that the checkpoint itself also fails this pilot rather than exposing
a quality result hidden by the custom engine. Tokenization of all three prompts
is identical between the retained and Niwaki tokenizers. The 99B and 113B native
PLE tensors are bit-identical, and sampled rows from the external Q4 PLE have
roughly 0.90--0.93 cosine similarity with the native Q2 rows, consistent with
the same base table at different precision. A one-token stock/native-Q2 versus
engine/external-Q4 trace selected the same top token with logits 15.0 and
14.875 and stayed close through all 48 layer checksums. This rules out a gross
tokenizer, PLE, or layer-layout error, but is not full-sequence numerical parity.
A full 300-prompt Niwaki run is not justified until a larger pilot clears this
gate.

The engine now also reads the checkpoint's native paired Q2/group-128 PLE
directly from its 128 indexed shard tensors. An independent decoder matched two
sampled 2,560-value gathers exactly (checksums `-0.38299560546875` and
`-0.17431640625`). Repeating the same three-prompt bounded-thinking gate with
that native PLE still failed every requested keyword-count constraint. Aggregate
decode was 37.76 tok/s (36.65--38.70), and the guarded peak footprint was 26.6
GiB. The external Q4 PLE is therefore not the cause of this pilot's quality
failure; native Q2 changes the trajectory and costs about 5.8% throughput here,
but does not recover instruction following.

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
