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

### Native concurrency control

The first 30 official prompts were generated through the native four-request
path with Q8 SSD PLE, MTP/thinking off, temperature zero, and max 4,096. The
official scorer reported **15/30 (50.00%)** prompt-level strict and loose and
17/33 (51.52%) instruction-level strict and loose, with zero request errors.
The older automatic-MTP serial artifact scores 11/30 (36.67%) on the same slice,
but differs in PLE/MTP configuration and is not a concurrency A/B.

A controlled rolling-admission A/B therefore used the same first 12 prompts,
same Q8 PLE, max 512, and MTP off. All 12 concurrent responses were byte-identical
to serial. Aggregate decode was 39.20 versus 38.35 tok/s, while end-to-end was
36.10 versus 36.34 tok/s; refill correctness is validated, but mixed-length
IFBench does not yet show an end-to-end throughput win.

### Five-row Qwen3.8-27B control

A small cross-check used original dataset indices 0, 10, 20, 30, and 40, which
cover multiple-keyword counts, an exact word position, number count, pronoun
count, and punctuation. Both models used the same official strict and loose
verifiers, temperature zero, thinking off, and a 4,096-token maximum. REAP used
Q4/group-64 target weights, Q8/group-32 SSD PLE, MTP off, and serial requests;
Qwen3.8-27B used affine Q4/group-64 in `mlx-vlm` 0.7.1 with four-way continuous
batching followed by one final request.

Both checkpoints scored **1/5 strict and 1/5 loose**, passing only the pronoun
count. Their five paired verdicts were identical. REAP generated 853 tokens at
a 38.28 tok/s per-request median and reached 38.81 GiB peak footprint. The 27B
control generated 2,104 tokens; its four-request batch reported 19.06 aggregate
tok/s, the final serial request 17.39 tok/s, and a 17.56 GB MLX peak. Batch and
serial throughput are not directly comparable.

This five-row slice is diagnostic rather than an accuracy estimate. In
particular, it shows no REAP-specific backend loss under this no-thinking
contract, but it does not reproduce Qwen's published 79.5: the Qwen model card
does not disclose the IFBench generation contract, thinking is enabled by
default, and its recommended sampling differs from IFBench's documented
temperature-zero protocol.

### Ten-row bounded-thinking control

The first ten rows of the same stratified selection (original indices 0, 10,
..., 90) were then tested with the Qwen-recommended thinking sampler:
temperature 1.0, top-p 0.95, top-k 20, seed 0, xhigh reasoning, a 2,730-token
thinking budget within a 4,096-token maximum, and only the final answer passed
to the official verifier. MTP was off. The 27B control used serial generation
because `mlx-vlm` 0.7.1 continuous batching ended all five initial probes before
closing the thinking block; those invalid outputs were rejected rather than
scored.

| Bounded-thinking control | Strict | Loose | Generated tokens | Median decode | Peak memory |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B affine Q4/group-64, `mlx-vlm` | **7/10 (70%)** | **7/10 (70%)** | 23,459 | 16.79 tok/s | 16.75 GB MLX |
| REAP-288 Q4 target + Q8 SSD PLE, custom engine | **5/10 (50%)** | **5/10 (50%)** | 29,174 | 36.86 tok/s | 38.88 GiB footprint |

The paired strict/loose verdicts were identical: 4 both-pass, 3 27B-only, 1
REAP-only, and 2 both-fail. Every serial 27B response closed its thinking block;
the custom engine also returned a final answer for every case. The result
demonstrates both that thinking lifecycle explains the earlier 27B 1/5 result
and that the REAP checkpoint has a real instruction-following deficit on this
small slice. Ten deliberately stratified cases remain too few to estimate the
300-row score, but 27B's 70% is directionally compatible with its published
79.5 rather than the misleading no-thinking result.

Niwaki 99B is not promoted as the quality default. A three-prompt pilot repeated
or exhausted its allowance in both this engine and a stock `mlx-vlm` generation
on the same checkpoint, consistent with the checkpoint author's documented
long/repetitive reasoning limitation.

### VQ-2.1bpw native quality gate

`TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw` revision
`64b0fb0f98a552d91fb9abd5531d547b2e78c8a8` retains all 512 experts and routes
top-10, but stores its expert projections with mixed VQ geometry and its PLE as
d8/K256 VQ rows. The 49,156,164,890-byte non-MTP artifact was verified against
its safetensors index before testing.

The current head of `mlx-lm` PR 1788 is not compatible with this converted
artifact's RMSNorm convention. Commit `ac83bb4` folds the Qwen `+1` only while
converting the official nested layout; this artifact already has flat keys but
still carries zero-centered non-gated norm weights. Current head therefore
loaded successfully but emitted repeated garbage. Using the pre-fold commit
`8a36d1ece51b34a2e76d9f7064a7f090334f39c4`, whose forward applies
`1 + weight`, restored an exact deterministic smoke response. The failed
current-head output is a runtime/checkpoint mismatch, not a model-quality
result.

On the same 30-row stratified no-thinking slice used by the REAP stock control,
native VQ scored **12/30 strict and 13/30 loose**, exactly tying stock REAP.
The paired loose result was 10 both-pass, 14 both-fail, 3 VQ-only, and 3
REAP-only. Three VQ responses reached 4,096 tokens; total generation was 18,456
tokens at 17.47 end-to-end tok/s across the two serial segments.

The stronger gate reproduced this engine's bounded-thinking lifecycle in an
offline `mlx-lm` runner: temperature 1.0, top-p 0.95, top-k 20, seed 0,
2,730-token reasoning budget, 4,096-token total maximum, the identical forced
suffix, serial requests, and MTP off. It scored **6/10 strict and 7/10 loose**
with 7/10 interventions and no missing final answers. Prompt-level loose paired
against REAP was 5 both-pass, 3 both-fail, and 2 VQ-only; paired against 27B was
6 both-pass, 2 both-fail, 1 VQ-only, and 1 27B-only. Thus the VQ checkpoint
beats REAP 7/10 to 5/10 and ties the local 27B control on this small slice.

The bounded VQ run generated 25,663 tokens in 1,421.0 seconds. Per-request
decode stayed near 18.1--18.5 tok/s; the memory guard recorded 46.1 GiB peak
physical footprint, 25.2 GiB peak RSS, and 6.1 GiB minimum available on the
64 GB M5 Pro. These are native-reference results, not product targets. They
justify implementing the VQ backend, but do not demonstrate the project's
PP, decode, or 40 GiB footprint goals.

Source: <https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw>

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

A subsequent run covered the first 30 public prompts with the corrected bounded
lifecycle, temperature 1.0, top-p 0.95, top-k 20, seed 0, xhigh effort, and a
4,096-token maximum. It scored **19/30 (63.33%)** at both prompt-level strict and
loose accuracy, and 21/33 (63.64%) at instruction level. There were no request
errors; two responses reached the length limit. Aggregate decode was 34.88 tok/s
and the per-request median was 35.46 tok/s. This slice contains count and word
constraints and is not an estimate of the full 300-row distribution, but it
demonstrates that the earlier non-thinking result materially understates the
checkpoint's usable instruction-following path.

Replacing only the row-addressable Q4/group-32 PLE with the original BF16 PLE
raised the same slice to **21/30 (70.00%)** at both prompt-level strict and loose
accuracy and 23/33 (69.70%) at instruction level. One response reached the
length limit. Aggregate decode was 36.84 tok/s and the per-request median was
37.25 tok/s across 81,843 generated tokens. The BF16 table occupies 95.37 GiB
on SSD versus 29.8 GiB for Q4; both are read on demand instead of being made
resident. The directional speed difference is not treated as a throughput
claim because these were separate serial runs.

An affine Q8/group-32 PLE control occupied 53.64 GiB and scored **20/30
(66.67%)** at both prompt-level strict and loose accuracy and 21/33 (63.64%) at
instruction level. All 30 requests completed without errors or reaching the
length limit. Aggregate decode was 35.73 tok/s and the per-request median was
36.07 tok/s across 78,555 generated tokens. Its score and storage fall between
Q4 and BF16; separate-run throughput differences are not treated as speedups.

## HumanEval runtime control

The REAP model card reports 91.5% HumanEval pass@1 but identifies only the 164
problems, one run per build, and unit-test verification; it does not publish its
prompt template, stopping rules, or generation harness. OpenAI HumanEval itself
defines prompts, reference completions, and tests, but not a chat wrapper or
generation stops. Results are consequently reported under two explicit
protocols rather than treated as directly interchangeable with the model-card
number.

The raw-completion control used greedy decoding, one sample, at most 512 new
tokens, and the EvalPlus direct-completion stop set. REAP used Q4/group-64
target weights and Q8/group-32 SSD PLE in the custom engine; the dense
Qwen3.8-27B control used the `mlx-community` affine Q4/group-64 checkpoint in
`mlx-vlm` 0.7.1. MTP was disabled in both runs. The official HumanEval tests ran
inside a network-denied macOS sandbox.

| Raw-completion control | pass@1 | Median decode |
|---|---:|---:|
| REAP-288 Q4 target + Q8 SSD PLE, custom engine | **133/164 (81.10%)** | **38.13 tok/s** |
| Qwen3.8-27B affine Q4/group-64, `mlx-vlm` | **132/164 (80.49%)** | **17.70 tok/s** |

The standard EvalPlus instruction-model protocol instead asks for a
self-contained Python script, pre-fills an empty `<think>` block and the
opening Python fence, generates greedily for at most 768 tokens, and sanitizes
the complete solution before testing. EvalPlus 0.3.1 was used here. For the
test bridge, the sanitized solution was evaluated as the entire candidate
program while the original HumanEval tests remained unchanged.

| EvalPlus no-thinking chat control | pass@1 | Median decode | Runtime peak |
|---|---:|---:|---:|
| REAP-288 Q4 target + Q8 SSD PLE, custom engine | **149/164 (90.85%)** | **38.94 tok/s** | not captured |
| Qwen3.8-27B affine Q4/group-64, `mlx-vlm` | **150/164 (91.46%)** | **17.21 tok/s** | **17.29 GB MLX** |

The paired chat result was 143 both-pass, 6 REAP-only, 7 27B-only, and 8
both-fail. A one-problem difference on the exact same benchmark contract rules
out the custom backend as the source of the previously apparent large quality
collapse. It also nearly reproduces the REAP card's 91.5% claim, although the
card's unpublished protocol prevents claiming exact reproduction. The custom
server did not yet accept request-level multi-token stop strings, so some REAP
responses continued into generated self-tests; its generated-token count is
not directly comparable with the better-stopped `mlx-vlm` run.

Turning verified MTP off on the raw Q8-PLE REAP control changed the score from
132/164 to 133/164. The paired result was 124 both-pass, 8 MTP-only, 9
non-MTP-only, and 23 both-fail. This excludes a systematic MTP quality loss,
while confirming that deterministic trajectories can change around close
logits. Exact EvalPlus raw stop strings applied to the prior MTP outputs left
their 132/164 score unchanged.

The direct MLX runner also supports continuous batching through
`--concurrency`. A directional first-four-problem smoke test at concurrency 4,
raw prompts, greedy decoding, MTP off, and a 64-token limit generated 256 tokens
at **50.22 aggregate tok/s** and a **17.68 GB MLX peak**. All four reached the
limit, so this is only a scheduler/memory check, not a quality result or a
single-stream decode claim. Future 27B bulk evaluations may use this mode when
their protocol tolerates batched stopping.

Sources:

- <https://github.com/openai/human-eval>
- <https://github.com/evalplus/evalplus/blob/master/docs/cli.md>
- <https://github.com/evalplus/evalplus/blob/master/evalplus/provider/utility.py>
- <https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit>
- <https://huggingface.co/mlx-community/Qwen3.8-27B-4bit>

### Earlier same-checkpoint controls

Before the exact EvalPlus protocol was identified, the project's conservative
raw boundaries gave the following same-checkpoint runtime comparison.

After removing accidental protocol suffixes from otherwise raw completions, the
custom engine scored **127/164 (77.44%)**. Current stock `mlx-vlm` commit
`1ecf1ecdd28af102eded679be0daa5c76ab2a068`, using the identical local checkpoint
and identical row-addressable PLE, scored **130/164 (79.27%)**. The paired result
was 118 both-pass, 25 both-fail, 9 custom-only, and 12 stock-only. The 1.83-point
gap excludes this backend as the explanation for a large checkpoint-quality
loss, while the changed paired verdicts show that exact logits parity still
matters for deterministic benchmark reproduction.

The custom engine's median decode was 50.01 tok/s versus 31.46 tok/s for stock;
stock reported a 43.50 GB MLX peak. A non-thinking chat prompt scored only
110/164 (67.07%), but its union with the raw-completion successes covered
146/164 (89.02%). That oracle union is not a valid pass@1 score; it demonstrates
large prompt sensitivity and why the model card's unpublished harness prevents
an exact reproduction claim.

An otherwise identical custom-engine run replaced only the Q4 PLE with the
original BF16 table. It scored **133/164 (81.10%)**: 118 problems passed under
both PLE precisions, 15 passed only with BF16, 9 passed only with Q4, and 22
failed under both. Median decode was 48.86 tok/s versus 50.01 tok/s with Q4, a
2.3% reduction. PLE quantization therefore has a measurable but bounded effect:
BF16 recovered 3.66 pass@1 points, not the much larger gap caused by using the
checkpoint's non-thinking instruction-following path.

With automatic MTP enabled, the affine Q8/group-32 table scored **132/164
(80.49%)** with a 46.55 tok/s
median decode rate. Against Q4 it retained 121 common passes, lost 6, and gained
11; against BF16 it retained 122 common passes, lost 10, and gained 11. A fixed
2,048-row reconstruction sample measured Q8 RMSE `3.59e-5` versus BF16, about
17 times lower than Q4's `6.16e-4`. Q8 therefore retains nearly all measured
BF16 quality while reducing the SSD table from 95.37 to 53.64 GiB; it does not
reduce resident memory because all three formats are already demand-read.

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
