# Hot expert paging: policy and grouped prefill experiment

This is an experimental `qwen38-paged-model-probe` path, not a replacement for
the resident server. The fast resident MoE kernels are not yet integrated with
the paged storage layout. Memory reduction alone does not make this path ready
for deployment.

## Policy

The cache retains frequently and recently used experts. Only requested experts
are read from the original safetensors on a miss; an in-flight GPU lease prevents
eviction. The frequency count halves every 4096 cache accesses. The former
64-access interval was shorter than a single 480-access decode step and could
forget frequency before the next token. Old popularity still expires; no expert
is permanently pinned because it was popular once.

A bounded 16384-entry ghost history retains only keys/frequencies/recency after
eviction, not weights. A returning expert recovers its decayed history. This is
a cache-policy correction, not a demonstrated end-to-end speedup. Grouped PP
counts one access per expert group, not every row within that group.

`--trace-policy` captures at most 200000 access/lease-boundary/budget events. CPU
replay compares 64 (without history), 4096, 8192 and 16384-access decay. The
4096 replay must match live misses and evictions. Fixed 16/24 GiB expert-cache
counterfactuals are also printed: these are neither process memory measurements
nor throughput predictions. Recorded routing is reused; storage, thermals,
workspace, MTP and the adaptive budget controller are not simulated.

## PP implementation and correctness

Tokens are grouped by their original routed expert, projected together, restored
to original token/rank order, then accumulated in the original BF16 rank order.
Eight expert groups share a completion fence. All original top-10 experts,
quantized weights and BF16 scales/biases are retained. The full-model probe
splits PP into at most 64-token chunks to respect the GatedDeltaNet batch limit.

In pinned MLX, `sorted=true` may select a grouped GEMM at 16 or more rows and
change rounding. That variant failed full-text logit hashes and was rejected.
Grouped paging uses `sorted=false` to preserve reference QMV arithmetic.

Layers 0/47 at 1/4/32/64 rows matched the reference exactly. Three full-model
plain-text prompts (66/68/68 tokens: English cache explanation, Python review,
Chinese meeting notes) also matched every FP32-converted BF16 logit hash:

| Prompt | Rowwise PP tok/s | Grouped PP tok/s | Parallel-read grouped tok/s |
| --- | ---: | ---: | ---: |
| Cache explanation | 3.319 | 2.916 | 2.645 |
| Python review | 3.725 | 3.936 | 3.818 |
| Chinese notes | 6.360 | 6.662 | 6.754 |

These are sequential single runs, not controlled confidence intervals. OS file
cache was not flushed and thermal/frequency state was not locked. All runs had
an eight-token greedy decode warmup, not a fully warmed expert cache. The first
two columns used packed decode; the last used unpacked decode, with identical
warmup routing/hash. PP timing includes chunk completion and copying/hashing
logits. These numbers must not be compared directly to long-context resident PP.

Model: Qwen3.8-Flash-Next-REAP-288-MLX-4bit-repacked-MTP-Q8-REAP288-L47,
Q4/group64 experts, BF16 quantization metadata, MTP disabled; Apple M5 Pro,
64 GiB. PP loads dominate: the grouped serial runs spent 18.88/11.81/4.98 seconds
in the loader out of 22.63/17.28/10.21 total seconds. Three temporary reader
threads per expert did not establish a gain and remain opt-in (`--parallel-reads`).
Packed decode copies selected expert arrays and is also opt-in (`--packed-decode`),
not an established optimization. Production settings have not changed.

## Safety and limits

Use the private strict MLX allocator plus the external memory guard, not only
MLX's advisory limit. The tested 28/36 process target reserves a 24 GiB expert
steady cache and permits a 28 GiB expert burst. Before PP it returns to steady
capacity to free workspace. Full jobs require 40 GiB starting available memory,
stop below 12 GiB available, and have an external 36 GiB process ceiling plus
an internal 34 GiB early-stop threshold. This is a short-context budget, not a
safe long-context capacity claim.

The parallel full-text run peaked at 27.2 GiB and restored 27.0 GiB. A 256-token
serial/unpacked trace peaked at 30.0 GiB and restored 27.3 GiB, without OOM.
That run produced 3.094 tok/s excluding the first token (median step 0.294 s,
p95 0.569 s), 10536 misses and 470 evictions. All four recorded-budget policy
replays had the same misses, so it does not establish a frequency-policy gain.
Step 238 had zero new misses yet took 0.178 s. This is a single-step observation,
not a warm throughput benchmark, but it rules out SSD misses as the sole problem:
the paged compute path has substantial overhead even when all experts hit RAM.

Remaining work: test under meaningful cache pressure and topic changes, preserve
a hot working set across PP scans, and implement a pooled/slot-indexed GPU layout
that can use fused kernels without copying all selected weights each token.
Do not enable the paged path in the service until end-to-end latency justifies
the memory savings. No claim of reaching the resident decode or 600 PP target.

## Fixed-cap replay result

The clean-build rerun reproduced the 256-token hash and 10536 live misses.
Its measured decode was 2.828 tok/s (median 0.304 s, p95 0.826 s), with loader
time 52.44 s versus 45.10 s in the earlier run. This variation is another reason
not to label the earlier 3.094 result a speedup. Peak/restored footprint remained
30.0/27.3 GiB and minimum available memory was 17.5 GiB.

| Fixed expert cache only | Decay 64, no history | 4096 + history | 8192 + history | 16384 + history |
| --- | ---: | ---: | ---: | ---: |
| 16 GiB misses | 14475 | 14475 | 14476 | 14964 |
| 24 GiB misses | 10880 | 10880 | 10880 | 10880 |

All columns reuse identical routing and lease boundaries. Long-lived popularity
can be worse: 16384 raised misses by 3.38% at 16 GiB. Neither history nor 4096
aging established a benefit on this trace. The policy is bounded and testable,
but must not be sold as an optimized or universally best hot-set criterion.

Verification: fresh Release build; CPU cache tests, memory-guard tests, MLX
backend tests (including multi-array concatenate), real-tokenizer tests, layer
parity and full-model hash checks. No serving configuration was modified.
