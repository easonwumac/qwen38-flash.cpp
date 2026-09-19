# Benchmark contract

Performance gates compare this engine with the retained reference using the same
machine, model files, prompt token IDs, generated-token limit, greedy/sampling
settings, context state, cache state, MTP configuration, and residency policy.

Every result records:

- hardware model, RAM, OS build, power source, and relevant power mode;
- engine commit and clean/dirty state;
- checkpoint identity, quantization, resident bytes, KV/state cache types;
- prompt corpus hash, input tokens, output tokens, and context length;
- prefill time/rate, time to first token, inter-token latency distribution,
  decode rate, end-to-end rate, and peak resident memory;
- MTP draft depth, proposed/accepted tokens, acceptance distribution, verifier
  cost, and fallback count;
- warm-up policy, run count, median, p10/p90, thermal observations, and failures.

## Fixed workload families

1. multilingual natural prose;
2. mixed source code;
3. tool/JSON generation;
4. long-context retrieval at increasing context lengths;
5. adversarial low-MTP-acceptance prompts.

Repeated text, cached prompts, or a single high-acceptance coding prompt may be
reported as diagnostics but cannot prove a universal acceptance gate.

## Daily-use acceptance and unfinished targets

On 2026-09-19 the user accepted the existing VQ configuration for daily use and
closed the optimization phase. See the [closeout](daily-use-2026-09-19.md).
This is not a claim that the original performance release gates passed.
The current VQ stretch targets are 600 PP / 40 target-only / 60 mixed-workload
MTP tok/s under 40 GiB; they remain unfulfilled as a combined requirement.
Earlier 45/65 decode thresholds belonged to the historical optimization plan,
not the measured daily-use baseline. Historical REAP numbers do not qualify VQ.

## Gates for future performance promotion

- Greedy output/state parity on committed fixtures for numerical-preserving
  changes. A changed numerical path must separately pass paired quality gates;
  current batched MTP is not promised target-only token parity.
- Quality suite does not regress from the retained reference.
- Report paired non-MTP controlled median and p10/p90; 40 tok/s remains a target,
  not a condition already satisfied by the accepted daily baseline.
- Report paired cold/warm prefill with no regression on the retained corpus.
  600 prompt tok/s through 8K and at 32K remains an optimization target; larger
  contexts publish measured PP/TTFT and memory degradation instead of
  extrapolating the short-context result.
- MTP mixed-workload median targets 60 tok/s and p10 should not fall below the
  non-MTP path; measure fallback overhead rather than assuming it is free.
- Prefix-cache correctness and a context sweep through the maximum feasible size,
  with an explicit 262,144-token result or a measured hardware-memory limit.
- No leaks, sanitizer errors, request corruption, or unrecovered failed state.

## Continuous-decode gate

Cross-request batching is compared with four sequential requests using the same
live server, prompts, sampler seeds, output limits, and MTP setting. Promotion
requires exact per-request token parity, repeatability after reversing arrival
order, higher aggregate decode throughput, and a peak footprint below 40 GiB on
the 64 GB validation Mac. Per-request tok/s is not presented as aggregate tok/s.

The following are historical controls, not current VQ promotion evidence.
The retained warm directional check uses REAP-288 Q4/group-64, standalone Q8
PLE, the `speed` profile, MTP/thinking off, four distinct short prompts, and 128
greedy output tokens each. It measured 45.70 aggregate decode tok/s versus four
serial requests at 41.48--41.68 tok/s, with exact output parity and a 38.9 GiB
peak footprint. The corresponding layer-major microbenchmark measured 45.86
versus 38.41 aggregate tok/s (1.194x) across 4 x 14 measured steps. These are
directional concurrency results, not single-stream decode claims.

A rolling-admission IFBench control used the first 12 official prompts, a
512-token maximum, and otherwise the same MTP-off runtime. Four slots produced
3,665 tokens at 39.20 aggregate decode tok/s versus 38.35 serial (+2.23%); all
12 responses were byte-identical. End-to-end throughput was 36.10 versus 36.34
tok/s (-0.64%) because new-request prefill pauses surviving decode rows. The
four-slot peak footprint was 39.0 GiB. This validates refill correctness and a
small decode gain, not a mixed-length end-to-end throughput claim.
