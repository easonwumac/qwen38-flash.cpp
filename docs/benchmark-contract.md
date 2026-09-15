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

## Release gates

- Greedy output parity on committed fixtures.
- Quality suite does not regress from the retained reference.
- Non-MTP controlled median is at least 45 tok/s.
- Warm exact prefill is at least 600 prompt tok/s through 8K. 32K remains a
  600 PP/s optimization target; larger contexts publish measured PP/TTFT and
  memory degradation instead of extrapolating the short-context result.
- MTP controlled median is at least 65 tok/s and p10 does not fall below the
  non-MTP path because unprofitable verification must fall back.
- Prefix-cache correctness and a context sweep through the maximum feasible size,
  with an explicit 262,144-token result or a measured hardware-memory limit.
- No leaks, sanitizer errors, request corruption, or unrecovered failed state.

## Continuous-decode gate

Cross-request batching is compared with four sequential requests using the same
live server, prompts, sampler seeds, output limits, and MTP setting. Promotion
requires exact per-request token parity, repeatability after reversing arrival
order, higher aggregate decode throughput, and a peak footprint below 40 GiB on
the 64 GB validation Mac. Per-request tok/s is not presented as aggregate tok/s.

The retained warm directional check uses REAP-288 Q4/group-64, standalone Q8
PLE, the `speed` profile, MTP/thinking off, four distinct short prompts, and 128
greedy output tokens each. It measured 45.70 aggregate decode tok/s versus four
serial requests at 41.48--41.68 tok/s, with exact output parity and a 38.9 GiB
peak footprint. The corresponding layer-major microbenchmark measured 45.86
versus 38.41 aggregate tok/s (1.194x) across 4 x 14 measured steps. These are
directional concurrency results, not single-stream decode claims.
