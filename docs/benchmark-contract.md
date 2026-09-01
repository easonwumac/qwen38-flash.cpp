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
- MTP controlled median is at least 65 tok/s and p10 does not fall below the
  non-MTP path because unprofitable verification must fall back.
- Prefix-cache correctness and a context sweep through the maximum feasible size,
  with an explicit 262,144-token result or a measured hardware-memory limit.
- No leaks, sanitizer errors, request corruption, or unrecovered failed state.
