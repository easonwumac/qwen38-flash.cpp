# SSD idle checkpoint policy and block attention feasibility

## SSD-backed request-boundary release

When SSD prefix caching is enabled, NativeEngine now drops its persistent RAM
checkpoint at successful request completion if that checkpoint is confirmed
SSD-backed. Previously the extended conversation checkpoint survived in RAM
between requests even after persistence. Next reuse takes the existing SSD
load_longest path. No change to target weights, arithmetic, or SSD byte capacity.

The retry save of an original prompt now tracks its boolean outcome on the
matching RAM entry as well. Mark it unbacked before writing, so exceptions and
capacity refusals retain the RAM fallback. With SSD disabled, RAM behavior is
unchanged. This is not a hard RAM cap under SSD failure and does not eliminate
live per-request KV. Released allocations may remain in the MLX allocator cache.
SSD reload can increase next-request TTFT; no claim of unchanged cache-hit PP.

Fresh Release build and five CTests passed, including existing SSD roundtrip,
capacity refusal, and write-failure cases. Those tests do not exercise a full
NativeEngine multi-request lifecycle. End-to-end SSD reload quality, latency,
and physical-memory delta remain pending; the running service was not changed.
This implements our C++ engine policy, not an oMLX configuration change.

## Segmented attention reference (not promoted)

qwen38-block-attention-probe uses independent K/V blocks directly in separate
matrix multiplications. It concatenates only score blocks for a global softmax,
then sums probability x value contributions. It never reconstructs full K/V
inside the candidate attention. A separate dense baseline is kept for validation,
so this experiment cannot be used to claim a process-memory saving.

M5 Pro 64 GiB; synthetic FP32 single-head data, 16 queries, dimension 128,
65,536 key/value rows in 16 blocks of 4096. Unmasked attention, no QSA/GQA,
RoPE, model, MTP, or SSD serialization. Eight alternating-order paired rounds;
first pair discarded for warm ranges. No thermal control. Candidate maximum
absolute error 1.04308e-6, below the probe's 1e-5 check but NOT bit-exact.

Warm dense times 0.746–1.481 ms versus blocks 2.703–5.401 ms. This naive
multi-dispatch strategy is rejected for serving because it is slower. It does
not establish production BF16 quality or long-context retrieval correctness.
A direct block-addressing Metal kernel preserving QSA selection and reduction
semantics remains unimplemented; no segmented KV was enabled in the server.

Probe safety: strict allocator limit 1 GiB/cache 16 MiB; external start 16 GiB,
available floor 12 GiB, process RSS/footprint caps 4 GiB. Sampled footprint 0.2 GiB.
Raw output: block-attention-reference.txt.
