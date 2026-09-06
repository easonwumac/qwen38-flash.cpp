# 8K full-Q8 SSD prefix reuse

Completed one cold/reuse request pair on the current SSD-idle-release engine.
Actual prompt length 8,221 tokens, generated 16 tokens each time. Second request
reused 8,220 tokens via SSD and produced identical output tokens and identical
MTP acceptance (10/20, five rounds). No full-model 32K run: minimum available
memory of 8.2 GiB was too close to the 8 GiB stop line.

| Metric | First request | SSD reuse |
| --- | ---: | ---: |
| First nonempty-stream callback latency | 18,843.6 ms | 112.02 ms |
| Whole complete_stream call | 20,372.8 ms | 502.033 ms |
| Reported prompt_ms | 18,473.2 ms | 0.569375 ms |
| Reported generation_ms | 1,852.28 ms | 450.668 ms |
| Idle physical footprint bytes | 41,314,498,416 | 41,330,309,096 |
| Idle MLX active bytes | 40,548,682,674 | 40,548,682,674 |

Prompt_ms alone is misleading for lazy SSD loads; first callback and whole-call
wall time are measured externally by the probe. Callback timing is not exact
GPU first-token timing and depends on output buffering. OS file-cache warmth
is uncontrolled, so this does not establish cold-NVMe reload performance.
The first prompt rate is about 445 tokens/s including this path's overhead,
not evidence of 600 PP. No 32K/65K/128K claims.

Hardware/model: M5 Pro 64 GiB, full-Q8 MTP REAP-288-L47 capsule, unchanged
affine Q4/group64 target, lossless16 qmeta, resident 12:29, speed profile,
depth4 with existing fallback, prefill512, decoded-qmeta cache disabled,
thinking off. Synthetic numbered weather records followed by a short code
request; this is output replay parity, not a retrieval/quality benchmark.

Guard startup40 / available8 / footprint and RSS42 GiB, strict allocator40 GiB,
allocator cache256 MiB. Observed peak footprint39.2 GiB, RSS31.8 GiB. New
experiment-only SSD cache capped4 GiB, about596 MiB retained for inspection at
experiments/ssd-long-check.mEhuMI. No serving configuration/model files changed.
No RAM-only control was run: idle memory savings are not quantified by this pair.

## Save/load audit

Our decode-state serializer passes array references to MlxSafetensors::save;
it does not explicitly copy all tensor data into a C++ staging buffer. Pinned
MLX io/safetensors.cpp, however, converts every array with contiguous(), collects
all results, evaluates them together, then writes arrays. Noncontiguous KV
slices may therefore require simultaneously live contiguous copies. Existing
contiguous arrays may be shared, so this is not proof all data is duplicated.
Next bounded experiment: compare collective versus per-array materialization
while preserving safetensors metadata/layout and failure-safe persistence.
No serializer change was made in this milestone.

Fresh Release build and five CTests pass. Added bounded 8192/32768 probe modes,
external first-callback/wall timing and idle-memory accounting. A final explicit
long-mode mismatch/cache-miss guard was added after the measured run; the recorded
output independently satisfies its checks. Raw output: ssd-8k-reuse-output.txt.
