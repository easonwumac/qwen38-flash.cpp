# Eight-layer PP peak-memory gate

2026-09-06. Status: bounded research gate passed for route fusion, full-model
promotion pending. Defaults and running services unchanged. Peak reduction is
prioritized before resident-weight reduction; no lower precision or expert paging.

## Setup

`qwen38-pp-reduce-group-probe MODEL 0|1 [barrier]` runs actual layers0–7 with
sequential hidden states, including full attention and PLE. Four successive
512-token chunks,2048 tokens total; deterministic diverse token IDs from the
retained REAP-288 affine Q4/group64 pack. Lossless16 metadata, top10, deferred
temporary metadata, no decoded-metadata request cache, speed kernels. No MTP
drafter or language head: these are partial-model PP tests, not full token rates.

M5 Pro64GiB, strict private pinned MLX runtime,64MiB allocator cache,10GiB MLX
allocation ceiling,12GiB process/RSS guard ceiling,26GiB startup availability,
12GiB ongoing availability floor. Single GPU test at a time. No thermal pinning
or filesystem cache flushing. Eight iterations per process; first two excluded,
median of remaining six. Production-equivalent barrier8; speculative state is
materialized at each iteration end. Full output plus all materialized
KV/QSA/GDN/PLE tensor values are hashed after timing; this extra CPU conversion
can affect the process-level guard peak, but not the reported MLX per-iteration
allocation peak. Reported physical bytes are end-of-iteration samples, not peaks.

## Results

Initial A/B/B/A process order:

| Variant | Warm median ms | Median MLX peak bytes | Final-state hash |
| --- | ---: | ---: | --- |
|Stock A1|517.961|7,590,926,234|14440003744645817219|
|Route fusion B1|501.633|7,553,179,934|same|
|Route fusion B2|496.859|7,553,203,486|same|
|Stock A2|516.378|7,590,793,626|same|

About36MiB lower allocation peak and3–4% faster partial-model PP. End-of-iteration
physical footprint was about20–24MiB lower in these runs; sampled guard peaks
rounded to7.2GiB stock and7.1GiB fused. These metrics must not be conflated.
No48-layer extrapolation or residency claim is justified.

Barrier sweep, route fusion on (single process per stride):

| Barrier | Warm ms | MLX peak bytes | Decision |
| --- | ---: | ---: | --- |
|8|502.081|7,553,203,998|Retain|
|4|506.461|7,550,681,374|Tiny memory gain, no promotion|
|2|522.513|7,494,587,654|Slower, reject|
|1|552.373|7,241,680,344|About297MiB below stride8 but slower, reject|

All output/state hashes matched. Earlier synchronization is not an acceptable
way to save memory under the user's PP non-regression requirement.

## Additional SwiGLU candidate

`QWEN38_PP_SWIGLU=1` fuses sigmoid and two products in grouped PP only. It follows
MLX's typed Sigmoid expression and BF16 intermediate boundaries. An initial
float-typed formula failed the exact unit oracle and was corrected, not accepted
as approximately equivalent. Synthetic full-array tests cover varied magnitudes,
signs and row widths1/16/128. This flag remains off in product profiles.

Combined route+SwiGLU runs:497.748/497.767ms, MLX peaks7,552,908,062 and
7,552,924,702 bytes, same complete-group hash. A route-only recheck was507.703ms,
7,553,188,638 bytes; stock recheck516.168ms,7,590,793,626 bytes. Against previous
route-only runs the timing overlaps, so an independent SwiGLU speedup is not
established. Peak improvement is only~0.3MiB: eliminating an intermediate array
does not imply the peak allocation phase becomes materially smaller.

## Remaining gates

Fresh Release build and all5 CTests passed, in addition to the guarded multi-layer
checks. Full resident-engine cold/warm PP, long-context cache extension, decode,
MTP and output-quality validation still block product promotion. These probes
do not satisfy the complete peak-memory goal, nor demonstrate resident30–36GiB.
No admission or runtime safety limits were lowered. Raw output, including cold
runs and negative barrier results, is retained in the adjacent JSON.
The final non-mutating full-model admission check refused41.1GiB available
against44GiB required; no full model was loaded in this milestone.
