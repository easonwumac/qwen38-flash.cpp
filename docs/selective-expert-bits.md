# BF16-sourced selective Q2/Q3 routed-expert feasibility

2026-09-08: completed a bounded projection-selective screen, not a full-model
conversion or calibrated per-expert recipe. Production weights, runtime
defaults, and serving processes were not changed. No candidate weights were
written to disk. Full-model256K/36GiB feasibility remains unproven.

## Method and safety

`devtools/selective_expert_bits_probe.py` reads only16 distributed expert slices
(IDs0,18,...,270) from each retained REAP288 BF16 layer shard, checking geometry,
dtype, and byte lengths. It hashes the actual sampled BF16 bytes. Gate/up are
split from the source `[288,1280,2560]` gate_up tensor; down is `[288,2560,640]`.
All candidates, including the comparison Q4, are freshly affine-quantized from
BF16 with group64. No Q4-to-Q2/Q3 double quantization is involved.

M5 Pro64GiB, Python MLX0.32.2, layers0/24/47, rows1/5/512, seeds3801/3802/3803:
27 separate sequential guarded processes, each with all five recipes and31
samples per recipe after three warmup sweeps. Each timing sweep rotates and
reverses recipe order. Raw samples, input seeds and sampled-weight hashes are
in `selective-expert-bits-results.json`. An earlier seed3801 layer0 S1 smoke
was exploratory and is not included in that dataset.

Inputs are synthetic normal BF16 values, with fixed cyclic top10 routes and
nonuniform synthetic routing weights. S1 touches10 of the16 loaded experts;
larger S covers all16. This is NOT a natural-prompt calibration or quality
corpus, and is not a complete decoder layer: it omits router selection,
shared expert, residual/HC, attention, and downstream layers. There is no
sampling, real context, cache, or MTP. S5 only models a verifier-sized work
shape; S512 only models a PP-sized work shape.

Every recipe uses sorted generic MLX gather_qmm and the same BF16 elementwise
and reduction path. **The control is not our production fused/lossless16 Q4
kernel or its exact weight pack.** A16-expert working set also does not model
the cache behavior of the full288-expert bank. Python dispatch/evaluation is
included. No thermal pinning or concurrent-application activity controls; no
model/GPU benchmark was launched concurrently by this task.

External guard: startup16GiB, minimum available12GiB, RSS/physical footprint
caps4GiB. Requested MLX allocation limit2GiB, cache64MiB; ordinary MLX limit
behavior is not treated as a hard OOM guarantee. External guard is retained.
The first three processes used0.25s polling, the rest0.1s. All27 exited0;
sampled maximum footprint0.5GiB, RSS0.2GiB, minimum available40.1GiB. Two short
processes reported0.0GiB because they completed between samples, not zero use.
Maximum recorded allocator peak was approximately0.60GiB. All formats plus
BF16 reference coexist, so process footprint is NOT a measured candidate-only
RAM saving.

## Results

Speed columns are medians of nine paired control/candidate timing ratios
(three layers times three seeds); greater than1 is faster. They are not
end-to-end token/s or a confidence interval.

| Gate/up/down bits | Packed MiB,16 experts incl. metadata | Reduction vs fresh Q4 | S1 ratio | S5 ratio | S512 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
|4/4/4|42.1875|baseline|1.000|1.000|1.000|
|2/4/4|35.9375|14.81%|1.040|0.993|1.005|
|4/2/4|35.9375|14.81%|1.030|0.987|1.005|
|2/2/4|29.6875|29.63%|1.106|1.004|1.013|
|3/3/4|35.9375|14.81%|1.021|0.947|0.996|

For2/2/4 the nine S1 ratios span1.027–1.150, S5 spans0.978–1.024,
S512 spans0.993–1.017. Thus there is a modest S1 generic-kernel signal,
but no material PP/verifier speed gain here. Q3 S5 regresses consistently.
These ratios cannot establish a win against production Q4.

Routed-output relative L2 error versus BF16 reference, range across all27
inputs (not accuracy loss, not perplexity, and not a whole-layer error):

| Recipe | Relative L2 range |
| --- | ---: |
|4/4/4|0.131–0.172|
|2/4/4|0.345–0.468|
|4/2/4|0.314–0.448|
|2/2/4|0.466–0.609|
|3/3/4|0.232–0.301|

Plain affine Q2 substantially increases error even when only one projection
is changed. The late-layer synthetic results do not justify automatically
quantizing the tail more aggressively. No claim that real task accuracy falls
by these percentages is valid.

## Memory interpretation and decision

Changing one whole gate/up projection fromQ4 toQ2 saves6.25MiB per16 experts;
both save12.5MiB. If applied to all288 experts in all48 equal-geometry layers,
the logical packed-bank differences are5.273GiB and10.547GiB respectively.
This is arithmetic extrapolation, not a converted artifact or observed RSS
reduction. Selective coverage scales that payload difference; preservation of
sensitive layers, compact production metadata, residency and temporary copies
must be accounted for separately. It is not a256K/36GiB demonstration.

Do not promote broad uncalibrated Q2 or convert the full model based on this
screen. A useful next gate is calibration using real activation distributions,
with sensitive experts/projections left higher precision and held-out mixed
tasks (including code, structured output and long-context retrieval). Any
resulting mixed-format path must also beat or match the actual optimized Q4
kernel, including routing/partition dispatch costs. This screen does not reject
all calibrated Q2 schemes; it rejects treating plain affine Q2 as an already
validated quality/speed win.

Six focused input-validation tests, Python byte compilation, a clean Release
rebuild, and all five existing CTest suites passed (the retained model was
provided for tokenizer tests). Existing macOS26.0/26.2 library deployment-target
linker warnings remain. `git diff --check` passed.
