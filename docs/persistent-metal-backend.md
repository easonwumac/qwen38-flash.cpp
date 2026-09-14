# Persistent Metal backend

The Niwaki Q3 fusion is retained as a backend primitive, not as an MLX runtime
toggle. The first direct-Metal gate uses the real layer-3 routed weights through
a read-only mmap-backed `MTLBuffer`, persistent pipelines and scratch buffers,
and one command buffer for Q3 gate/up plus down/reduce.

On Apple M5 Pro 64 GiB, Niwaki 99B Q3/group-64, ten fixed experts, BF16
activations, 31 interleaved samples and a 64 MiB GPU cache eviction before each
sample:

- joined GPU median: 0.106--0.145 ms;
- joined submit-to-completion median: 0.271--0.347 ms;
- split-command-buffer median: 0.433--0.494 ms;
- warm generic MLX routed oracle: 0.351--0.740 ms;
- direct/MLX cosine: 0.999812; RMSE: 4.52e-5; max absolute error: 1.79e-4.

This establishes two separate effects: fixed-geometry Q3 kernels reduce device
work, while keeping dependent kernels in one command buffer removes about
0.15 ms of host/submit latency. It does not yet establish a whole-model token/s
gain because shared MoE, attention, GDN, hyper-connections, healing maps and the
LM head still execute in MLX.

The second gate added the Q4/group-32 shared gate/up/down, Q8/group-64 shared
router, and routed/shared merge to the same command buffer. Five independent
process runs, each with 31 cache-evicted samples, measured 0.293--0.310 ms
submit-to-completion and 0.136--0.142 ms GPU time. The warm full MLX oracle
ranged from 0.372 to 0.788 ms. Full-output cosine was 0.999776, RMSE 6.86e-5,
and max absolute error 2.75e-4. This clears the primitive performance gate but
remains an approximate path until token-trajectory testing is possible.

Two follow-up changes tightened this primitive. BF16 route weights with
per-expert BF16 accumulation changed full-output cosine from 0.999776 to
0.999777 and RMSE from 6.86e-5 to 6.85e-5 without a measurable speed cost.
Packing four Q3 output rows into each SIMD did not change the output or improve
quality and was slower, so it was removed. In contrast, fusing the three
gate/up/router dispatches into one and routed/shared down/merge into another
preserved the exact candidate hash while reducing the cache-evicted GPU median
from 0.138--0.141 ms to 0.125--0.131 ms across five process runs. Submit wall
time remained host-noise limited at 0.279--0.305 ms versus 0.291--0.302 ms for
the five-dispatch control.

The third gate moved the Q8/group-64 512-way router and top-10 selection into
the same command buffer. Across five process runs its cache-evicted GPU median
was 0.172--0.175 ms and submit wall median was 0.330--0.350 ms for router,
selection, and the complete fused MoE. All ten expert IDs and all BF16 route
weights matched the MLX router exactly. This removes the CPU routing barrier;
the remaining full-output difference is in the quantized projection reduction
order, not expert selection.

The fourth gate now covers the complete MLP half-layer in one command buffer:
MLP HyperConnection normalization/read, the device router, routed and shared
MoE, the Niwaki rank-64 healing correction, and HyperConnection write. Across
five independent processes, each with 31 samples and a 64 MiB device-cache
eviction before every sample, GPU medians were 0.232--0.241 ms and
submit-to-completion medians were 0.397--0.419 ms. The complete MLX oracle
measured 1.335--1.888 ms under the same synthetic one-token harness. The direct
path matched all ten router IDs and BF16 route weights exactly; complete
half-layer cosine rounded to 1.000000, with 2.44e-4 RMSE and 3.91e-3 maximum
absolute error. These are primitive results on Apple M5 Pro 64 GiB with real
layer-3 Niwaki 99B Q3/group-64 routed, Q4/group-32 shared and HyperConnection,
Q8/group-64 router, BF16 rank-64 maps/activations, deterministic input, no
context, sampling, or MTP. They do not yet imply whole-model token throughput.

Matching the generic Q4 projection's packed-slice reduction order did not
reduce the HyperConnection error and raised its GPU median from the observed
0.033--0.098 ms optimized range to 0.087 ms in the tested run. The optimized
contiguous-eight affine dot path is retained; future parity work must isolate
normalization and BF16 expression rounding rather than revisiting reduction
partitioning.

The first 128K full-attention primitives are also validated. A parallel exact
QSA selector scores four 128-wide query heads against 32,768 pooled blocks,
then repeatedly bitonic-sorts 256 candidates and retains the upper 128. Across
five independent 31-sample processes, the fused selector plus block-direct
affine-Q8 attention measured 0.286--0.309 ms GPU and 0.455--0.485 ms wall. Its
top-128 set matched the CPU reference exactly; the first Q8 attention head had
0.999999 cosine and 4.56e-4 maximum absolute error against a CPU online-softmax
reference. The discarded single-thread 128-element heap selected the same set
but required 2.78 ms GPU because its private arrays and serial work dominated.

Real layer-3 Q4/group-32 indexer-QK, query/gate, key and value projections took
0.138--0.141 ms GPU; normalization plus partial RoPE took 0.0060--0.0065 ms;
gate plus the real Q4 output projection took 0.0645--0.0685 ms. The projection
primitive had 0.999972 cosine, 2.72e-3 RMSE and 1.56e-2 maximum absolute error
against MLX. Adding the independently measured stages and HyperConnection read
places the current full-attention half-layer device frontier at approximately
0.52--0.55 ms, before a final single-command-buffer measurement. The QSA test
uses deterministic synthetic pooled keys and a fully cold 131,072-token affine
Q8 K/V cache; it does not yet include the current-token BF16 tail, state update,
or an end-to-end retrieval claim.

The fifth gate connects both HyperConnection halves, real attention and MoE
weights, exact device routing, the rank-64 healing map, and the current token's
BF16 K/V tail in one command buffer. On five independent processes, the
complete layer-3 GPU median was 0.690--0.850 ms and wall median was
0.884--1.041 ms, with the same final BF16 output hash in every process. The
attention half alone was 0.505--0.669 ms GPU and 0.698--0.856 ms wall. For
directional context, the existing guarded MLX synthetic 128K layer probe had
per-process medians of 1.231--2.284 ms in five adjacent runs. This is not yet a
production speedup claim: the direct path uses different deterministic content,
does not persist the appended hot K/V or QSA raw/pooled state, and still needs a
token-trajectory and needle gates.

A subsequent complete-layer oracle used the identical 10,240-wide BF16 input,
arbitrary but identical 131,072-token affine-Q8 K/V bytes/scales/biases, pooled
QSA bank, current BF16 K/V tail, and real layer-3 Niwaki weights on both paths.
The persistent output matched MLX at 0.999990 cosine, 1.38e-3 RMSE and 7.81e-3
maximum absolute error. The oracle's first lazy/compile evaluation took 20.6 ms
and is reported only for parity, not as a performance control. The remaining
correctness risk is recursive accumulation across tokens and layers, not a
large one-layer numerical mismatch.

The sixth gate makes the attention state persistent. A 2,048-token fixed hot
BF16 K/V slab and four-row QSA pending buffer are updated in place; every fourth
decode row is mean-pooled, normalized, RoPE-encoded and appended directly to
the fixed pooled-key bank. The append kernel measured 0.0042--0.0048 ms GPU and
was bit-exact against the host reference for hot K/V, pending raw keys and the
first newly completed pooled block. The selector now accepts arbitrary block
counts through 256-row padding and an odd-group carry in its merge tree, so
crossing from 32,768 to 32,769 blocks does not double work to the next power of
two. Both top-128 sets matched the CPU reference exactly.

A four-token recursive trajectory then fed each direct layer output back as the
next input while both direct and MLX states accumulated hot K/V and QSA rows.
It covers tail sizes one through three and the fourth-token transition where a
new pooled block becomes selectable. Against MLX, the worst layer-output cosine
was 0.999954, maximum RMSE was 3.78e-3 and maximum absolute error was 1.76e-2.
This test used Apple M5 Pro 64 GiB, real Niwaki 99B layer-3 Q3/group-64 routed,
Q4/group-32 shared/attention and BF16 healing/HyperConnection weights, a
deterministic recursive stream, a synthetic 131,072-token affine-Q8 cold KV
history, a 512-token QSA budget, greedy/no-sampling execution and no MTP. It is
a layer-backend correctness gate, not whole-model throughput or retrieval data.

The seventh gate covers the complete stateful GDN block used by the 36 linear
attention layers. Four real Q4/group-32 input projections, width-four causal
convolution and state shift, normalized Q/K, sigmoid beta, exponential decay,
the in-place BF16 `[48,128,128]` recurrence, sigmoid-gated RMSNorm and the real
Q4/group-32 output projection execute in one command buffer. Across five
independent processes with 31 cache-evicted samples each, GPU medians were
0.232--0.276 ms. First-token output matched MLX at
0.999790 cosine, 2.45e-4 RMSE and 1.10e-3 maximum absolute error; recurrent
state max error was 3.97e-4. A four-token recursive trajectory bounded output
error at 2.45e-4 RMSE/1.10e-3 max and final recurrent-state error at 5.49e-4,
so the state error did not grow materially in this gate.

This work also exposed a required loader rule: safetensors tensor payloads are
not guaranteed to begin at naturally aligned addresses. Several Niwaki layer-0
BF16 scale/bias tensors begin at odd byte offsets. Binding those bytes directly
as `bfloat*` produced non-finite results even though packed U32 weights read
correctly. The retained kernel reads such BF16 parameters bytewise and
reconstructs their 16-bit value; persistent runtime imports must use the same
unaligned-safe contract or copy into aligned storage. Measurements used Apple
M5 Pro 64 GiB, real Niwaki 99B layer-0 weights, deterministic recursive BF16
inputs, zero initial GDN state, greedy/no-sampling execution and no MTP. This is
still a block-level result without HyperConnection, MoE or whole-model timing.

The eighth gate joins that GDN block to both real layer-0 HyperConnection
halves, the device Q8 router/top-10 path, Q3/group-64 routed experts,
Q4/group-32 shared expert and the BF16 rank-64 healing map. Across five
independent processes with 31 cache-evicted samples each, the complete
single-command-buffer layer measured 0.454--0.464 ms GPU. Its output matched a
complete MLX layer-0 oracle at 0.999998 cosine, 6.13e-4 RMSE and 3.91e-3 maximum
absolute error. A four-token recursive complete-layer trajectory retained
0.999985 minimum cosine, 1.62e-3 maximum RMSE and 7.81e-3 maximum absolute
error. The same unaligned-safe BF16 loads were extended to model projection,
HyperConnection and healing parameters; aligned scratch/state buffers remain
native `bfloat*`.

For scale only, combining the five-process upper medians for 36 routed GDN
layers and 12 full-attention layers gives about 25.2 ms of device layer time,
or 39.6 token/s before embedding, final head and host overhead. This is not an
end-to-end throughput claim, but it is deliberately conservative because 24
real layers are shared-only and should avoid routed expert work. A dedicated
shared-only layer gate and runtime integration are required before claiming the
40 token/s target.

The ninth gate replaces part of that estimate with a real shared-only GDN
layer. The layer-10 command contains both HyperConnection halves, its complete
persistent GDN state update, Q4/group-32 shared gate/up/down projections,
Q8/group-64 shared router, rank-64 healing and final stream write; no absent
routed bank is touched. Across five independent 31-sample cache-evicted
processes, GPU medians were 0.334--0.338 ms. Complete output versus the MLX
layer-10 oracle measured 0.999997 cosine, 7.26e-4 RMSE and 3.91e-3 maximum
absolute error. Conditions match the preceding gates: Apple M5 Pro 64 GiB,
actual Niwaki 99B weights, deterministic BF16 stream, zero initial GDN state,
greedy/no-sampling and no MTP. A shared-only full-attention layer remains to be
measured directly.

Using the measured routed/shared difference as a temporary estimate for the six
shared-only full-attention layers, the checkpoint's actual 24 routed and 24
shared-only split projects about 21.8 ms of layer-device work, approximately
45.9 token/s before embedding, final head and host overhead. This is useful
headroom evidence, not an end-to-end result.

The tenth gate directly measures the remaining combination: shared-only
full-attention layer 11. It uses the same 131,072-token affine-Q8 cold KV,
512-token exact QSA selection and current BF16 tail as the routed attention
gate, with real layer-11 attention/HyperConnection weights and its shared-only
MoE/healing path. Five independent 31-sample cache-evicted processes measured
0.586--0.588 ms GPU. Complete output matched the MLX layer-11 oracle at
0.999990 cosine, 1.44e-3 RMSE and 1.17e-2 maximum absolute error.

All four deployed layer combinations are now represented by actual weights.
Using conservative five-process upper medians--0.464 ms routed GDN, 0.729 ms
routed full attention, 0.337 ms shared-only GDN and 0.588 ms shared-only full
attention--the real 18/6/18/6 layer split totals about 22.3 ms of device layer
work, or 44.8 token/s before embedding, final head and host overhead. The
40-token/s no-MTP target is therefore no longer blocked by the layer-kernel
compute bound, but only runtime integration and non-layer overhead can establish
the end-to-end result.

The next acceptance gates are:

1. integrate the persistent state and layer dispatch into the runtime;
2. require an adjacent 16K needle improvement, then repeat at 65K and 128K;
3. validate long-run Q8 hot-slab flushes before the 40 GiB memory gate;
4. move MTP verification onto the persistent backend and measure 60 token/s.

Run the bounded primitive probe with:

```sh
./build-v1-all/qwen38-persistent-metal-moe-probe MODEL_DIRECTORY
```
