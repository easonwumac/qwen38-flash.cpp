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

The next acceptance gates are:

1. combine the full-attention primitives, current-token tail/state update and
   both HyperConnections into one fixed-buffer layer command;
2. match the retained MLX token trajectory before extending beyond one layer;
3. require an adjacent 16K needle improvement, then repeat at 65K and 128K.

Run the bounded primitive probe with:

```sh
./build-v1-all/qwen38-persistent-metal-moe-probe MODEL_DIRECTORY
```
