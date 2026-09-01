# Model execution bring-up

## First real checkpoint operation

`qwen38-qmm-smoke` executes the retained checkpoint's full Q4 affine `lm_head`:

- packed weight: U32 `[248320, 320]`;
- scales and affine biases: BF16 `[248320, 40]`;
- activation: BF16 `[1, 2560]`;
- quantization: affine 4-bit, group size 64, transposed projection;
- output: 248,320 finite logits.

Five independent warm-page runs produced the same argmax (609) and checksum
(437.524). Evaluation latency was 20.93-22.47 ms, median 21.47 ms. One measured
process peaked near 432 MB footprint while touching the roughly 357 MB head
shard. These are component diagnostics, not decode throughput.

## Native token embedding

The second real operation gathers packed Q4 rows for the fixed tokenizer parity
IDs `[9419, 11, 1814, 0]`, gathers the matching BF16 scales/biases, and
dequantizes to BF16 `[4, 2560]`. The result is finite and deterministic with
checksum 1.89569 and L2 norm 0.500975. A representative process measured 23.20 ms
for the first lazy load and 0.248 ms median for five subsequent in-process
lookups. This separates one-time weight materialization from steady request cost.

## Loader boundary discovered

Passing raw safetensors BF16 pages to the staged MLX C managed-buffer constructor
did not preserve values: the first expected scale `-0.00631714` became roughly
`-6.3e9`, and the projection emitted non-finite logits. This unsafe route was
removed. The correctness backend uses MLX's lazy safetensors `Load` primitives,
which read the same scale correctly and produce finite deterministic logits.

The independent manifest/mmap parser remains the source of validation, indexing,
storage policy, and future direct-Metal paging. A future direct backend must
explicitly define safetensors-to-kernel BF16 bit semantics and pass numerical
fixtures before replacing the safe loader.

## Qwen3.8 Hyper-Connection parity

The native graph now implements the architecture's four residual streams:

1. repeat each token embedding four times along the feature dimension;
2. grouped RMS-normalize each 2,560-wide stream and apply the checkpoint's
   offset norm (`1 + weight`);
3. produce the block input through the quantized 10,240 → 320 → 10,240
   SiLU/sigmoid mixer and mean across streams;
4. produce four write gates with `2 * sigmoid(project(normalized) / 4)`;
5. broadcast the block output through those gates and add it to the streams.

On fixed real token IDs `[9419, 11, 1814, 0]`, the C++ path and the independent
MLX-Python oracle agree after float32 materialization:

| Value | Checksum |
|---|---:|
| layer-0 mixed input | 383.095612 |
| layer-0 injection gates | 0.867310 |
| synthetic block write-back | 81.441620 |
| final mixer output | 1037.270386 |

The synthetic write-back deliberately feeds the mixed input as a stand-in block
output. It verifies Hyper-Connection semantics in isolation; the attention and
MoE block implementations are the next model-graph milestone.

## Sparse MoE decode path

The native one-token path now computes the BF16 router, float32 softmax, normalized
top-10 selection, three affine Q4 projections per selected expert, probability
weighted reduction, and the gated shared expert. For the fixed layer-0 fixture,
C++ and MLX-Python select experts
`[78,62,113,257,137,249,232,239,254,51]` and produce output checksum
`0.0970327854`.

The stable implementation currently materializes router scores on the host and
launches projections per selected expert. A six-run component measurement was
31.95 ms cold and 1.68 ms warm median. MLX's generic device-side routing plus
`gather_qmm` preserved the generated tokens but regressed a full-model second
step from the 1.3--1.6 second range to 2.30 seconds, so it was rejected.

An opt-in two-dispatch Q4 Metal route is available with
`QWEN38_FUSED_MOE=1`. Its structure is adapted from MTPLX's Apache-2.0
`moe_glu_decode` work, modified to consume the checkpoint's separate gate/up
packs. An initial guard incorrectly required 512 experts, so early 1.68x and
1.33x observations did not engage the kernel and were invalidated. With the
correct 288-expert guard and row-contiguous MLX contract, the layer-0 checksum
is `0.09534358978` versus stable `0.09703278542`; the full model retains the
`9419 -> 11 -> 271` trajectory and matches the independent oracle's reported
logits (`13.6875`, `14.875`). A fresh-server eight-token greedy HTTP run on a
64 GB M5 Pro, prompt `Hello`, empty cache, produced byte-identical text but
measured 0.980 tok/s versus the stable route's earlier 1.050 tok/s. The fused
route is therefore a research hook, not a speed recommendation.

Direct custom-kernel access also requires row-contiguous inputs. Disabling that
MLX contract caused a Metal watchdog timeout on later layers even after the
40 GB trunk had been read into the filesystem cache. Full-model experiments now
run through `devtools/memory_guard.py`; sequential whole-trunk prewarming is no
longer used because it can combine filesystem cache and MLX residency into
unsafe unified-memory pressure.

## Stateful Gated DeltaNet decode

The layer-0 linear-attention path covers stateful one-token decode: four Q4 input
projections, width-4 depthwise causal convolution and its three-row cache, SiLU,
Q/K normalization and grouped-query expansion, float32 forget-gate construction,
beta-gated delta recurrence, the BF16 `[1,48,128,128]` recurrent state,
sigmoid-gated RMSNorm, and Q4 output projection.

For isolated deterministic inputs derived from token IDs 9419 and 11, C++ and
the independent MLX-Python oracle agree on all retained checks:

| Value | Checksum |
|---|---:|
| first output | -2.499662638 |
| second output | 1.829259396 |
| recurrent state after step two | 70.005004883 |

A six-run measurement put the second stateful step near 1.23 ms warm median.
This composed path is the correctness baseline. It materializes many elementwise
nodes and traverses the recurrent state repeatedly, so the optimized engine will
replace its prework and recurrence with fused Metal kernels. Multi-token chunked
prefill remains required before the block is complete.

## Stateful full-attention decode

The layer-3 full-attention path now executes Q4 Q/K/V/O projections, Qwen offset
RMSNorm, 64-of-256 dimensional partial RoPE, two-key/value-head to 24-query-head
GQA expansion, growing BF16 KV caches, float32 softmax, and the query-derived
sigmoid output gate. Two deterministic decode inputs match the independent
oracle exactly:

| Value | Checksum |
|---|---:|
| first output | 5.597035408 |
| second output | 11.505207062 |

The second step measures around 0.75 ms warm median in isolation. At two tokens,
the QSA indexer selects the entire visible causal tail, so ordinary attention is
semantically exact. The compressed-key indexer and block-selection mask for
contexts above the 2,048-token budget remain outstanding; this limitation is
explicit rather than silently treating all long-context tokens as selected.

## PLE n-gram SSD table

The host runtime implements the official splitmix/primes n-gram hash, EOS segment
reset semantics, affine Q4 row dequantization, and two storage layouts. The
original 30 GB safetensors layout is the correctness fallback. When the matching
headerless `.aos` artifact is present, each row's packed weights, scales, and
biases are contiguous and fetched with one `pread`, avoiding three distant
regions and avoiding a 30 GB virtual mapping in the serving path.

For the retained model, 16-row AoS gathers matched the safetensors fallback
element-for-element. A measured process used about 5.5 MB peak footprint; the
first gather was 4.53 ms and the next was 1.66 ms. Correctness and residency are
therefore established, but serial SSD reads are not the final decode path.
The stateful PLE graph is now implemented as well: n-gram gathering, Q4 key and
value projections, grouped offset RMSNorm, signed-square-root gating, and the
dilated depthwise-convolution cache all execute natively from C++. An independent
MLX-Python oracle verifies the two-token path. The first eight sampled elements
are bit-identical; aggregate L1 differs by roughly 0.3% because BF16 graph fusion
can round later channels by one ULP. The second-token component path measures
about 1.3--1.7 ms warm. Persistent parallel reads and fusion remain optimization
work, but PLE is no longer missing from the correctness graph.

## Composed decoder layer

The native decoder layer now composes the retained execution order rather than
testing blocks only in isolation: optional PLE injection, attention
Hyper-Connection read/write, stateful GDN or full attention, then MoE
Hyper-Connection read/write. Real two-token smoke runs cover all three distinct
layer shapes in the checkpoint:

| Layer | Shape | Warm second-token latency |
|---:|---|---:|
| 0 | GDN + MoE | 2.1--3.0 ms |
| 1 | PLE + GDN + MoE | 2.4--2.7 ms |
| 3 | full attention + MoE | 2.0--3.0 ms |

These are isolated component numbers with hot filesystem pages, not end-to-end
throughput claims. They establish state continuity and graph composition before
the 48-layer model loop is optimized.

## Full 48-layer greedy decode

The complete native correctness graph now loads all 48 decoder layers, embeds a
token, carries every recurrent/KV/PLE state, applies the final Hyper-Connection
mixer and Q4 vocabulary head, and performs greedy argmax. Starting from token
9419, both the C++ graph and the retained MTPLX reference produce token sequence
`11, 271`. Their maximum logits differ by 0.375 and 0.125 respectively; a
per-layer trace shows small BF16 composition differences from layer 0 that
accumulate through the trunk, while the greedy decisions remain stable.

The first cold all-weight pass measured 6.64 s. With filesystem pages warm, the
second token improved from 3.74 s to 1.56 s across repeated process runs. Peak
footprint was about 41.3 GB. This is explicitly the unfused correctness graph,
not the target engine: it exposes the dispatch/synchronization baseline that the
compiled layer runs, fused selected-expert kernels, and head work must replace.

A materializing per-layer trace narrowed the floor further. First-token layers
cost roughly 65--115 ms. On token two, hot layers reach 6--11 ms while several
middle layers still cost 75--111 ms. Sequentially reading every trunk shard into
the filesystem cache did not improve the result, ruling out ordinary cold file
pages as the dominant cause. A generic 10-expert batched QMM was element-exact
for gate, up, hidden, and down projections, but regressed complete token latency
from 1.33--1.61 s to 2.35 s. Production therefore keeps the exact loop while a
direct selected-expert Metal path is developed.

The guarded eight-step benchmark makes the page-temperature effect explicit.
For greedy continuation from token 9419 it measured
`5043.7, 2200.3, 247.1, 75.7, 72.8, 77.7, 75.8, 70.4 ms`. Thus the genuinely
warm stable path is currently about 13--14 tok/s, while short HTTP requests are
dominated by the first two expert-page population steps. Peak RSS was 29.0 GiB
and the guard's minimum reclaimable memory was 10.6 GiB. A second immediate
16-step cold start hit the Metal watchdog while still above 19 GiB reclaimable;
it was not an OOM, but establishes that full-model runs must be serialized and
guarded rather than launched back-to-back.

The layer trace now points to expert residency rather than arithmetic as the
cold bottleneck: token-two layers 0--11 are mostly 6--11 ms, while layers 12--35
spend roughly 70--83 ms each. Once the relevant pages are populated, whole-token
latency falls to the 70 ms range. The next optimization track is therefore a
bounded resident expert tier, not unsafe whole-file cache prewarming.

The bounded tier is now implemented with page locks on original Q4 expert
tensors rather than duplicate buffers. `QWEN38_RESIDENT_EXPERT_RANGE=12:34`
pins at most 22 contiguous layers (the implementation rejects wider ranges).
On the 64 GB M5 Pro, the progressive guarded sweep was:

| Resident range | Token-two latency | Peak RSS |
|---|---:|---:|
| none | 2183.9 ms | 30.2 GiB |
| 12:16 | 2014.0 ms | 30.8 GiB |
| 12:24 | 935.9 ms | 32.5 GiB |
| 12:28 | 713.2 ms | 34.3 GiB |
| 12:32 | 660.6 ms | 35.0 GiB |
| 12:34 | 551.7 ms | 35.4 GiB |

All rows retained token sequence `11,271` and logits `13.3125,14.75`. The
eight-step 12:28 run improved cold-inclusive sustained throughput from 2.48 to
5.65 tok/s, but its last five steps remained 70--77 ms (about 13--14 tok/s).
Resident paging therefore solves cold/short-request latency within the desired
30--36 GiB envelope; warm decode still requires arithmetic/dispatch fusion.

## Performance implication

A full-vocabulary head at about 21.5 ms already consumes most of a 45 tok/s
budget. The complete target cannot meet that gate if every greedy token pays this
exact path unchanged. Exact argmax pruning/candidate certification, head layout,
and overlap therefore form an explicit optimization track; sampled requests must
retain the full distribution path.
