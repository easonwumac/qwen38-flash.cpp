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

## Performance implication

A full-vocabulary head at about 21.5 ms already consumes most of a 45 tok/s
budget. The complete target cannot meet that gate if every greedy token pays this
exact path unchanged. Exact argmax pruning/candidate certification, head layout,
and overlap therefore form an explicit optimization track; sampled requests must
retain the full distribution path.
