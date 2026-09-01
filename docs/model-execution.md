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

## Performance implication

A full-vocabulary head at about 21.5 ms already consumes most of a 45 tok/s
budget. The complete target cannot meet that gate if every greedy token pays this
exact path unchanged. Exact argmax pruning/candidate certification, head layout,
and overlap therefore form an explicit optimization track; sampled requests must
retain the full distribution path.
