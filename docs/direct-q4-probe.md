# Direct affine-Q4 MPP developer probe

This developer-only probe evaluates an INT8 MPP matmul that reads the retained
affine-Q4 weights directly. It creates no persistent W8 sidecar and does not
change any runtime profile or production default.

## Build and run

Configure with MLX enabled as usual, then build only the probe:

```sh
cmake -S . -B build-all
cmake --build build-all --target qwen38-direct-q4-probe --parallel 2
```

Run it through the resource guard, substituting a local shard and tensor base:

```sh
python3 devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 --interval .1 -- \
  build-all/qwen38-direct-q4-probe \
  --shard /path/to/model-00001-of-00131.safetensors \
  --tensor language_model.model.layers.0.linear_attn.in_proj_z \
  --warmups 3 --iterations 15
```

Standard output is one JSON object containing exactness gates, semantic gates,
quality metrics, all timing samples, and scratch size. Guard diagnostics go to
standard error.

## Layout and arithmetic

The retained weight is `uint32[N,K/8]`. Each word contains eight unsigned Q4
codes in low-to-high nibble order; the kernel does not use Cider's W4 packing
or subtract 8. Activation quantization is per token and group 64 and emits
INT8 values, an FP32 scale, and the FP32 sum of the original BF16 input group.
For every group the kernel computes:

```text
act_scale * weight_scale * dot_i32(aq, unsigned_q4_code)
  + weight_bias * sum(original_input)
```

The MPP INT32 accumulator is reset and flushed to FP32 after each 64-value
group. Its worst unsigned-Q4 group dot magnitude is `127 * 15 * 64 = 121920`,
well inside INT32. For the `[512,2560]` input, temporary storage is 1.25 MiB for INT8
activations plus 0.15625 MiB for scales and sums: 1.40625 MiB total.

The MPP structure is adapted from Cider
`cider/kernels/pergroup_int8_gemm.metal` at commit
`4d91fcee9439f7aea17ae6e965271d9536c604a0`, under the MIT license.

## 2026-09-06 result

The measured tuple was the real layer-0 `linear_attn.in_proj_z` affine-Q4
weight `[6144,2560]`. Inputs were deterministic synthetic normal BF16 values
(seed 38), not natural-prompt activations. The machine was an Apple M5 Pro
with 64 GiB RAM.

- M32 and M128 unsigned-nibble unpack plus INT8 MPP matched the INT32 reference
  exactly (`max_abs=0`).
- Zero activations produced exact zeros. All-zero Q4 codes with signed nonzero
  affine biases matched `bias * sum(original_input)` exactly in the small BF16
  semantic gate.
- M512 output versus native MLX affine Q4 was finite, cosine `0.999829`, maximum
  absolute error `0.073120`, and RMSE `0.005664`.
- Direct MPP including activation quantization and BF16 output conversion took
  `0.977125 ms` mean and `0.974625 ms` median across 15 alternating samples.
  Native affine Q4 took `0.766686 ms` mean and `0.768166 ms` median. Direct MPP
  therefore reached only `0.784635x` mean and `0.788166x` median speed.
- Resource guard observed 0.1 GiB peak footprint, 0.1 GiB peak RSS, and 33.2 GiB
  minimum available memory. Persistent W8 storage was zero.

This result is a negative performance result, so the direct kernel is not
integrated into full-model execution. It is a projection-only microbenchmark;
it does not measure GDN state or end-to-end prefill. It also is not bit-exact
to native affine Q4: native execution may round a BF16 dequantized value before
multiplication, while this kernel applies the algebraic affine epilogue in
FP32 and converts the final output to BF16.
