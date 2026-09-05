# W8A8 MPP projection probe

`qwen38-w8a8-gdn-probe` is a developer microbenchmark for a native Metal 4
MPP INT8 matrix multiplication. It includes per-row activation quantization,
fused activation/weight scale application, and BF16 output conversion. The W8
sidecar is derived from the dequantized affine-Q4 source weight used by the MLX
reference path.

The kernel is adapted from
[Mininglamp-AI/cider at `4d91fcee`](https://github.com/Mininglamp-AI/cider/tree/4d91fcee9439f7aea17ae6e965271d9536c604a0),
whose relevant W8A8 Metal implementation is MIT licensed. The retained source
header includes the MIT copyright and license text.

## Build

Configure with an MLX C installation, then build only the probe. Paths below
are placeholders so the command is portable:

```sh
cmake -S . -B build-all \
  -DQWEN38_MLXC_INCLUDE_DIR=/path/to/mlx-c/include \
  -DQWEN38_MLX_LIBRARY_DIR=/path/to/mlx/lib
cmake --build build-all --target qwen38-w8a8-gdn-probe --parallel 2
```

## Guarded runs

The synthetic case constructs an affine-Q4 weight and uses deterministic
synthetic normal BF16 activations:

```sh
python3 devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 --interval .1 -- \
  build-all/qwen38-w8a8-gdn-probe \
  --rows 512 --k 2560 --n 6144 --warmups 3 --iterations 10
```

For a real affine-Q4 tuple, replace the shard and tensor placeholders. The
activations remain synthetic and are labeled as such in JSON output:

```sh
python3 devtools/memory_guard.py \
  --min-start-gib 16 --min-available-gib 12 \
  --max-rss-gib 4 --max-footprint-gib 4 --interval .1 -- \
  build-all/qwen38-w8a8-gdn-probe \
  --rows 512 --k 2560 --n 6144 --warmups 3 --iterations 10 \
  --shard /path/to/model-NNNNN-of-NNNNN.safetensors \
  --tensor language_model.model.layers.0.linear_attn.in_proj_z
```

## Current evidence and limits

On an Apple M5 Pro with 64 GiB unified memory, the real layer-0
`linear_attn.in_proj_z` weight measured a 1.36x mean and 1.44x median speed
ratio over the current affine-Q4 BF16 reference at M512. MPP output was finite,
with cosine 0.999878, maximum absolute error 0.086914, and RMSE 0.004777.
Full-geometry integer checks were exact at M32 and M128. The per-layer W8
sidecar for a 6144-by-2560 projection is 15.023 MiB.

This is a single-projection microbenchmark with synthetic activations. The W8
sidecar is not wired into model loading or default execution, and the result
does not establish full-model prefill throughput, long-context performance, or
the 600 tokens/s product target. Metal 4 MPP support is required at runtime.
