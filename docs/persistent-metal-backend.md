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

The next acceptance gates are:

1. match the retained MLX token trajectory before extending beyond one layer;
2. encode one complete full-attention layer with fixed buffers and no MLX
   synchronization inside the layer;
3. require an adjacent 16K needle improvement, then repeat at 65K and 128K.

Run the bounded primitive probe with:

```sh
./build-v1-all/qwen38-persistent-metal-moe-probe MODEL_DIRECTORY
```
