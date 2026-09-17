# Persistent VQ and ANE probe

Date: 2026-09-17

This probe tested two ways to reduce decode latency without changing target
top-k or pruning experts. It used an Apple M5 Pro with 64 GiB unified memory,
macOS 26.5, Xcode 26.5, and the
`Qwen3.8-Flash-Next-VQ-2.1bpw` checkpoint. No thermal controller was used.

## ANE language-head result

A synthetic Core ML `2560 -> 248320` FP16 linear layer was compressed to a
4-bit per-group palette and followed by argmax. `MLComputePlan` assigned the
linear operation to the Neural Engine and argmax to the CPU. With
`CPU_AND_NE`, 10 warmups and 60 measured predictions gave 4.318 ms median,
4.275 ms p10, and 4.403 ms p90. The model used deterministic synthetic weights;
this was a scheduling and latency probe, not a numerical-quality test.

The retained persistent Metal Q8 head, including the final mixer and CPU top-2
scan, measured 3.002 ms wall and 2.368 ms GPU median before this milestone.
The ANE route was therefore rejected: even a smaller 4-bit representation was
about 1.3 ms slower and would also introduce a new approximation.

## Retained persistent-kernel changes

The persistent d8 VQ kernels had fallen behind the production MLX kernels in
two exact optimizations:

- cache the four packed words covering each group of eight 14-bit codes;
- evaluate three down-projection route slots across 30 SIMD lanes, then restore
  the original per-slot reduction order.

After porting both, a real-weight d8 layer retained 0.999991 cosine and
0.001447 RMSE against the MLX layer oracle. Its isolated GPU median fell from
about 0.733 ms to 0.614--0.685 ms across the observed runs.

The backend-only 48-layer trunk used token 9419, a deterministic synthetic
BF16 stream, reset state, no MTP, and 3 warmups plus 11 measured iterations.
The earlier group-3 baseline was 33.740 ms GPU / 35.172 ms wall. Three retained
group-2 runs measured:

| Run | GPU median | Wall median |
|---|---:|---:|
| 1 | 29.534 ms | 31.012 ms |
| 2 | 29.481 ms | 31.091 ms |
| 3 | 29.444 ms | 30.991 ms |

The command-buffer grouping sweep showed groups 2 and 3 were effectively tied;
larger groups regressed, reaching 39.742 ms wall at 48 layers per command
buffer. The product default was therefore not changed from measurement noise.

The language head now performs an exact GPU top-2 reduction after producing Q8
logits. The same oracle token and logit were retained, while the observed head
median changed from 3.002 ms wall / 2.368 ms GPU to 2.557 ms wall / 2.340 ms
GPU. Full logits remain available on GPU; only 486 candidates are scanned by
the CPU.

Finally, a backend-only greedy run used three warmup tokens followed by 64
continuous greedy tokens, MTP off, no prompt context, and layer group 2. The
first process reached 27.97 tok/s while weights were cold. Two warm processes
reached 30.56 and 30.52 tok/s and ended on the same token (2420).

## Decision

Retain the exact packed-word cache, down-slot parallelism, GPU top-2 reduction,
and the backend-only timing modes. Do not integrate the ANE language head. The
persistent path is now near the current MLX short-decode result, but it has not
yet demonstrated a 40 tok/s result or enough multi-token parity to become the
default VQ backend.

## Compact-codebook follow-up

The persistent d8 kernels subsequently adopted the same decode-only codebook
policy as the production MLX path: signed centred INT8 for gate and affine U8
for up/down, with per-dimension FP16 scale and bias. The original FP16
codebooks remain the source of the derived buffers and continue to serve wide
prefill. Derived persistent buffers add about 18 MiB across the 46 d8 layers.

The real-weight d8 layer fell again from about 0.615 ms to 0.531 ms. Relative
to the MLX production oracle it retained 0.999991 cosine and 0.001449 RMSE.
Three 48-layer trunk runs measured 26.205/26.215/26.226 ms GPU and
27.616/27.654/27.636 ms wall. Three 64-token greedy runs measured
33.75/33.77/33.28 tok/s and all ended on token 2420. These used the same M5 Pro,
group-2 command buffers, greedy sampling, MTP off, three warmup tokens, no
prompt context, and no thermal controller as the earlier follow-up.

This improves the warm persistent result by about 10.6% over its prior
30.53 tok/s, but still misses the 40 tok/s target. The next gate is eliminating
the embedding/trunk/head host round trips and measuring the remaining shared
expert and GDN cost before adding another numerical approximation.
