#!/usr/bin/env python3
"""Probe fused Q4 LM-head dot products plus threadgroup-local argmax."""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx


SOURCE = r"""
    constexpr int K = 2560;
    constexpr int GS = 64;
    constexpr int NG = K / GS;
    constexpr int WPG = GS / 8;
    constexpr int VOCAB = 248320;
    threadgroup float local_values[32];
    threadgroup uint local_ids[32];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint row = threadgroup_position_in_grid.x * 32 + sg;
    float acc = -INFINITY;
    if (row < VOCAB) {
        const device uint32_t* wr = w + (size_t)row * (K / 8);
        const device T* sr = scales + (size_t)row * NG;
        const device T* br = biases + (size_t)row * NG;
        acc = 0.0f;
        for (int g = lane; g < NG; g += 32) {
            float qacc = 0.0f, xsum = 0.0f;
            for (int wi = 0; wi < WPG; ++wi) {
                uint32_t word = wr[g * WPG + wi];
                for (int nib = 0; nib < 8; ++nib) {
                    float xv = (float)x[g * GS + wi * 8 + nib];
                    qacc += (float)((word >> (4 * nib)) & 15) * xv;
                    xsum += xv;
                }
            }
            acc += (float)sr[g] * qacc + (float)br[g] * xsum;
        }
        acc = simd_sum(acc);
    }
    if (lane == 0) {
        local_values[sg] = (float)((T)acc);
        local_ids[sg] = row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float best = local_values[0];
        uint best_id = local_ids[0];
        for (int i = 1; i < 32; ++i) {
            float value = local_values[i];
            uint id = local_ids[i];
            if (value > best || (value == best && id < best_id)) {
                best = value; best_id = id;
            }
        }
        winners[threadgroup_position_in_grid.x * 2] = best;
        winners[threadgroup_position_in_grid.x * 2 + 1] = (float)best_id;
    }
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = {}

    def tensor(name):
        shard = index[name]
        if shard not in loaded:
            loaded[shard] = mx.load(str(args.model / shard))
        return loaded[shard][name]

    prefix = "language_model.lm_head"
    weight = tensor(prefix + ".weight")
    scales = tensor(prefix + ".scales")
    biases = tensor(prefix + ".biases")
    x = (mx.sin(mx.arange(2560, dtype=mx.float32) * 0.01) * 0.01).astype(mx.bfloat16)
    kernel = mx.fast.metal_kernel(
        name="qwen38_lm_head_block_argmax",
        input_names=["x", "w", "scales", "biases"],
        output_names=["winners"],
        source=SOURCE,
    )

    def generic():
        logits = mx.quantized_matmul(
            x, weight, scales, biases, transpose=True, group_size=64, bits=4
        )
        token = mx.argmax(logits)
        return token, logits[token].astype(mx.float32)

    def fused():
        (winners,) = kernel(
            inputs=[x, weight, scales, biases],
            template=[("T", x.dtype)],
            grid=(7760 * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(7760, 2)],
            output_dtypes=[mx.float32],
        )
        block = mx.argmax(winners[:, 0])
        return winners[block, 1].astype(mx.uint32), winners[block, 0]

    generic_token, generic_logit = generic()
    fused_token, fused_logit = fused()
    mx.eval(generic_token, generic_logit, fused_token, fused_logit)

    def timing(function):
        samples = []
        for _ in range(8):
            started = time.perf_counter()
            token, logit = function()
            mx.eval(token, logit)
            samples.append((time.perf_counter() - started) * 1000)
        return samples[0], sorted(samples[1:])[3]

    generic_cold, generic_warm = timing(generic)
    fused_cold, fused_warm = timing(fused)
    print(json.dumps({
        "generic_token": int(generic_token.item()),
        "fused_token": int(fused_token.item()),
        "generic_logit": float(generic_logit.item()),
        "fused_logit": float(fused_logit.item()),
        "generic_cold_ms": generic_cold,
        "generic_warm_ms": generic_warm,
        "fused_cold_ms": fused_cold,
        "fused_warm_ms": fused_warm,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
