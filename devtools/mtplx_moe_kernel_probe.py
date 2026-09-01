#!/usr/bin/env python3
"""Probe a two-dispatch selected-MoE design derived from MTPLX.

The kernel structure and reduction strategy are adapted from MTPLX's
``moe_glu_decode.py`` (Apache-2.0). This probe keeps gate/up packs separate so
it can run directly against the retained checkpoint without repacking weights.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

GATE_UP = r"""
    constexpr int K = 2560;
    constexpr int GS = 64;
    constexpr int NG = K / GS;
    constexpr int WPG = GS / 8;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;
    if (gsid >= 6400) return;
    const int slot = gsid / 640;
    const int row = gsid % 640;
    const uint expert = experts[slot];
    const device uint32_t* gwr = gw + ((size_t)expert * 640 + row) * (K / 8);
    const device uint32_t* uwr = uw + ((size_t)expert * 640 + row) * (K / 8);
    const device T* gsr = gs + ((size_t)expert * 640 + row) * NG;
    const device T* gbr = gb + ((size_t)expert * 640 + row) * NG;
    const device T* usr = us + ((size_t)expert * 640 + row) * NG;
    const device T* ubr = ub + ((size_t)expert * 640 + row) * NG;
    float ag = 0.0f, au = 0.0f;
    for (int g = lane; g < NG; g += 32) {
        const device T* xg = x + g * GS;
        float qg = 0.0f, qu = 0.0f, xs = 0.0f;
        for (int wi = 0; wi < WPG; ++wi) {
            uint32_t wg = gwr[g * WPG + wi];
            uint32_t wu = uwr[g * WPG + wi];
            for (int nib = 0; nib < 8; ++nib) {
                float xv = (float)xg[wi * 8 + nib];
                xs += xv;
                qg += (float)((wg >> (4 * nib)) & 15) * xv;
                qu += (float)((wu >> (4 * nib)) & 15) * xv;
            }
        }
        ag += (float)gsr[g] * qg + (float)gbr[g] * xs;
        au += (float)usr[g] * qu + (float)ubr[g] * xs;
    }
    ag = simd_sum(ag); au = simd_sum(au);
    if (lane == 0) {
        float gv = (float)((T)ag), uv = (float)((T)au);
        h[(size_t)slot * 640 + row] = (T)((float)((T)(gv / (1.0f + exp(-gv)))) * uv);
    }
"""

DOWN = r"""
    constexpr int K = 640;
    constexpr int GS = 64;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint d = threadgroup_position_in_grid.x * 32 + sg;
    if (d >= 2560) return;
    float acc = 0.0f;
    for (int slot = 0; slot < 10; ++slot) {
        const uint expert = experts[slot];
        const device uint32_t* wr = dw + ((size_t)expert * 2560 + d) * (K / 8);
        const device T* sr = ds + ((size_t)expert * 2560 + d) * (K / GS);
        const device T* br = db + ((size_t)expert * 2560 + d) * (K / GS);
        const device T* hv = h + slot * K;
        float dot = 0.0f;
        for (int k = lane; k < K; k += 32) {
            uint32_t word = wr[k / 8];
            float q = (float)((word >> (4 * (k % 8))) & 15);
            dot = fma((float)sr[k / GS] * q + (float)br[k / GS], (float)hv[k], dot);
        }
        dot = simd_sum(dot);
        acc += rw[slot] * (float)((T)dot);
    }
    if (lane == 0) y[d] = (T)acc;
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

    def qmm(x, name, expert=None):
        w, s, b = tensor(name + ".weight"), tensor(name + ".scales"), tensor(name + ".biases")
        if expert is not None:
            w, s, b = w[expert], s[expert], b[expert]
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)

    ids = mx.array([9419], dtype=mx.int32)
    ep = "language_model.model.embed_tokens"
    x = mx.dequantize(tensor(ep + ".weight")[ids], tensor(ep + ".scales")[ids],
                      tensor(ep + ".biases")[ids], group_size=64, bits=4).reshape(1, 1, 2560)
    stream = mx.tile(x, (1, 1, 4))
    hc = "language_model.model.layers.0.mlp_hyper_connection"
    normed = mx.fast.rms_norm(stream.reshape(1, 1, 4, 2560),
                              mx.ones((2560,), dtype=x.dtype), 1e-6)
    normed *= tensor(hc + ".hc_norm.weight").reshape(4, 2560) + 1
    flat = normed.reshape(1, 1, 10240)
    hd = qmm(flat, hc + ".input_mix_weight_down") / 4
    hu = qmm(hd * mx.sigmoid(hd), hc + ".input_mix_weight_up")
    x = (mx.sigmoid(hu.reshape(1, 1, 4, 2560)) * normed).mean(axis=2)

    prefix = "language_model.model.layers.0.mlp"
    probs = mx.softmax((x @ tensor(prefix + ".gate.weight").T).astype(mx.float32), axis=-1)
    experts = mx.argpartition(probs, kth=-10, axis=-1)[..., -10:].reshape(-1).astype(mx.uint32)
    weights = mx.take(probs.reshape(-1), experts).astype(mx.float32)
    weights /= weights.sum()
    base = prefix + ".switch_mlp"

    refs = []
    for e, weight in zip(experts.tolist(), weights.tolist()):
        gate = qmm(x, base + ".gate_proj", int(e))
        up = qmm(x, base + ".up_proj", int(e))
        refs.append(qmm(gate * mx.sigmoid(gate) * up, base + ".down_proj", int(e)) * weight)
    reference = sum(refs).reshape(-1)

    ka = mx.fast.metal_kernel(name="qwen38_mtplx_sep_gu", input_names=[
        "x", "gw", "gs", "gb", "uw", "us", "ub", "experts"], output_names=["h"],
        header=HEADER, source=GATE_UP)
    kb = mx.fast.metal_kernel(name="qwen38_mtplx_down", input_names=[
        "h", "dw", "ds", "db", "experts", "rw"], output_names=["y"],
        header=HEADER, source=DOWN)

    def custom():
        (hidden,) = ka(inputs=[x.reshape(-1), tensor(base + ".gate_proj.weight"),
            tensor(base + ".gate_proj.scales"), tensor(base + ".gate_proj.biases"),
            tensor(base + ".up_proj.weight"), tensor(base + ".up_proj.scales"),
            tensor(base + ".up_proj.biases"), experts], template=[("T", x.dtype)],
            grid=(200 * 1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(10, 640)], output_dtypes=[x.dtype])
        (output,) = kb(inputs=[hidden, tensor(base + ".down_proj.weight"),
            tensor(base + ".down_proj.scales"), tensor(base + ".down_proj.biases"),
            experts, weights], template=[("T", x.dtype)], grid=(80 * 1024, 1, 1),
            threadgroup=(1024, 1, 1), output_shapes=[(2560,)], output_dtypes=[x.dtype])
        return hidden, output

    _, candidate = custom()
    mx.eval(reference, candidate)
    timings = []
    for _ in range(8):
        started = time.perf_counter(); _, value = custom(); mx.eval(value)
        timings.append((time.perf_counter() - started) * 1000)
    delta = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
    print(json.dumps({"experts": experts.tolist(),
        "reference_checksum": float(reference.astype(mx.float32).sum().item()),
        "candidate_checksum": float(candidate.astype(mx.float32).sum().item()),
        "max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item()),
        "cold_ms": timings[0], "warm_median_ms": sorted(timings[1:])[3]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
