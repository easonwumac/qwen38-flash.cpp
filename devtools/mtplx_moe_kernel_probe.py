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
    constexpr int ROWS = 4;
    constexpr int VALUES = 8;
    constexpr int BLOCK = VALUES * 32;
    constexpr int NG = K / GS;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = threadgroup_position_in_grid.x * 8 + sg * ROWS;
    if (row0 >= 2560) return;
    float acc[ROWS] = {0.0f};
    for (int slot = 0; slot < 10; ++slot) {
        const uint expert = experts[slot];
        const int packed_row = K / 8;
        const size_t base = (size_t)expert * 2560 + row0;
        const device uint8_t* wr =
            (const device uint8_t*)(dw + base * packed_row) + lane * 4;
        const device T* sr = ds + base * NG + lane / 8;
        const device T* br = db + base * NG + lane / 8;
        const device T* xv = h + slot * K + lane * VALUES;
        float dot[ROWS] = {0.0f};
        for (int k0 = 0; k0 < K; k0 += BLOCK) {
            float xt[VALUES] = {0.0f};
            float sum = 0.0f;
            if (k0 + lane * VALUES < K) {
                for (int i = 0; i < VALUES; i += 4) {
                    float x0 = (float)xv[i], x1 = (float)xv[i + 1];
                    float x2 = (float)xv[i + 2], x3 = (float)xv[i + 3];
                    sum += x0 + x1 + x2 + x3;
                    xt[i] = x0;
                    xt[i + 1] = x1 / 16.0f;
                    xt[i + 2] = x2 / 256.0f;
                    xt[i + 3] = x3 / 4096.0f;
                }
            }
            for (int r = 0; r < ROWS; ++r) {
                const device uint16_t* words =
                    (const device uint16_t*)(wr + r * packed_row * 4);
                float q = 0.0f;
                for (int i = 0; i < VALUES / 4; ++i) {
                    uint16_t value = words[i];
                    q += xt[4 * i] * (value & 0x000f) +
                        xt[4 * i + 1] * (value & 0x00f0) +
                        xt[4 * i + 2] * (value & 0x0f00) +
                        xt[4 * i + 3] * (value & 0xf000);
                }
                dot[r] += (float)sr[r * NG] * q + (float)br[r * NG] * sum;
            }
            wr += BLOCK / 2;
            sr += BLOCK / GS;
            br += BLOCK / GS;
            xv += BLOCK;
        }
        for (int r = 0; r < ROWS; ++r) {
            dot[r] = simd_sum(dot[r]);
            acc[r] += rw[slot] * (float)((T)dot[r]);
        }
    }
    if (lane == 0) {
        for (int r = 0; r < ROWS; ++r) y[row0 + r] = (T)acc[r];
    }
"""


def qdot_gate_source(rows: int, values: int) -> str:
    """Build an MLX-qmv-style gate/up kernel for a fixed output tile."""
    return rf"""
    constexpr int K = 2560;
    constexpr int GS = 64;
    constexpr int ROWS = {rows};
    constexpr int VALUES = {values};
    constexpr int BLOCK = VALUES * 32;
    constexpr int NG = K / GS;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = threadgroup_position_in_grid.x * (2 * ROWS) + sg * ROWS;
    const int slot = row0 / 640;
    const int expert_row = row0 % 640;
    if (slot >= 10) return;
    const uint expert = experts[slot];
    const int packed_row = K / 8;
    const size_t expert_base = (size_t)expert * 640 + expert_row;
    const device uint32_t* gwr = gw + expert_base * packed_row + lane * (VALUES / 8);
    const device uint32_t* uwr = uw + expert_base * packed_row + lane * (VALUES / 8);
    const device T* gsr = gs + expert_base * NG + lane / (GS / VALUES);
    const device T* gbr = gb + expert_base * NG + lane / (GS / VALUES);
    const device T* usr = us + expert_base * NG + lane / (GS / VALUES);
    const device T* ubr = ub + expert_base * NG + lane / (GS / VALUES);
    const device T* xv = x + lane * VALUES;
    float gate_acc[ROWS] = {{0.0f}};
    float up_acc[ROWS] = {{0.0f}};
    for (int k0 = 0; k0 < K; k0 += BLOCK) {{
        float xt[VALUES];
        float sum = 0.0f;
        for (int i = 0; i < VALUES; i += 4) {{
            float x0 = (float)xv[i], x1 = (float)xv[i + 1];
            float x2 = (float)xv[i + 2], x3 = (float)xv[i + 3];
            sum += x0 + x1 + x2 + x3;
            xt[i] = x0;
            xt[i + 1] = x1 / 16.0f;
            xt[i + 2] = x2 / 256.0f;
            xt[i + 3] = x3 / 4096.0f;
        }}
        for (int r = 0; r < ROWS; ++r) {{
            const device uint16_t* wg =
                (const device uint16_t*)(gwr + r * packed_row);
            const device uint16_t* wu =
                (const device uint16_t*)(uwr + r * packed_row);
            float qg = 0.0f, qu = 0.0f;
            for (int i = 0; i < VALUES / 4; ++i) {{
                uint16_t gv = wg[i], uv = wu[i];
                qg += xt[4 * i] * (gv & 0x000f) +
                    xt[4 * i + 1] * (gv & 0x00f0) +
                    xt[4 * i + 2] * (gv & 0x0f00) +
                    xt[4 * i + 3] * (gv & 0xf000);
                qu += xt[4 * i] * (uv & 0x000f) +
                    xt[4 * i + 1] * (uv & 0x00f0) +
                    xt[4 * i + 2] * (uv & 0x0f00) +
                    xt[4 * i + 3] * (uv & 0xf000);
            }}
            gate_acc[r] += (float)gsr[r * NG] * qg + (float)gbr[r * NG] * sum;
            up_acc[r] += (float)usr[r * NG] * qu + (float)ubr[r * NG] * sum;
        }}
        gwr += BLOCK / 8;
        uwr += BLOCK / 8;
        gsr += BLOCK / GS;
        gbr += BLOCK / GS;
        usr += BLOCK / GS;
        ubr += BLOCK / GS;
        xv += BLOCK;
    }}
    for (int r = 0; r < ROWS; ++r) {{
        gate_acc[r] = simd_sum(gate_acc[r]);
        up_acc[r] = simd_sum(up_acc[r]);
        if (lane == 0) {{
            float gv = (float)((T)gate_acc[r]);
            float uv = (float)((T)up_acc[r]);
            h[(size_t)slot * 640 + expert_row + r] =
                (T)((float)((T)(gv / (1.0f + exp(-gv)))) * uv);
        }}
    }}
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

    qdot_kernels = {}
    for rows in (1, 2, 4, 8):
        for values in (8, 16):
            key = f"r{rows}v{values}"
            qdot_kernels[key] = mx.fast.metal_kernel(
                name=f"qwen38_moe_gate_{key}",
                input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "experts"],
                output_names=["h"], header=HEADER, source=qdot_gate_source(rows, values))

    def custom():
        (hidden,) = ka(inputs=[x.reshape(-1), tensor(base + ".gate_proj.weight"),
            tensor(base + ".gate_proj.scales"), tensor(base + ".gate_proj.biases"),
            tensor(base + ".up_proj.weight"), tensor(base + ".up_proj.scales"),
            tensor(base + ".up_proj.biases"), experts], template=[("T", x.dtype)],
            grid=(200 * 1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(10, 640)], output_dtypes=[x.dtype])
        (output,) = kb(inputs=[hidden, tensor(base + ".down_proj.weight"),
            tensor(base + ".down_proj.scales"), tensor(base + ".down_proj.biases"),
            experts, weights], template=[("T", x.dtype)], grid=(320 * 64, 1, 1),
            threadgroup=(64, 1, 1), output_shapes=[(2560,)], output_dtypes=[x.dtype])
        return hidden, output

    _, candidate = custom()
    mx.eval(reference, candidate)
    qdot_results = {}
    gate_inputs = [x.reshape(-1), tensor(base + ".gate_proj.weight"),
        tensor(base + ".gate_proj.scales"), tensor(base + ".gate_proj.biases"),
        tensor(base + ".up_proj.weight"), tensor(base + ".up_proj.scales"),
        tensor(base + ".up_proj.biases"), experts]
    baseline_hidden, _ = custom()
    mx.eval(baseline_hidden)
    for key, kernel in qdot_kernels.items():
        rows = int(key.split("v")[0][1:])
        timings_for_variant = []
        output = None
        for _ in range(16):
            started = time.perf_counter()
            (output,) = kernel(inputs=gate_inputs, template=[("T", x.dtype)],
                grid=((6400 // (2 * rows)) * 64, 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(10, 640)], output_dtypes=[x.dtype])
            mx.eval(output)
            timings_for_variant.append((time.perf_counter() - started) * 1000)
        error = mx.abs(output.astype(mx.float32) - baseline_hidden.astype(mx.float32))
        qdot_results[key] = {
            "median_ms": sorted(timings_for_variant[2:])[len(timings_for_variant[2:]) // 2],
            "max_abs": float(error.max().item()),
        }
    gate_timings = []
    for _ in range(8):
        started = time.perf_counter(); hidden, _ = custom(); mx.eval(hidden)
        gate_timings.append((time.perf_counter() - started) * 1000)
    timings = []
    for _ in range(8):
        started = time.perf_counter(); _, value = custom(); mx.eval(value)
        timings.append((time.perf_counter() - started) * 1000)
    delta = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
    print(json.dumps({"experts": experts.tolist(),
        "reference_checksum": float(reference.astype(mx.float32).sum().item()),
        "candidate_checksum": float(candidate.astype(mx.float32).sum().item()),
        "max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item()),
        "gate_warm_median_ms": sorted(gate_timings[1:])[3],
        "qdot_gate": qdot_results,
        "cold_ms": timings[0], "warm_median_ms": sorted(timings[1:])[3]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
