#!/usr/bin/env python3
"""Independent first-token GatedDeltaNet oracle for the retained checkpoint."""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = {}

    def tensor(name):
        shard = index[name]
        if shard not in loaded:
            loaded[shard] = mx.load(str(args.model / shard))
        return loaded[shard][name]

    def qmm(x, name):
        return mx.quantized_matmul(
            x,
            tensor(name + ".weight"),
            scales=tensor(name + ".scales"),
            biases=tensor(name + ".biases"),
            transpose=True,
            group_size=64,
            bits=4,
        )

    ids = mx.array([9419], dtype=mx.int32)
    embed = "language_model.model.embed_tokens"
    embedding = mx.dequantize(
        tensor(embed + ".weight")[ids],
        tensor(embed + ".scales")[ids],
        tensor(embed + ".biases")[ids],
        group_size=64,
        bits=4,
    ).reshape(1, 1, 2560)
    stream = mx.tile(embedding, (1, 1, 4))
    hc = "language_model.model.layers.0.attn_hyper_connection"
    grouped = stream.reshape(1, 1, 4, 2560)
    normed = mx.fast.rms_norm(grouped, mx.ones((2560,), dtype=embedding.dtype), 1e-6)
    normed *= tensor(hc + ".hc_norm.weight").reshape(4, 2560) + 1
    flat = normed.reshape(1, 1, 10240)
    down = qmm(flat, hc + ".input_mix_weight_down") / 4
    up = qmm(down * mx.sigmoid(down), hc + ".input_mix_weight_up")
    x = (mx.sigmoid(up.reshape(1, 1, 4, 2560)) * normed).mean(axis=2)

    prefix = "language_model.model.layers.0.linear_attn"
    qkv = qmm(x, prefix + ".in_proj_qkv")
    conv_weight = tensor(prefix + ".conv1d.weight").reshape(10240, 4)[:, 3]
    conv = qkv * conv_weight
    conv = conv * mx.sigmoid(conv)
    query = conv[..., :2048].reshape(1, 1, 16, 128)
    key = conv[..., 2048:4096].reshape(1, 1, 16, 128)
    value = conv[..., 4096:].reshape(1, 1, 48, 128)
    ones = mx.ones((128,), dtype=query.dtype)
    query = mx.fast.rms_norm(query, ones, 1e-6) * mx.array(1.0 / 128, dtype=query.dtype)
    key = mx.fast.rms_norm(key, ones, 1e-6) * mx.array(
        1.0 / math.sqrt(128), dtype=key.dtype
    )
    query = mx.repeat(query, 3, axis=2)
    key = mx.repeat(key, 3, axis=2)
    beta = mx.sigmoid(qmm(x, prefix + ".in_proj_b")).reshape(1, 1, 48, 1)
    similarity = (query * key).sum(axis=3, keepdims=True)
    recurrent = value * beta * similarity
    normalized = mx.fast.rms_norm(
        recurrent, tensor(prefix + ".norm.weight"), config["rms_norm_eps"]
    )
    z = qmm(x, prefix + ".in_proj_z").reshape(1, 1, 48, 128)
    gated = normalized * mx.sigmoid(z)
    output = qmm(gated.reshape(1, 1, 6144), prefix + ".out_proj")
    mx.eval(output)
    print(
        json.dumps(
            {"checksum": float(mx.sum(output.astype(mx.float32)).item())},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
