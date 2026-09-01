#!/usr/bin/env python3
"""Independent two-step full-attention oracle below the QSA budget."""

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

    embed = "language_model.model.embed_tokens"
    hc = "language_model.model.layers.3.attn_hyper_connection"
    prefix = "language_model.model.layers.3.self_attn"

    def make_input(token_id):
        ids = mx.array([token_id], dtype=mx.int32)
        embedding = mx.dequantize(
            tensor(embed + ".weight")[ids],
            tensor(embed + ".scales")[ids],
            tensor(embed + ".biases")[ids],
            group_size=64,
            bits=4,
        ).reshape(1, 1, 2560)
        grouped = mx.tile(embedding, (1, 1, 4)).reshape(1, 1, 4, 2560)
        normed = mx.fast.rms_norm(
            grouped, mx.ones((2560,), dtype=embedding.dtype), 1e-6
        )
        normed *= tensor(hc + ".hc_norm.weight").reshape(4, 2560) + 1
        flat = normed.reshape(1, 1, 10240)
        down = qmm(flat, hc + ".input_mix_weight_down") / 4
        up = qmm(down * mx.sigmoid(down), hc + ".input_mix_weight_up")
        return (mx.sigmoid(up.reshape(1, 1, 4, 2560)) * normed).mean(axis=2)

    def rope(x, position):
        rotary = 64
        half = rotary // 2
        frequencies = [
            config["rope_parameters"]["rope_theta"] ** (-2.0 * index / rotary)
            for index in range(half)
        ]
        angles = [position * frequency for frequency in frequencies]
        cosine = mx.array(
            [math.cos(angle) for angle in angles] * 2, dtype=x.dtype
        ).reshape(1, 1, 1, rotary)
        sine = mx.array(
            [math.sin(angle) for angle in angles] * 2, dtype=x.dtype
        ).reshape(1, 1, 1, rotary)
        part = x[..., :rotary]
        rotated = mx.concatenate((-part[..., half:], part[..., :half]), axis=-1)
        return mx.concatenate((part * cosine + rotated * sine, x[..., rotary:]), axis=-1)

    keys = None
    values = None
    position = 0

    def step(x):
        nonlocal keys, values, position
        query_gate = qmm(x, prefix + ".q_proj").reshape(1, 1, 24, 512)
        query, gate = query_gate[..., :256], query_gate[..., 256:]
        query_norm = tensor(prefix + ".q_norm.weight") + 1
        key_norm = tensor(prefix + ".k_norm.weight") + 1
        query = mx.fast.rms_norm(query, query_norm, 1e-6).swapaxes(1, 2)
        key = mx.fast.rms_norm(
            qmm(x, prefix + ".k_proj").reshape(1, 1, 2, 256), key_norm, 1e-6
        ).swapaxes(1, 2)
        value = qmm(x, prefix + ".v_proj").reshape(1, 1, 2, 256).swapaxes(1, 2)
        query = rope(query, position)
        key = rope(key, position)
        keys = key if keys is None else mx.concatenate((keys, key), axis=2)
        values = value if values is None else mx.concatenate((values, value), axis=2)
        position += 1
        repeated_keys = mx.repeat(keys, 12, axis=1)
        repeated_values = mx.repeat(values, 12, axis=1)
        scores = query @ repeated_keys.swapaxes(2, 3)
        scores *= mx.array(1.0 / math.sqrt(256), dtype=scores.dtype)
        probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(query.dtype)
        attended = (probabilities @ repeated_values).swapaxes(1, 2).reshape(1, 1, 6144)
        gated = attended * mx.sigmoid(gate.reshape(1, 1, 6144))
        return qmm(gated, prefix + ".o_proj")

    first = step(make_input(9419))
    mx.eval(first, keys, values)
    second = step(make_input(11))
    mx.eval(second, keys, values)
    print(
        json.dumps(
            {
                "first_checksum": float(mx.sum(first.astype(mx.float32)).item()),
                "second_checksum": float(mx.sum(second.astype(mx.float32)).item()),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
