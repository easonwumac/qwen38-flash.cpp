#!/usr/bin/env python3
"""Independent stateful GatedDeltaNet oracle for the retained checkpoint."""

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
    hc = "language_model.model.layers.0.attn_hyper_connection"
    prefix = "language_model.model.layers.0.linear_attn"

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

    conv_state = mx.zeros((1, 3, 10240), dtype=mx.bfloat16)
    recurrent_state = mx.zeros((1, 48, 128, 128), dtype=mx.bfloat16)

    def step(x):
        nonlocal conv_state, recurrent_state
        qkv = qmm(x, prefix + ".in_proj_qkv")
        conv_input = mx.concatenate((conv_state, qkv), axis=1)
        conv_state = conv_input[:, 1:]
        conv = mx.conv1d(
            conv_input,
            tensor(prefix + ".conv1d.weight"),
            stride=1,
            padding=0,
            dilation=1,
            groups=10240,
        )
        conv *= mx.sigmoid(conv)
        query = conv[..., :2048].reshape(1, 1, 16, 128)
        key = conv[..., 2048:4096].reshape(1, 1, 16, 128)
        value = conv[..., 4096:].reshape(1, 1, 48, 128)
        ones = mx.ones((128,), dtype=query.dtype)
        query = mx.fast.rms_norm(query, ones, 1e-6) * mx.array(
            1.0 / 128, dtype=query.dtype
        )
        key = mx.fast.rms_norm(key, ones, 1e-6) * mx.array(
            1.0 / math.sqrt(128), dtype=key.dtype
        )
        query = mx.repeat(query, 3, axis=2).reshape(1, 48, 1, 128)
        key = mx.repeat(key, 3, axis=2).reshape(1, 48, 1, 128)
        value = value.reshape(1, 48, 128)
        beta = mx.sigmoid(qmm(x, prefix + ".in_proj_b")).reshape(1, 48, 1)
        decay_input = qmm(x, prefix + ".in_proj_a") + tensor(prefix + ".dt_bias")
        decay = mx.exp(
            -mx.exp(tensor(prefix + ".A_log").astype(mx.float32))
            * mx.log1p(mx.exp(decay_input.astype(mx.float32)))
        ).astype(mx.bfloat16)
        recurrent_state *= decay.reshape(1, 48, 1, 1)
        recalled = (recurrent_state * key).sum(axis=3)
        delta = (value - recalled) * beta
        recurrent_state += delta.reshape(1, 48, 128, 1) * key
        recurrent = (recurrent_state * query).sum(axis=3).reshape(1, 1, 48, 128)
        normalized = mx.fast.rms_norm(
            recurrent, tensor(prefix + ".norm.weight"), config["rms_norm_eps"]
        )
        z = qmm(x, prefix + ".in_proj_z").reshape(1, 1, 48, 128)
        output = qmm(
            (normalized * mx.sigmoid(z)).reshape(1, 1, 6144),
            prefix + ".out_proj",
        )
        return output

    first = step(make_input(9419))
    mx.eval(first, conv_state, recurrent_state)
    second = step(make_input(11))
    mx.eval(second, conv_state, recurrent_state)
    print(
        json.dumps(
            {
                "first_checksum": float(mx.sum(first.astype(mx.float32)).item()),
                "second_checksum": float(mx.sum(second.astype(mx.float32)).item()),
                "state_checksum": float(mx.sum(recurrent_state.astype(mx.float32)).item()),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
