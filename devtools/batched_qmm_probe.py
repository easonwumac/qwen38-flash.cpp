#!/usr/bin/env python3
"""Probe MLX batched selected-expert QMM semantics on the retained checkpoint."""

import argparse
import json
from pathlib import Path

import mlx.core as mx


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

    def qmm(x, name, experts=None):
        weight = tensor(name + ".weight")
        scales = tensor(name + ".scales")
        biases = tensor(name + ".biases")
        if experts is not None:
            weight, scales, biases = weight[experts], scales[experts], biases[experts]
        return mx.quantized_matmul(
            x, weight, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
        )

    ids = mx.array([9419], dtype=mx.int32)
    embed = "language_model.model.embed_tokens"
    x = mx.dequantize(
        tensor(embed + ".weight")[ids],
        tensor(embed + ".scales")[ids],
        tensor(embed + ".biases")[ids],
        group_size=64,
        bits=4,
    ).reshape(1, 1, 2560)
    experts = mx.array([78, 62, 113, 257, 137, 249, 232, 239, 254, 51], dtype=mx.int32)
    base = "language_model.model.layers.0.mlp.switch_mlp"
    gate_prefix = base + ".gate_proj"
    up_prefix = base + ".up_proj"
    down_prefix = base + ".down_proj"
    reference = mx.concatenate([qmm(x, gate_prefix, int(expert)) for expert in experts.tolist()])
    broadcast = qmm(x, gate_prefix, experts)
    tiled = qmm(mx.broadcast_to(x, (10, 1, 2560)), gate_prefix, experts)
    up_reference = mx.concatenate([qmm(x, up_prefix, int(expert)) for expert in experts.tolist()])
    up_batched = qmm(x, up_prefix, experts)
    reference_hidden = reference * mx.sigmoid(reference) * up_reference
    batched_hidden = broadcast * mx.sigmoid(broadcast) * up_batched
    down_reference = mx.concatenate([
        qmm(reference_hidden[i : i + 1], down_prefix, int(expert))
        for i, expert in enumerate(experts.tolist())
    ])
    down_batched = qmm(batched_hidden, down_prefix, experts)
    mx.eval(reference, broadcast, tiled, up_reference, up_batched, down_reference, down_batched)
    print(
        json.dumps(
            {
                "reference_shape": reference.shape,
                "broadcast_shape": broadcast.shape,
                "tiled_shape": tiled.shape,
                "broadcast_max_abs": float(
                    mx.max(mx.abs(reference - broadcast)).item()
                ),
                "tiled_max_abs": float(mx.max(mx.abs(reference - tiled)).item()),
                "up_max_abs": float(mx.max(mx.abs(up_reference - up_batched)).item()),
                "hidden_max_abs": float(
                    mx.max(mx.abs(reference_hidden - batched_hidden)).item()
                ),
                "down_max_abs": float(
                    mx.max(mx.abs(down_reference - down_batched)).item()
                ),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
