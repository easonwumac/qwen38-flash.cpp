#!/usr/bin/env python3
"""Independent MLX-Python oracle for the C++ hyper-connection smoke path."""

import argparse
import json
from pathlib import Path

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    loaded: dict[str, dict[str, mx.array]] = {}

    def tensor(name: str) -> mx.array:
        shard = index[name]
        if shard not in loaded:
            loaded[shard] = mx.load(str(args.model / shard))
        return loaded[shard][name]

    bits = 4
    group_size = 64
    hidden = config["hidden_size"]
    streams = config["hc_count"]
    ids = mx.array([9419, 11, 1814, 0], dtype=mx.int32)
    embed_base = "language_model.model.embed_tokens"
    embedding = mx.dequantize(
        tensor(embed_base + ".weight")[ids],
        tensor(embed_base + ".scales")[ids],
        tensor(embed_base + ".biases")[ids],
        group_size=group_size,
        bits=bits,
    ).reshape(1, 4, hidden)
    stream = mx.tile(embedding, (1, 1, streams))

    def projection(x: mx.array, name: str) -> mx.array:
        return mx.quantized_matmul(
            x,
            tensor(name + ".weight"),
            scales=tensor(name + ".scales"),
            biases=tensor(name + ".biases"),
            transpose=True,
            group_size=group_size,
            bits=bits,
        )

    def read(value: mx.array, prefix: str, inject: bool):
        grouped = value.reshape(1, 4, streams, hidden)
        normalized = mx.fast.rms_norm(
            grouped, mx.ones((hidden,), dtype=value.dtype), config["rms_norm_eps"]
        )
        norm_weight = tensor(prefix + ".hc_norm.weight").reshape(streams, hidden) + 1
        normalized = normalized * norm_weight
        flat = normalized.reshape(1, 4, streams * hidden)
        down = projection(flat, prefix + ".input_mix_weight_down") / streams
        up = projection(down * mx.sigmoid(down), prefix + ".input_mix_weight_up")
        mixed = (mx.sigmoid(up.reshape(1, 4, streams, hidden)) * normalized).mean(axis=2)
        if not inject:
            return mixed, None
        gate = 2 * mx.sigmoid(projection(flat, prefix + ".block_inject_weight") / streams)
        return mixed, gate.reshape(1, 4, streams, 1)

    layer_prefix = "language_model.model.layers.0.attn_hyper_connection"
    mixed, injection = read(stream, layer_prefix, True)
    written = (
        stream.reshape(1, 4, streams, hidden)
        + mixed.reshape(1, 4, 1, hidden) * injection
    ).reshape(1, 4, streams * hidden)
    final, _ = read(written, "language_model.model.hyper_connection_mixer", False)
    mx.eval(mixed, injection, written, final)
    print(
        json.dumps(
            {
                "mixed_checksum": float(mx.sum(mixed.astype(mx.float32)).item()),
                "injection_checksum": float(mx.sum(injection.astype(mx.float32)).item()),
                "stream_checksum": float(mx.sum(written.astype(mx.float32)).item()),
                "final_checksum": float(mx.sum(final.astype(mx.float32)).item()),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
