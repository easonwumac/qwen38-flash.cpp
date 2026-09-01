#!/usr/bin/env python3
"""Independent MLX-Python oracle for one Qwen3.8 sparse-MoE decode token."""

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
    loaded = {}

    def tensor(name):
        shard = index[name]
        if shard not in loaded:
            loaded[shard] = mx.load(str(args.model / shard))
        return loaded[shard][name]

    def qmm(x, name, expert=None):
        weight = tensor(name + ".weight")
        scales = tensor(name + ".scales")
        biases = tensor(name + ".biases")
        if expert is not None:
            weight, scales, biases = weight[expert], scales[expert], biases[expert]
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
    stream = mx.tile(x, (1, 1, 4))
    hc = "language_model.model.layers.0.mlp_hyper_connection"
    grouped = stream.reshape(1, 1, 4, 2560)
    normed = mx.fast.rms_norm(grouped, mx.ones((2560,), dtype=x.dtype), 1e-6)
    normed *= tensor(hc + ".hc_norm.weight").reshape(4, 2560) + 1
    flat = normed.reshape(1, 1, 10240)
    down = qmm(flat, hc + ".input_mix_weight_down") / 4
    up = qmm(down * mx.sigmoid(down), hc + ".input_mix_weight_up")
    x = (mx.sigmoid(up.reshape(1, 1, 4, 2560)) * normed).mean(axis=2)

    prefix = "language_model.model.layers.0.mlp"
    logits = x @ tensor(prefix + ".gate.weight").T
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1).reshape(-1)
    experts = mx.argsort(-probabilities)[: config["num_experts_per_tok"]]
    weights = probabilities[experts]
    if config.get("norm_topk_prob", True):
        weights /= weights.sum()
    weights = weights.astype(logits.dtype)
    mx.eval(experts, weights)
    expert_ids = [int(item) for item in experts.tolist()]

    output = None
    for position, expert in enumerate(expert_ids):
        gate = qmm(x, prefix + ".switch_mlp.gate_proj", expert)
        up = qmm(x, prefix + ".switch_mlp.up_proj", expert)
        hidden = gate * mx.sigmoid(gate) * up
        current = qmm(hidden, prefix + ".switch_mlp.down_proj", expert) * weights[position]
        output = current if output is None else output + current
    shared_gate = qmm(x, prefix + ".shared_expert.gate_proj")
    shared_up = qmm(x, prefix + ".shared_expert.up_proj")
    shared = qmm(
        shared_gate * mx.sigmoid(shared_gate) * shared_up,
        prefix + ".shared_expert.down_proj",
    )
    output += shared * mx.sigmoid(x @ tensor(prefix + ".shared_expert_gate.weight").T)
    mx.eval(output)
    print(
        json.dumps(
            {
                "experts": expert_ids,
                "weights": [float(item) for item in weights.tolist()],
                "checksum": float(mx.sum(output.astype(mx.float32)).item()),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
