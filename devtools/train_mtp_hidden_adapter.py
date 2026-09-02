#!/usr/bin/env python3
"""Distill target final hidden states into a small depth-aware MTP residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train_mtp_head_adapter import acceptance_metrics, greedy_metrics, load_rows


class HiddenAdapter(nn.Module):
    def __init__(self, width: int, rank: int, depth_count: int) -> None:
        super().__init__()
        self.lora_a = (
            0.01 * mx.random.normal((depth_count, rank, width))
        ).astype(mx.float32)
        self.lora_b = mx.zeros((width, rank), dtype=mx.float32)

    def __call__(self, hidden: mx.array, depth: mx.array) -> mx.array:
        selected_a = self.lora_a[depth.astype(mx.int32) - 1]
        projected = mx.matmul(hidden[:, None, :], selected_a.swapaxes(1, 2))[:, 0, :]
        return hidden + projected @ self.lora_b.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("lm_head_shard", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = load_rows(args.calibration)
    if rows.target_hidden is None:
        parser.error("calibration file does not contain target hidden states (requires v3)")
    rng = np.random.default_rng(args.seed)
    active = np.flatnonzero(rows.active)
    groups = np.unique(rows.request_id[active])
    rng.shuffle(groups)
    split = max(1, min(len(groups) - 1, round(args.train_fraction * len(groups))))
    train_groups = set(int(value) for value in groups[:split])
    train_indices = active[
        np.array([int(rows.request_id[i]) in train_groups for i in active])
    ]
    eval_indices = np.flatnonzero(
        np.array([int(value) not in train_groups for value in rows.request_id])
    )

    hidden = mx.array(rows.hidden)
    target_hidden = mx.array(rows.target_hidden)
    depths = mx.array(rows.depth)
    depth_count = int(np.max(rows.depth))
    adapter = HiddenAdapter(rows.hidden.shape[1], args.rank, depth_count)
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def loss_fn(model: HiddenAdapter, batch: mx.array) -> mx.array:
        corrected = model(hidden[batch], depths[batch])
        residual = corrected - target_hidden[batch]
        return mx.mean(mx.square(residual))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    final_loss = 0.0
    for _ in range(args.steps):
        batch_np = rng.choice(
            train_indices, size=min(args.batch_size, len(train_indices)), replace=True
        )
        loss, gradients = loss_and_grad(adapter, mx.array(batch_np.astype(np.int32)))
        optimizer.update(adapter, gradients)
        mx.eval(adapter.parameters(), optimizer.state, loss)
        final_loss = float(loss.item())

    eval_ids = mx.array(eval_indices.astype(np.int32))
    base_hidden = hidden[eval_ids]
    corrected_hidden = adapter(base_hidden, depths[eval_ids])
    expected_hidden = target_hidden[eval_ids]
    base_mse = float(mx.mean(mx.square(base_hidden - expected_hidden)).item())
    corrected_mse = float(mx.mean(mx.square(corrected_hidden - expected_hidden)).item())

    weights = mx.load(str(args.lm_head_shard))
    prefix = "language_model.lm_head"
    base_logits = mx.quantized_matmul(
        base_hidden,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.scales"],
        weights[f"{prefix}.biases"],
        transpose=True,
        group_size=64,
        bits=4,
    ).astype(mx.float32)
    corrected_logits = mx.quantized_matmul(
        corrected_hidden,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.scales"],
        weights[f"{prefix}.biases"],
        transpose=True,
        group_size=64,
        bits=4,
    ).astype(mx.float32)
    mx.eval(base_logits, corrected_logits)
    base_np = np.asarray(base_logits)
    corrected_np = np.asarray(corrected_logits)
    report = {
        "schema_version": 1,
        "rank": args.rank,
        "steps": args.steps,
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "train_final_mse": final_loss,
        "base_hidden_mse": base_mse,
        "corrected_hidden_mse": corrected_mse,
        "eval": greedy_metrics(corrected_np, rows, eval_indices),
        "acceptance": acceptance_metrics(corrected_np, rows, eval_indices),
        "base_acceptance": acceptance_metrics(base_np, rows, eval_indices),
    }
    arrays = {"hidden_lora_b": adapter.lora_b.astype(mx.bfloat16)}
    for depth in range(depth_count):
        arrays[f"hidden_lora_a.depth{depth + 1}"] = adapter.lora_a[depth].astype(mx.bfloat16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output), arrays, {"metadata_json": json.dumps(report, sort_keys=True)}
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
