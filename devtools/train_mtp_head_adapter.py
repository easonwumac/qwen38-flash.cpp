#!/usr/bin/env python3
"""Train a target-safe LoRA correction on captured Qwen3.8 MTP head rows."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


MAGIC = b"Q38MTPA1"
MAGIC_V2 = b"Q38MTPA2"
MAGIC_V3 = b"Q38MTPA3"
HEADER = struct.Struct("<8sII")
ROW_PREFIX = struct.Struct("<IIIIQ")
ROW_PREFIX_V2 = struct.Struct("<IIIIQQ")


@dataclass(frozen=True)
class CalibrationRows:
    hidden: np.ndarray
    target_hidden: np.ndarray | None
    depth: np.ndarray
    draft: np.ndarray
    target: np.ndarray
    matched: np.ndarray
    position: np.ndarray
    request_id: np.ndarray
    round_id: np.ndarray
    active: np.ndarray


def load_rows(path: Path) -> CalibrationRows:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("calibration file is truncated before its header")
    magic, version, width = HEADER.unpack_from(data)
    if (magic, version) not in {
        (MAGIC, 1), (MAGIC_V2, 2), (MAGIC_V3, 3)
    } or width < 1:
        raise ValueError("unsupported MTP calibration header")
    row_prefix = ROW_PREFIX_V2 if version >= 2 else ROW_PREFIX
    row_size = row_prefix.size + width * 4 * (2 if version == 3 else 1)
    payload_size = len(data) - HEADER.size
    if payload_size % row_size:
        raise ValueError("calibration file contains a partial row")
    count = payload_size // row_size
    hidden = np.empty((count, width), dtype=np.float32)
    target_hidden = np.empty((count, width), dtype=np.float32) if version == 3 else None
    depth = np.empty(count, dtype=np.uint32)
    draft = np.empty(count, dtype=np.uint32)
    target = np.empty(count, dtype=np.uint32)
    matched = np.empty(count, dtype=np.bool_)
    position = np.empty(count, dtype=np.uint64)
    request_id = np.zeros(count, dtype=np.uint64)
    offset = HEADER.size
    for index in range(count):
        values = row_prefix.unpack_from(data, offset)
        d, dr, ta, ma, pos = values[:5]
        if version >= 2:
            request_id[index] = values[5]
        offset += row_prefix.size
        hidden[index] = np.frombuffer(data, dtype="<f4", count=width, offset=offset)
        offset += width * 4
        if target_hidden is not None:
            target_hidden[index] = np.frombuffer(
                data, dtype="<f4", count=width, offset=offset
            )
            offset += width * 4
        depth[index], draft[index], target[index] = d, dr, ta
        matched[index], position[index] = bool(ma), pos

    active = np.zeros(count, dtype=np.bool_)
    round_id = np.empty(count, dtype=np.uint64)
    prior_position: int | None = None
    prefix_matches = True
    current_round = -1
    for index in range(count):
        current = int(position[index])
        if prior_position != current:
            prior_position = current
            prefix_matches = True
            current_round += 1
        round_id[index] = current_round
        active[index] = prefix_matches
        prefix_matches = prefix_matches and bool(matched[index])
    return CalibrationRows(
        hidden, target_hidden, depth, draft, target, matched,
        position, request_id, round_id, active
    )


class HeadAdapter(nn.Module):
    def __init__(self, width: int, vocabulary: int, rank: int, depth_count: int) -> None:
        super().__init__()
        self.lora_a = (
            0.01 * mx.random.normal((depth_count, rank, width))
        ).astype(mx.float32)
        self.lora_b = mx.zeros((vocabulary, rank), dtype=mx.float32)

    def __call__(self, hidden: mx.array, depth: mx.array) -> mx.array:
        projected = self.project(hidden, depth)
        return projected @ self.lora_b.T

    def project(self, hidden: mx.array, depth: mx.array) -> mx.array:
        selected_a = self.lora_a[depth.astype(mx.int32) - 1]
        return mx.matmul(hidden[:, None, :], selected_a.swapaxes(1, 2))[:, 0, :]

    def score_tokens(
        self, hidden: mx.array, depth: mx.array, tokens: mx.array
    ) -> mx.array:
        projected = self.project(hidden, depth)
        token_weights = self.lora_b[tokens.astype(mx.int32)]
        return mx.sum(projected * token_weights, axis=1)


def greedy_metrics(
    logits: np.ndarray, rows: CalibrationRows, indices: np.ndarray
) -> dict[str, int | float]:
    predictions = np.argmax(logits, axis=1).astype(np.uint32)
    target = rows.target[indices]
    base = rows.draft[indices]
    matches = predictions == target
    base_matches = base == target
    return {
        "rows": int(len(indices)),
        "accuracy": float(np.mean(matches)) if len(indices) else 0.0,
        "base_accuracy": float(np.mean(base_matches)) if len(indices) else 0.0,
        "repaired": int(np.sum(matches & ~base_matches)),
        "broken": int(np.sum(~matches & base_matches)),
    }


def acceptance_metrics(
    logits: np.ndarray, rows: CalibrationRows, indices: np.ndarray
) -> dict[str, int | float]:
    predictions = np.argmax(logits, axis=1).astype(np.uint32)
    accepted = 0
    base_accepted = 0
    rounds = 0
    current_round: int | None = None
    corrected_live = True
    base_live = True
    for local_index, row_index in enumerate(indices):
        row_round = int(rows.round_id[row_index])
        if current_round != row_round:
            current_round = row_round
            corrected_live = True
            base_live = True
            rounds += 1
        corrected_match = predictions[local_index] == rows.target[row_index]
        base_match = rows.draft[row_index] == rows.target[row_index]
        if corrected_live and corrected_match:
            accepted += 1
        else:
            corrected_live = False
        if base_live and base_match:
            base_accepted += 1
        else:
            base_live = False
    return {
        "rounds": rounds,
        "accepted": accepted,
        "base_accepted": base_accepted,
        "accepted_delta": accepted - base_accepted,
        "tokens_per_round": (rounds + accepted) / rounds if rounds else 0.0,
        "base_tokens_per_round": (rounds + base_accepted) / rounds if rounds else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("lm_head_shard", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mismatch-weight", type=float, default=1.5)
    parser.add_argument("--loss", choices=("ce", "pairwise"), default="ce")
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()
    if args.rank < 1 or args.steps < 1 or args.batch_size < 1:
        parser.error("rank, steps and batch-size must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("train-fraction must be between zero and one")

    rows = load_rows(args.calibration)
    active_indices = np.flatnonzero(rows.active)
    split_groups = rows.request_id if np.unique(rows.request_id).size > 1 else rows.round_id
    round_positions = np.unique(split_groups[active_indices])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(round_positions)
    split = max(1, min(len(round_positions) - 1, round(args.train_fraction * len(round_positions))))
    train_positions = set(int(value) for value in round_positions[:split])
    train_indices = active_indices[
        np.array([int(split_groups[i]) in train_positions for i in active_indices])
    ]
    eval_indices = np.flatnonzero(
        np.array([int(value) not in train_positions for value in split_groups])
    )

    weights = mx.load(str(args.lm_head_shard))
    prefix = "language_model.lm_head"
    weight = weights[f"{prefix}.weight"]
    scales = weights[f"{prefix}.scales"]
    biases = weights[f"{prefix}.biases"]
    hidden = mx.array(rows.hidden)
    base_logits = mx.quantized_matmul(
        hidden, weight, scales, biases, transpose=True, group_size=64, bits=4
    )
    mx.eval(base_logits)

    depth_count = int(np.max(rows.depth))
    adapter = HeadAdapter(
        rows.hidden.shape[1], int(base_logits.shape[1]), args.rank, depth_count
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def loss_fn(model: HeadAdapter, batch: mx.array) -> mx.array:
        h = hidden[batch]
        depths = mx.array(rows.depth)[batch]
        labels = mx.array(rows.target)[batch].astype(mx.int32)
        drafts = mx.array(rows.draft)[batch].astype(mx.int32)
        matches = mx.array(rows.matched)[batch]
        if args.loss == "pairwise":
            row_base = base_logits[batch].astype(mx.float32)
            base_target = mx.take_along_axis(row_base, labels[:, None], axis=1).reshape(-1)
            base_draft = mx.take_along_axis(row_base, drafts[:, None], axis=1).reshape(-1)
            delta_target = model.score_tokens(h, depths, labels)
            delta_draft = model.score_tokens(h, depths, drafts)
            corrected_gap = base_target + delta_target - base_draft - delta_draft
            repair = mx.maximum(0.0, float(args.margin) - corrected_gap)
            preserve = mx.square(delta_target)
            return mx.mean(mx.where(matches, preserve, repair))
        logits = base_logits[batch].astype(mx.float32) + model(h, depths)
        selected = mx.take_along_axis(logits, labels[:, None], axis=1).reshape(-1)
        ce = mx.logsumexp(logits, axis=1) - selected
        row_weights = mx.where(
            matches, 1.0, float(args.mismatch_weight)
        )
        return mx.mean(ce * row_weights)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    for step in range(args.steps):
        batch_np = rng.choice(train_indices, size=min(args.batch_size, len(train_indices)), replace=True)
        loss, gradients = loss_and_grad(adapter, mx.array(batch_np.astype(np.int32)))
        optimizer.update(adapter, gradients)
        mx.eval(adapter.parameters(), optimizer.state, loss)

    base_eval = np.asarray(
        base_logits[mx.array(eval_indices.astype(np.int32))].astype(mx.float32)
    )
    adapter_eval = np.asarray(
        adapter(
            hidden[mx.array(eval_indices.astype(np.int32))],
            mx.array(rows.depth[eval_indices]),
        )
    )
    corrected_eval = base_eval + adapter_eval
    scale_sweep = {}
    for scale in (0.125, 0.25, 0.5, 0.75, 1.0):
        scale_sweep[str(scale)] = acceptance_metrics(
            base_eval + scale * adapter_eval, rows, eval_indices
        )
    depth_masks = {}
    for first_depth in (1, 2, 3):
        mask = (rows.depth[eval_indices] >= first_depth).astype(np.float32)[:, None]
        depth_masks[f"d{first_depth}plus"] = acceptance_metrics(
            base_eval + mask * adapter_eval, rows, eval_indices
        )
    report = {
        "schema_version": 1,
        "rank": args.rank,
        "steps": args.steps,
        "train_rows": int(len(train_indices)),
        "eval": greedy_metrics(corrected_eval, rows, eval_indices),
        "eval_base": greedy_metrics(base_eval, rows, eval_indices),
        "acceptance": acceptance_metrics(corrected_eval, rows, eval_indices),
        "scale_sweep": scale_sweep,
        "depth_masks": depth_masks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".safetensors":
        arrays = {"lora_b": adapter.lora_b.astype(mx.bfloat16)}
        for depth in range(depth_count):
            arrays[f"lora_a.depth{depth + 1}"] = adapter.lora_a[depth].astype(mx.bfloat16)
        mx.save_safetensors(
            str(args.output), arrays, {"metadata_json": json.dumps(report, sort_keys=True)}
        )
    else:
        np.savez(
            args.output,
            lora_a=np.asarray(adapter.lora_a, dtype=np.float32),
            lora_b=np.asarray(adapter.lora_b, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(report, sort_keys=True)),
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
