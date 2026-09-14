#!/usr/bin/env python3
"""Measure whether Niwaki BF16 healing maps admit a cheap approximation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
    return 8 + header_size, header


def bf16_tensor(path: Path, payload: int, record: dict[str, object]) -> np.ndarray:
    if record["dtype"] != "BF16":
        raise ValueError(f"expected BF16 map, got {record['dtype']}")
    begin, end = record["data_offsets"]
    shape = tuple(record["shape"])
    raw = np.memmap(path, dtype="<u2", mode="r", offset=payload + begin,
                    shape=((end - begin) // 2,))
    bits = np.asarray(raw, dtype=np.uint32) << 16
    return bits.view(np.float32).reshape(shape)


def low_rank_capture(delta: np.ndarray, rank: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    width = min(delta.shape[1], rank + 16)
    omega = rng.standard_normal((delta.shape[1], width), dtype=np.float32)
    basis, _ = np.linalg.qr(delta @ omega, mode="reduced")
    small = basis.T @ delta
    singular = np.linalg.svd(small, compute_uv=False)
    captured = float(np.square(singular[:rank], dtype=np.float64).sum())
    total = float(np.square(delta, dtype=np.float64).sum())
    return captured / total if total else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--layers", default="0,12,24,36,47")
    parser.add_argument("--ranks", default="16,32,64,128")
    args = parser.parse_args()

    index = json.loads((args.model_directory / "model.safetensors.index.json").read_text())
    wanted_layers = [int(value) for value in args.layers.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    opened: dict[Path, tuple[int, dict[str, object]]] = {}
    results = []
    for layer in wanted_layers:
        name = f"language_model.model.layers.{layer}.mlp.T"
        shard = args.model_directory / index["weight_map"][name]
        payload, header = opened.setdefault(shard, load_header(shard))
        matrix = bf16_tensor(shard, payload, header[name])
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} is not square: {matrix.shape}")
        delta = matrix.copy()
        diagonal_index = np.diag_indices(matrix.shape[0])
        delta[diagonal_index] -= 1.0
        matrix_norm = float(np.linalg.norm(matrix))
        delta_norm = float(np.linalg.norm(delta))
        diagonal = matrix[diagonal_index]
        results.append({
            "layer": layer,
            "shape": list(matrix.shape),
            "delta_over_map_norm": delta_norm / matrix_norm,
            "diagonal_mean": float(diagonal.mean()),
            "diagonal_std": float(diagonal.std()),
            "delta_max_abs": float(np.abs(delta).max()),
            "low_rank_delta_capture": {
                str(rank): low_rank_capture(delta, rank, layer) for rank in ranks
            },
        })
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
