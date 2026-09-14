#!/usr/bin/env python3
"""Build and install BF16 low-rank factors for Niwaki's T = I + delta maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct

import numpy as np

from niwaki_map_analysis import bf16_tensor, load_header


def as_bf16_bytes(values: np.ndarray) -> bytes:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.asarray(rounded >> 16, dtype="<u2").tobytes(order="C")


def factor(delta: np.ndarray, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    width = min(delta.shape[1], rank + 16)
    omega = rng.standard_normal((delta.shape[1], width), dtype=np.float32)
    basis, _ = np.linalg.qr(delta @ omega, mode="reduced")
    small = basis.T @ delta
    left_small, singular, right = np.linalg.svd(small, full_matrices=False)
    left = (basis @ left_small[:, :rank]) * singular[:rank]
    return left.astype(np.float32), right[:rank].astype(np.float32)


def write_safetensors(path: Path, tensors: list[tuple[str, np.ndarray]]) -> int:
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, values in tensors:
        payload = as_bf16_bytes(values)
        header[name] = {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for payload in payloads:
            handle.write(payload)
    os.replace(temporary, path)
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()
    if args.rank < 16 or args.rank > 1024 or args.rank % 16:
        raise ValueError("rank must be a multiple of 16 between 16 and 1024")

    index_path = args.model_directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    opened: dict[Path, tuple[int, dict[str, object]]] = {}
    tensors: list[tuple[str, np.ndarray]] = []
    for layer in range(48):
        base = f"language_model.model.layers.{layer}.mlp.T"
        shard = args.model_directory / weight_map[base]
        payload, header = opened.setdefault(shard, load_header(shard))
        matrix = bf16_tensor(shard, payload, header[base])
        delta = matrix.copy()
        delta[np.diag_indices(delta.shape[0])] -= 1.0
        left, right = factor(delta, args.rank, layer)
        tensors.append((base + "_delta_left", left))
        tensors.append((base + "_delta_right", right))

    sidecar_name = f"niwaki-maps-delta-r{args.rank}.safetensors"
    sidecar_path = args.model_directory / sidecar_name
    sidecar_size = write_safetensors(sidecar_path, tensors)
    for name, _ in tensors:
        weight_map[name] = sidecar_name
    metadata = index.setdefault("metadata", {})
    original_total = sum(
        path.stat().st_size for path in args.model_directory.glob("*.safetensors")
        if not path.name.startswith("niwaki-maps-delta-r")
    )
    metadata["total_size"] = original_total + sidecar_size
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_index, index_path)
    print(json.dumps({"sidecar": str(sidecar_path), "rank": args.rank,
                      "bytes": sidecar_size, "tensors": len(tensors)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
