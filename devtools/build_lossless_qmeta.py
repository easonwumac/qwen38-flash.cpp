#!/usr/bin/env python3
"""Build an exact compact affine-metadata sidecar for Qwen3.8 routed experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BITS = 13
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    offset: int
    shape: tuple[int, ...]
    dtype: str


def parse_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise RuntimeError(f"invalid safetensors length prefix: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(stream.read(header_size))
    return 8 + header_size, header


def tensor_locations(model: Path) -> dict[str, TensorLocation]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    result: dict[str, TensorLocation] = {}
    for shard, names in by_shard.items():
        path = model / shard
        data_offset, header = parse_header(path)
        for name in names:
            description = header[name]
            begin, _ = description["data_offsets"]
            result[name] = TensorLocation(
                path=path,
                offset=data_offset + begin,
                shape=tuple(description["shape"]),
                dtype=description["dtype"],
            )
    return result


def read_bf16(location: TensorLocation) -> np.ndarray:
    if location.dtype != "BF16":
        raise RuntimeError(f"expected BF16 metadata in {location.path}")
    count = int(np.prod(location.shape, dtype=np.int64))
    with location.path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            values = np.frombuffer(
                mapped, dtype="<u2", count=count, offset=location.offset
            ).copy()
        finally:
            mapped.close()
    return values.reshape(location.shape)


def pair_bank(
    locations: dict[str, TensorLocation], base: str
) -> tuple[np.ndarray, tuple[int, int, int]]:
    scales = read_bf16(locations[base + ".scales"])
    biases = read_bf16(locations[base + ".biases"])
    if scales.shape != biases.shape or len(scales.shape) != 3:
        raise RuntimeError(f"invalid affine metadata shape for {base}")
    pairs = scales.astype(np.uint32)
    pairs |= biases.astype(np.uint32) << np.uint32(16)
    return pairs, scales.shape


def pack_indices(indices: np.ndarray, groups: int, bits: int = BITS) -> np.ndarray:
    rows = indices.size // groups
    row_bytes = (groups * bits + 7) // 8
    tags = np.zeros((rows, row_bytes), dtype=np.uint8)
    shaped = indices.reshape(rows, groups).astype(np.uint32, copy=False)
    mask = np.uint32((1 << bits) - 1)
    for group in range(groups):
        values = shaped[:, group] & mask
        bit = group * bits
        byte = bit // 8
        shift = bit & 7
        tags[:, byte] |= (values << np.uint32(shift)).astype(np.uint8)
        tags[:, byte + 1] |= (values >> np.uint32(8 - shift)).astype(np.uint8)
        if shift + bits > 16:
            tags[:, byte + 2] |= (values >> np.uint32(16 - shift)).astype(np.uint8)
    return tags


def unpack_indices(tags: np.ndarray, groups: int, bits: int = BITS) -> np.ndarray:
    if tags.ndim != 2 or tags.shape[1] != (groups * bits + 7) // 8:
        raise RuntimeError("invalid packed qmeta geometry")
    indices = np.empty((tags.shape[0], groups), dtype=np.uint32)
    mask = np.uint32((1 << bits) - 1)
    for group in range(groups):
        bit = group * bits
        byte = bit // 8
        shift = bit & 7
        window = tags[:, byte].astype(np.uint32)
        window |= tags[:, byte + 1].astype(np.uint32) << np.uint32(8)
        if shift + bits > 16:
            window |= tags[:, byte + 2].astype(np.uint32) << np.uint32(16)
        indices[:, group] = (window >> np.uint32(shift)) & mask
    return indices


def banks(layer_count: int) -> list[tuple[str, str]]:
    result = []
    for layer in range(layer_count):
        for projection in PROJECTIONS:
            base = (
                f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}"
            )
            result.append((base, projection))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--bits", choices=(13, 16), default=BITS, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    bits = args.bits
    dictionary_limit = 1 << bits
    model = args.model.resolve()
    output = args.output or Path(f"model-qmeta-lossless{bits}.safetensors")
    if not output.is_absolute():
        output = model / output
    if output.exists() and not args.overwrite:
        raise RuntimeError(f"output already exists: {output}")

    config = json.loads((model / "config.json").read_text())
    layer_count = int(config["text_config"]["num_hidden_layers"])
    locations = tensor_locations(model)
    bank_list = banks(layer_count)

    descriptions: list[tuple[str, tuple[int, ...], int]] = []
    unique_counts: dict[str, int] = {}
    cursor = 0
    for number, (base, _) in enumerate(bank_list, 1):
        pairs, shape = pair_bank(locations, base)
        unique_count = int(np.unique(pairs).size)
        if unique_count > dictionary_limit:
            raise RuntimeError(
                f"{base} needs {unique_count} entries, above {dictionary_limit}"
            )
        unique_counts[base] = unique_count
        rows = shape[0] * shape[1]
        row_bytes = (shape[2] * bits + 7) // 8
        tags_name = base + f".qmeta{bits}_tags"
        dictionary_name = base + f".qmeta{bits}_dict"
        tags_bytes = rows * row_bytes
        dictionary_bytes = unique_count * 4
        descriptions.append((tags_name, (shape[0], shape[1], row_bytes), tags_bytes))
        descriptions.append((dictionary_name, (unique_count,), dictionary_bytes))
        cursor += tags_bytes + dictionary_bytes
        print(
            f"scan {number:03d}/{len(bank_list)} unique={unique_count:4d} "
            f"payload={cursor / (1 << 30):.3f} GiB",
            file=sys.stderr,
        )

    header: dict[str, object] = {
        "__metadata__": {
            "schema": f"qwen38-lossless-qmeta{bits}-v1",
            "bits": str(bits),
            "source": "exact BF16 affine scale/bias pairs",
        }
    }
    offset = 0
    for name, shape, size in descriptions:
        dtype = "U8" if name.endswith("_tags") else "U32"
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    prefix = struct.pack("<Q", len(header_bytes))

    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    with partial.open("wb") as stream:
        stream.write(prefix)
        stream.write(header_bytes)
        digest.update(prefix)
        digest.update(header_bytes)
        for number, (base, _) in enumerate(bank_list, 1):
            pairs, shape = pair_bank(locations, base)
            dictionary, inverse = np.unique(pairs, return_inverse=True)
            if dictionary.size != unique_counts[base]:
                raise RuntimeError(f"non-deterministic dictionary size for {base}")
            if not np.array_equal(dictionary[inverse].reshape(-1), pairs.reshape(-1)):
                raise RuntimeError(f"lossless round-trip failed for {base}")
            tags = pack_indices(inverse, shape[2], bits)
            packed_indices = unpack_indices(tags, shape[2], bits)
            if not np.array_equal(
                dictionary[packed_indices].reshape(-1), pairs.reshape(-1)
            ):
                raise RuntimeError(f"packed lossless round-trip failed for {base}")
            dictionary = dictionary.astype("<u4", copy=False)
            for payload in (tags, dictionary):
                data = memoryview(payload).cast("B")
                stream.write(data)
                digest.update(data)
            print(
                f"write {number:03d}/{len(bank_list)} "
                f"{stream.tell() / (1 << 30):.3f} GiB",
                file=sys.stderr,
            )
        stream.flush()
        os.fsync(stream.fileno())
    if partial.stat().st_size != 8 + len(header_bytes) + offset:
        raise RuntimeError("lossless qmeta output size mismatch")
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": digest.hexdigest(),
                "bits": bits,
                "banks": len(bank_list),
                "max_unique_pairs": max(unique_counts.values()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
