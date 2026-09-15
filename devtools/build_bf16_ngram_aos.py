#!/usr/bin/env python3
"""Repack Qwen BF16 n-gram shards into row-addressable SSD storage."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO


KEY_PREFIX = (
    "model.language_model.layers.1.ple.ple_embedding."
    "ngram_embedding.shard_"
)


def tensor_header(source: BinaryIO) -> tuple[int, dict]:
    length_bytes = source.read(8)
    if len(length_bytes) != 8:
        raise RuntimeError("truncated safetensors header length")
    length = struct.unpack("<Q", length_bytes)[0]
    header = json.loads(source.read(length))
    return 8 + length, header


def copy_range(source: BinaryIO, target: BinaryIO, offset: int, size: int) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(8 << 20, remaining))
        if not chunk:
            raise RuntimeError("truncated safetensors payload")
        target.write(chunk)
        remaining -= len(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    weight_map = json.loads(args.index.read_text())["weight_map"]
    keys = [key for key in weight_map if key.startswith(KEY_PREFIX) and key.endswith(".weight")]
    keys.sort(key=lambda key: int(key[len(KEY_PREFIX) : -len(".weight")]))
    if [int(key[len(KEY_PREFIX) : -len(".weight")]) for key in keys] != list(range(128)):
        raise RuntimeError("expected exactly the contiguous BF16 n-gram shards 0..127")

    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"refusing to overwrite partial output {partial}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    try:
        with partial.open("xb") as target:
            for index, key in enumerate(keys):
                path = args.shard_dir / weight_map[key]
                with path.open("rb") as source:
                    data_start, header = tensor_header(source)
                    metadata = header[key]
                    if metadata["dtype"] != "BF16" or len(metadata["shape"]) != 2 or metadata["shape"][1] != 160:
                        raise RuntimeError(f"unexpected BF16 n-gram tensor geometry for {key}")
                    begin, end = metadata["data_offsets"]
                    copy_range(source, target, data_start + begin, end - begin)
                    total_rows += int(metadata["shape"][0])
                print(json.dumps({"shard": index, "rows": total_rows}), flush=True)
            target.flush()
            os.fsync(target.fileno())
        expected = total_rows * 160 * 2
        if partial.stat().st_size != expected:
            raise RuntimeError("BF16 AoS size does not match tensor geometry")
        partial.rename(args.output)
    except Exception:
        # Preserve a partial file for diagnosis; a later run refuses to clobber it.
        raise
    print(json.dumps({"rows": total_rows, "bytes": expected, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
