#!/usr/bin/env python3
"""Derive an APFS-cloned model pack with one tensor namespace removed.

This is intentionally strict: it never overwrites a destination, never falls
back to a physical copy for regular files, and publishes the destination only
after the filtered index and all retained files are complete.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
from pathlib import Path


def clone_file(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    clonefile = libc.clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(source), os.fsencode(destination), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"APFS clonefile failed for {source}")
    shutil.copystat(source, destination, follow_symlinks=False)


def safetensors_payload_bytes(path: Path) -> int:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors header: {path}")
        header_size = int.from_bytes(raw_size, "little")
        header = json.loads(handle.read(header_size))
    entries = [value for name, value in header.items() if name != "__metadata__"]
    if not entries:
        raise ValueError(f"empty safetensors shard: {path}")
    return max(int(value["data_offsets"][1]) for value in entries)


def derive(source: Path, destination: Path, prefix: str, skipped: set[str]) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model index has no weight_map")

    removed = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    if not removed:
        raise ValueError(f"no tensors start with {prefix!r}")
    retained = {name: shard for name, shard in weight_map.items() if name not in removed}
    removed_shards = set(removed.values())
    shared = removed_shards.intersection(retained.values())
    if shared:
        raise ValueError(f"removed namespace shares shards with retained tensors: {sorted(shared)}")

    payload = sum(safetensors_payload_bytes(source / shard) for shard in removed_shards)
    metadata = index.setdefault("metadata", {})
    total_size = metadata.get("total_size")
    if not isinstance(total_size, int) or total_size < payload:
        raise ValueError("invalid metadata.total_size")
    metadata["total_size"] = total_size - payload
    index["weight_map"] = retained

    stage = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(f"staging path already exists: {stage}")
    stage.mkdir(mode=0o700)
    try:
        for item in sorted(source.iterdir()):
            if item.name == index_path.name or item.name in removed_shards or item.name in skipped:
                continue
            target = stage / item.name
            if item.is_symlink():
                target.symlink_to(os.readlink(item))
            elif item.is_file():
                clone_file(item, target)
            else:
                raise ValueError(f"unsupported model-pack entry: {item}")
        temporary_index = stage / (index_path.name + ".tmp")
        temporary_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_index, stage / index_path.name)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "source": str(source),
        "destination": str(destination),
        "removed_prefix": prefix,
        "removed_tensors": len(removed),
        "retained_tensors": len(retained),
        "removed_shards": sorted(removed_shards),
        "removed_payload_bytes": payload,
        "skipped_files": sorted(skipped),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prefix", default="language_model.mtp.")
    parser.add_argument("--skip", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.destination, args.prefix, set(args.skip)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
