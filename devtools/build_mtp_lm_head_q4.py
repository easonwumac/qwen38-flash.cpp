#!/usr/bin/env python3
"""Derive an optional Q4-only MTP language head from a model's Q8 head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    try:
        import mlx.core as mx
    except ModuleNotFoundError as error:
        raise RuntimeError("this converter requires a Python environment with MLX") from error

    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    config = json.loads((args.model / "config.json").read_text())
    weight_map = index["weight_map"]
    prefix = (
        "language_model.lm_head"
        if "language_model.lm_head.weight" in weight_map
        else "lm_head"
    )
    quantization = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quantization, dict):
        raise RuntimeError("model config has no affine quantization metadata")
    quantization_key = prefix.removeprefix("language_model.")
    head_quantization = quantization.get(quantization_key, quantization)
    if (
        head_quantization.get("bits") != 8
        or head_quantization.get("group_size") != 64
    ):
        raise RuntimeError("source LM head must be affine Q8/group-64")
    names = [f"{prefix}.{suffix}" for suffix in ("weight", "scales", "biases")]
    shards = {weight_map[name] for name in names}
    if len(shards) != 1:
        raise RuntimeError("LM-head tensors must reside in one shard")
    shard = args.model / shards.pop()
    source_sha256 = sha256(shard)
    tensors = mx.load(str(shard))
    dense = mx.dequantize(
        tensors[names[0]], tensors[names[1]], tensors[names[2]],
        group_size=64, bits=8, mode="affine",
    )
    weight, scales, biases = mx.quantize(
        dense, group_size=64, bits=4, mode="affine"
    )
    mx.eval(weight, scales, biases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=args.output.name + ".", suffix=".tmp.safetensors", dir=args.output.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        mx.save_safetensors(
            str(temporary),
            {
                "mtp_lm_head.weight": weight,
                "mtp_lm_head.scales": scales,
                "mtp_lm_head.biases": biases,
            },
            {
                "source_shard": shard.name,
                "source_sha256": source_sha256,
                "source_bits": "8",
                "bits": "4",
                "group_size": "64",
            },
        )
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "source": str(shard),
        "source_sha256": source_sha256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
