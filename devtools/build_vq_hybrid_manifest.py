#!/usr/bin/env python3
"""Build a lightweight VQ checkpoint view from two compatible checkpoints.

The output contains tokenizer/configuration files and an index whose relative
shard paths point at the original checkpoint directories.  No model shard is
copied or linked.  Selected routed-MoE layers use the overlay checkpoint; every
other tensor continues to use the base checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TENSOR_SUFFIXES = ("codes", "codebook", "vq_scales")
COPIED_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "model.py",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def tensor_names(layer: int, projections: tuple[str, ...]) -> list[str]:
    prefix = f"model.layers.{layer}.mlp.switch_mlp"
    return [
        f"{prefix}.{projection}.{suffix}"
        for projection in projections
        for suffix in TENSOR_SUFFIXES
    ]


def vq_module_names(layer: int, projections: tuple[str, ...]) -> list[str]:
    prefix = f"model.layers.{layer}.mlp.switch_mlp"
    return [f"{prefix}.{projection}" for projection in projections]


def tensor_size(directory: Path, index: dict, name: str) -> int:
    shard = directory / index["weight_map"][name]
    with shard.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors length prefix: {shard}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(handle.read(header_size))
    begin, end = header[name]["data_offsets"]
    return int(end) - int(begin)


def relative_shard(output: Path, source: Path, shard: str) -> str:
    return os.path.relpath(source / shard, output)


def parse_layers(value: str) -> list[int]:
    try:
        layers = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("at least one non-negative layer is required")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument(
        "--projection-set",
        choices=("all", "gate-up", "down"),
        default="all",
        help="overlay all routed projections, gate+up together, or down only",
    )
    parser.add_argument(
        "--stream-cache-mib",
        type=int,
        default=0,
        help="stream overlay layers through this bounded expert cache",
    )
    parser.add_argument(
        "--stream-layers",
        type=parse_layers,
        help="overlay-layer subset to stream (defaults to every overlay layer)",
    )
    args = parser.parse_args()
    projections = {
        "all": PROJECTIONS,
        "gate-up": ("gate_proj", "up_proj"),
        "down": ("down_proj",),
    }[args.projection_set]

    base = args.base.resolve()
    overlay = args.overlay.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    base_config = load_json(base / "config.json")
    overlay_config = load_json(overlay / "config.json")
    base_index = load_json(base / "model.safetensors.index.json")
    overlay_index = load_json(overlay / "model.safetensors.index.json")
    base_text = base_config.get("text_config", {})
    overlay_text = overlay_config.get("text_config", {})
    compatibility_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "vocab_size",
    )
    for field in compatibility_fields:
        if base_text.get(field) != overlay_text.get(field):
            raise ValueError(f"incompatible text_config.{field}")

    layer_count = int(base_text["num_hidden_layers"])
    if any(layer >= layer_count for layer in args.layers):
        raise ValueError(f"layer must be below {layer_count}")

    output.mkdir(parents=True)
    for filename in COPIED_FILES:
        source = base / filename
        if source.is_file():
            shutil.copy2(source, output / filename)

    hybrid_config = dict(base_config)
    hybrid_config["qwen38_external_shards"] = True
    if args.stream_cache_mib < 0:
        raise ValueError("stream cache must not be negative")
    stream_layers = args.stream_layers or args.layers
    if args.stream_layers and not set(stream_layers).issubset(args.layers):
        raise ValueError("stream layers must be a subset of overlay layers")
    if args.stream_layers and not args.stream_cache_mib:
        raise ValueError("stream layers require a non-zero stream cache")
    if args.stream_cache_mib:
        hybrid_config["qwen38_streaming"] = {
            "layers": stream_layers,
            "expert_cache_bytes": args.stream_cache_mib * 1024 * 1024,
        }
    hybrid_modules = dict(base_config.get("vq_modules", {}))
    for layer in args.layers:
        for module in vq_module_names(layer, projections):
            if module not in overlay_config.get("vq_modules", {}):
                raise KeyError(f"overlay is missing VQ module: {module}")
            hybrid_modules[module] = overlay_config["vq_modules"][module]
    hybrid_config["vq_modules"] = hybrid_modules

    weight_map = {
        name: relative_shard(output, base, shard)
        for name, shard in base_index["weight_map"].items()
    }
    base_size = int(base_index.get("metadata", {}).get("total_size", 0))
    total_size = base_size
    for layer in args.layers:
        for name in tensor_names(layer, projections):
            if name not in base_index["weight_map"]:
                raise KeyError(f"base is missing tensor: {name}")
            if name not in overlay_index["weight_map"]:
                raise KeyError(f"overlay is missing tensor: {name}")
            weight_map[name] = relative_shard(
                output, overlay, overlay_index["weight_map"][name]
            )
            if base_size:
                total_size -= tensor_size(base, base_index, name)
                total_size += tensor_size(overlay, overlay_index, name)

    save_json(output / "config.json", hybrid_config)
    metadata = dict(base_index.get("metadata", {}))
    if base_size:
        metadata["total_size"] = total_size
    save_json(
        output / "model.safetensors.index.json",
        {"metadata": metadata, "weight_map": weight_map},
    )
    save_json(
        output / "hybrid.json",
        {
            "format": 1,
            "base": os.path.relpath(base, output),
            "overlay": os.path.relpath(overlay, output),
            "overlay_layers": args.layers,
            "overlay_projections": list(projections),
            "stream_layers": stream_layers if args.stream_cache_mib else [],
            "declared_weight_bytes": total_size if base_size else None,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "layers": args.layers,
                "projections": list(projections),
                "declared_weight_bytes": total_size if base_size else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
