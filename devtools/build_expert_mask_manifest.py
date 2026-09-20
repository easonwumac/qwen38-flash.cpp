#!/usr/bin/env python3
"""Build a lightweight checkpoint view with per-layer routed-expert masks.

The output references the base checkpoint's shards by relative path. It does
not copy, link, reorder, or requantize model weights. A layer omitted from the
mask retains every expert.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


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


def parse_layers(value: str) -> list[int]:
    try:
        return sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from error


def kept_experts(source: dict, layer: int, expert_count: int) -> list[int]:
    if "layer_results" in source:
        description = source["layer_results"].get(str(layer))
        if description is None:
            raise KeyError(f"ranking map is missing layer {layer}")
        pruned = {int(expert) for expert in description["pruned"]}
        return [expert for expert in range(expert_count) if expert not in pruned]
    values = source.get(str(layer))
    if values is None:
        raise KeyError(f"ranking map is missing layer {layer}")
    return sorted({int(expert) for expert in values})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ranking-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-layers",
        type=parse_layers,
        default=parse_layers("0,1,31,35,36,39"),
        help="layers that retain all experts",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    ranking_map = args.ranking_map.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    config = load_json(base / "config.json")
    index = load_json(base / "model.safetensors.index.json")
    ranking = load_json(ranking_map)
    text = config["text_config"]
    layer_count = int(text["num_hidden_layers"])
    expert_count = int(text["num_experts"])
    topk = int(text["num_experts_per_tok"])
    full_layers = set(args.full_layers)
    if any(layer < 0 or layer >= layer_count for layer in full_layers):
        raise ValueError(f"full layers must be within 0..{layer_count - 1}")

    masks: dict[str, list[int]] = {}
    expected_keep: int | None = None
    for layer in range(layer_count):
        if layer in full_layers:
            continue
        kept = kept_experts(ranking, layer, expert_count)
        if len(kept) < topk or len(kept) >= expert_count:
            raise ValueError(f"layer {layer} retains invalid expert count {len(kept)}")
        if any(expert < 0 or expert >= expert_count for expert in kept):
            raise ValueError(f"layer {layer} contains an invalid expert index")
        if expected_keep is None:
            expected_keep = len(kept)
        elif len(kept) != expected_keep:
            raise ValueError("ranking map must retain the same count on masked layers")
        masks[str(layer)] = kept

    output.mkdir(parents=True)
    for filename in COPIED_FILES:
        source = base / filename
        if source.is_file():
            shutil.copy2(source, output / filename)

    masked_config = dict(config)
    masked_config["qwen38_external_shards"] = True
    masked_config["qwen38_expert_keep"] = masks
    save_json(output / "config.json", masked_config)
    save_json(
        output / "model.safetensors.index.json",
        {
            "metadata": dict(index.get("metadata", {})),
            "weight_map": {
                name: os.path.relpath(base / shard, output)
                for name, shard in index["weight_map"].items()
            },
        },
    )
    save_json(
        output / "expert-mask.json",
        {
            "format": 1,
            "base": os.path.relpath(base, output),
            "ranking_map": os.path.relpath(ranking_map, output),
            "full_layers": sorted(full_layers),
            "masked_layers": layer_count - len(full_layers),
            "retained_experts": expected_keep,
            "topk": topk,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "full_layers": sorted(full_layers),
                "masked_layers": layer_count - len(full_layers),
                "retained_experts": expected_keep,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
