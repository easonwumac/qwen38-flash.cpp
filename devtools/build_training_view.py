#!/usr/bin/env python3
"""Build a zero-copy MLX training view without runtime-only tensor sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


RUNTIME_ONLY_PREFIXES = ("model-qmeta-",)


def build_view(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    linked: list[str] = []
    excluded: list[str] = []
    mtp_sidecars = sorted(source.glob("model-mtp-*.safetensors"))
    if len(mtp_sidecars) > 1:
        raise ValueError("source contains multiple model-mtp sidecars")
    for item in sorted(source.iterdir(), key=lambda path: path.name):
        if item.name.startswith(RUNTIME_ONLY_PREFIXES):
            excluded.append(item.name)
            continue
        if item in mtp_sidecars:
            excluded.append(item.name)
            continue
        if item.is_dir():
            continue
        destination = output / item.name
        destination.symlink_to(os.path.relpath(item, output))
        linked.append(item.name)

    if mtp_sidecars:
        destination = output / "mtp.safetensors"
        destination.symlink_to(os.path.relpath(mtp_sidecars[0], output))
        linked.append(destination.name)

    required = {"config.json", "model.safetensors.index.json"}
    missing = sorted(required.difference(linked))
    if missing:
        raise ValueError(f"training view is missing required files: {', '.join(missing)}")
    if not any(name.endswith(".safetensors") for name in linked):
        raise ValueError("training view contains no safetensors")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": str(source),
        "linked_count": len(linked),
        "excluded": excluded,
        "mtp_sidecar": mtp_sidecars[0].name if mtp_sidecars else None,
    }
    (output / "training-view.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_view(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
