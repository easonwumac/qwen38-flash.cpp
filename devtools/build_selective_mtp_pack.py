#!/usr/bin/env python3
"""Build a mixed-precision MTP sidecar from two shape-compatible packs."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GROUPS = {
    "fc": (
        "language_model.mtp.fc_embedding",
        "language_model.mtp.fc_hidden",
    ),
    "attention": (
        "language_model.mtp.layers.0.self_attn.q_proj",
        "language_model.mtp.layers.0.self_attn.k_proj",
        "language_model.mtp.layers.0.self_attn.v_proj",
        "language_model.mtp.layers.0.self_attn.o_proj",
    ),
    "hyper_connection": (
        "language_model.mtp.hyper_connection_mixer.input_mix_weight_down",
        "language_model.mtp.hyper_connection_mixer.input_mix_weight_up",
        "language_model.mtp.layers.0.attn_hyper_connection.input_mix_weight_down",
        "language_model.mtp.layers.0.attn_hyper_connection.input_mix_weight_up",
        "language_model.mtp.layers.0.mlp_hyper_connection.input_mix_weight_down",
        "language_model.mtp.layers.0.mlp_hyper_connection.input_mix_weight_up",
    ),
    "shared_expert": (
        "language_model.mtp.layers.0.mlp.shared_expert.gate_proj",
        "language_model.mtp.layers.0.mlp.shared_expert.up_proj",
        "language_model.mtp.layers.0.mlp.shared_expert.down_proj",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("q4_source", type=Path)
    parser.add_argument("q8_source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--q8-projections",
        nargs="+",
        choices=PROJECTIONS,
        default=[],
        help="routed-expert projections copied exactly from the Q8 source",
    )
    parser.add_argument(
        "--q8-groups",
        nargs="+",
        choices=sorted(GROUPS),
        default=[],
        help="non-routed projection groups copied exactly from the Q8 source",
    )
    args = parser.parse_args()

    try:
        import mlx.core as mx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "build_selective_mtp_pack.py requires a Python environment with MLX"
        ) from error

    q4 = mx.load(str(args.q4_source))
    q8 = mx.load(str(args.q8_source))
    if set(q4) != set(q8):
        missing_q4 = sorted(set(q8) - set(q4))
        missing_q8 = sorted(set(q4) - set(q8))
        raise RuntimeError(
            f"sidecar key mismatch: missing_q4={missing_q4} missing_q8={missing_q8}"
        )

    bases = [
        f"language_model.mtp.layers.0.mlp.switch_mlp.{projection}"
        for projection in args.q8_projections
    ]
    for group in args.q8_groups:
        bases.extend(GROUPS[group])
    if not bases:
        parser.error("at least one --q8-projections or --q8-groups value is required")
    if args.q8_projections and set(args.q8_projections) != set(PROJECTIONS):
        print(
            "warning: selecting only part of routed gate/up/down creates mixed "
            "bit widths; the current runtime will use its slower generic MoE path"
        )

    replaced: list[str] = []
    for base in bases:
        for suffix in ("weight", "scales", "biases"):
            key = f"{base}.{suffix}"
            if key not in q4:
                raise KeyError(f"missing MTP tensor: {key}")
            q4[key] = q8[key]
            replaced.append(key)

    mx.eval(*[q4[key] for key in replaced])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), q4)
    selected = [*args.q8_groups, *args.q8_projections]
    print(f"replaced {len(replaced)} tensors: {', '.join(selected)}")
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
