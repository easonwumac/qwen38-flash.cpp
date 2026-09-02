#!/usr/bin/env python3
"""Requantize routed-expert tensors in one safetensors shard.

Intended for bounded one-layer feasibility probes. Other tensors are copied
unchanged, while complete switch_mlp projections present in the shard are
dequantized from their inferred affine width and requantized to the requested
width. Production quality work should start from BF16 rather than requantizing
an already lossy checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx


def infer_bits(weight: mx.array, scales: mx.array, group_size: int) -> int:
    return weight.shape[-1] * 32 // (scales.shape[-1] * group_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    tensors = mx.load(str(args.source))
    bases = sorted(
        key.removesuffix(".weight")
        for key in tensors
        if ".mlp.switch_mlp." in key and key.endswith(".weight")
    )
    if not bases:
        raise SystemExit("shard contains no routed-expert projection")

    for base in bases:
        weight = tensors[base + ".weight"]
        scales = tensors[base + ".scales"]
        biases = tensors[base + ".biases"]
        source_bits = infer_bits(weight, scales, args.group_size)
        dense = mx.dequantize(
            weight, scales, biases, group_size=args.group_size, bits=source_bits
        )
        packed, new_scales, new_biases = mx.quantize(
            dense, group_size=args.group_size, bits=args.bits
        )
        mx.eval(packed, new_scales, new_biases)
        tensors[base + ".weight"] = packed
        tensors[base + ".scales"] = new_scales
        tensors[base + ".biases"] = new_biases
        del dense
        mx.clear_cache()
        print(f"{base}: Q{source_bits} -> Q{args.bits}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), tensors)
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
