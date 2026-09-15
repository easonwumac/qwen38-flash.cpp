#!/usr/bin/env python3
"""Quantize a BF16 n-gram AoS table into row-addressable affine Q8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DIMENSION = 160
GROUP_SIZE = 32
GROUPS = DIMENSION // GROUP_SIZE
BF16_ROW_BYTES = DIMENSION * 2
Q8_ROW_BYTES = DIMENSION + GROUPS * 4


def float_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 using round-to-nearest-even."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def bf16_to_float(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype="<u2").astype(np.uint32) << 16
    return bits.view(np.float32)


def quantize_rows(source: np.ndarray) -> np.ndarray:
    """Return rows encoded as uint8 weights, BF16 scales, then BF16 biases."""
    if source.dtype != np.dtype("<u2") or source.ndim != 2 or source.shape[1] != DIMENSION:
        raise ValueError("expected a little-endian BF16 bit matrix with width 160")
    values = bf16_to_float(source).reshape(-1, GROUPS, GROUP_SIZE)
    minimum = values.min(axis=2)
    maximum = values.max(axis=2)
    scales = (maximum - minimum) / np.float32(255.0)
    scale_bits = float_to_bf16(scales)
    bias_bits = float_to_bf16(minimum)
    stored_scales = bf16_to_float(scale_bits)
    stored_biases = bf16_to_float(bias_bits)
    safe_scales = np.where(stored_scales == 0, np.float32(1.0), stored_scales)
    quantized = np.rint(
        (values - stored_biases[:, :, None]) / safe_scales[:, :, None]
    )
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    quantized = np.where(
        (stored_scales == 0)[:, :, None], np.uint8(0), quantized
    )
    encoded = np.empty((source.shape[0], Q8_ROW_BYTES), dtype=np.uint8)
    encoded[:, :DIMENSION] = quantized.reshape(-1, DIMENSION)
    encoded[:, DIMENSION : DIMENSION + GROUPS * 2] = (
        scale_bits.view(np.uint8).reshape(-1, GROUPS * 2)
    )
    encoded[:, DIMENSION + GROUPS * 2 :] = (
        bias_bits.view(np.uint8).reshape(-1, GROUPS * 2)
    )
    return encoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.chunk_rows <= 0:
        raise ValueError("chunk rows must be positive")
    source_bytes = args.input.stat().st_size
    if source_bytes % BF16_ROW_BYTES:
        raise RuntimeError("BF16 AoS size is not a whole number of rows")
    rows = source_bytes // BF16_ROW_BYTES
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"refusing to overwrite partial output {partial}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.input.open("rb") as source, partial.open("xb") as target:
        while processed < rows:
            count = min(args.chunk_rows, rows - processed)
            raw = source.read(count * BF16_ROW_BYTES)
            if len(raw) != count * BF16_ROW_BYTES:
                raise RuntimeError("truncated BF16 AoS input")
            matrix = np.frombuffer(raw, dtype="<u2").reshape(count, DIMENSION)
            target.write(quantize_rows(matrix).tobytes())
            processed += count
            print(json.dumps({"rows": processed, "total": rows}), flush=True)
        target.flush()
        os.fsync(target.fileno())
    expected = rows * Q8_ROW_BYTES
    if partial.stat().st_size != expected:
        raise RuntimeError("Q8 AoS size does not match tensor geometry")
    partial.rename(args.output)
    print(json.dumps({"rows": rows, "bytes": expected, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
