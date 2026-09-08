#!/usr/bin/env python3
"""Bounded BF16-sourced projection-selective quantization feasibility probe.

Reads only 16 distributed experts, never loads a full model. Synthetic inputs
and fixed routes are NOT natural activations or a calibrated quantization
recipe. Timings compare generic MLX kernels, not the production fused Q4 path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time

import mlx.core as mx
import numpy as np


def read_experts(path: Path, suffix: str, ids: list[int], digest) -> mx.array:
    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("unexpected safetensors header size")
        header = json.loads(file.read(size))
        names = [name for name in header if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError("expected exactly one projection")
        entry = header[names[0]]
        shape = entry["shape"]
        expected = [288, 1280, 2560] if suffix.endswith("gate_up_proj") else [288, 2560, 640]
        if entry["dtype"] != "BF16" or shape != expected:
            raise ValueError("requires REAP288 BF16 expert geometry")
        row_bytes = math.prod(shape[1:]) * 2
        if entry["data_offsets"][1] - entry["data_offsets"][0] != row_bytes * 288:
            raise ValueError("invalid tensor byte span")
        arrays = []
        for expert in ids:
            file.seek(8 + size + entry["data_offsets"][0] + expert * row_bytes)
            data = file.read(row_bytes)
            if len(data) != row_bytes:
                raise ValueError("truncated tensor")
            digest.update(data)
            value = mx.array(np.frombuffer(data, dtype=np.uint16).copy()).view(mx.bfloat16)
            arrays.append(value.reshape(shape[1:]))
        result = mx.stack(arrays)
        mx.eval(result)
        return result


def metrics(value: mx.array, reference: mx.array) -> dict[str, float]:
    a, b = value.astype(mx.float32), reference.astype(mx.float32)
    cosine = mx.sum(a * b) / mx.maximum(mx.sqrt(mx.sum(a*a) * mx.sum(b*b)), 1e-30)
    relative = mx.sqrt(mx.sum((a-b)**2) / mx.maximum(mx.sum(b*b), 1e-30))
    mx.eval(cosine, relative)
    return {"cosine": float(cosine.item()), "relative_l2": float(relative.item())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--rows", type=int, choices=[1, 5, 512], default=1)
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--seed", type=int, default=3801)
    args = parser.parse_args()
    if not os.environ.get("QWEN38_MEMORY_GUARD"):
        parser.error("must run through memory_guard.py")
    if not 5 <= args.iterations <= 101:
        parser.error("iterations must be 5..101")
    mx.set_memory_limit(2 * 1024**3)
    mx.set_cache_limit(64 * 1024**2)
    ids = list(range(0, 288, 18))
    digest = hashlib.sha256()
    gate_up = read_experts(args.source, ".experts.gate_up_proj", ids, digest)
    dense = [mx.contiguous(gate_up[:, :640]), mx.contiguous(gate_up[:, 640:]),
             read_experts(args.source, ".experts.down_proj", ids, digest)]
    mx.eval(*dense)
    del gate_up
    mx.clear_cache()
    banks = {}
    for projection, weight in enumerate(dense):
        for bits in ([2, 3, 4] if projection < 2 else [4]):
            bank = mx.quantize(weight, group_size=64, bits=bits)
            mx.eval(*bank)
            banks[projection, bits] = bank
    recipes = {"q4": (4, 4, 4), "gate_q2": (2, 4, 4),
               "up_q2": (4, 2, 4), "gate_up_q2": (2, 2, 4),
               "gate_up_q3": (3, 3, 4)}
    slots = args.rows * 10
    mx.random.seed(args.seed)
    tokens = mx.random.normal((args.rows, 2560)).astype(mx.bfloat16)
    route_weights = mx.softmax(mx.random.normal((args.rows, 10)), axis=-1).astype(mx.bfloat16)
    # Ten distinct experts per token; uniform cyclic coverage, not real routing.
    expert_ids = mx.array([i % len(ids) for i in range(slots)], dtype=mx.uint32)
    order = mx.argsort(expert_ids)
    inverse = mx.argsort(order)
    sorted_ids = expert_ids[order]
    x = mx.repeat(tokens[:, None, :], 10, axis=1).reshape(slots, 1, 2560)[order]
    mx.eval(x, sorted_ids, inverse, route_weights)

    def finish(values: mx.array) -> mx.array:
        values = values[inverse].reshape(args.rows, 10, 2560)
        return (values * route_weights[:, :, None]).sum(axis=1)

    def forward(recipe: tuple[int, int, int] | None) -> mx.array:
        def project(value: mx.array, p: int) -> mx.array:
            if recipe is None:
                return mx.gather_mm(value, dense[p].swapaxes(-1, -2),
                                    rhs_indices=sorted_ids, sorted_indices=True)
            bits = recipe[p]
            return mx.gather_qmm(value, *banks[p, bits], rhs_indices=sorted_ids,
                                 sorted_indices=True, group_size=64, bits=bits)
        gate, up = project(x, 0), project(x, 1)
        return finish(project((gate * mx.sigmoid(gate)) * up, 2))

    reference = forward(None)
    mx.eval(reference)
    outputs = {name: forward(recipe) for name, recipe in recipes.items()}
    mx.eval(*outputs.values())
    for name, output in outputs.items():
        if not bool(mx.all(mx.isfinite(output)).item()):
            raise ValueError(f"nonfinite output: {name}")
    for _ in range(3):
        for recipe in recipes.values():
            mx.eval(forward(recipe))
    samples = {name: [] for name in recipes}
    names = list(recipes)
    for iteration in range(args.iterations):
        # Rotate and reverse to reduce fixed-order bias.
        sequence = names[iteration % len(names):] + names[:iteration % len(names)]
        if iteration % 2:
            sequence.reverse()
        for name in sequence:
            start = time.perf_counter()
            result = forward(recipes[name])
            mx.eval(result)
            samples[name].append((time.perf_counter() - start) * 1000)
    results = {}
    for name, recipe in recipes.items():
        results[name] = {
            "projection_bits": recipe,
            "packed_bytes": sum(a.nbytes for p, bits in enumerate(recipe) for a in banks[p, bits]),
            "median_ms": statistics.median(samples[name]), "samples_ms": samples[name],
            "vs_bf16": metrics(outputs[name], reference),
            "vs_q4": metrics(outputs[name], outputs["q4"]),
        }
    print(json.dumps({"source_name": args.source.name, "mlx_version": mx.__version__,
        "rows": args.rows, "expert_ids": ids, "top_k": 10,
        "input": "synthetic normal BF16; fixed cyclic routes; nonuniform weights",
        "seed": args.seed, "sampled_bf16_sha256": digest.hexdigest(),
        "calibrated": False, "production_fused_baseline": False,
        "peak_allocator_bytes": mx.get_peak_memory(), "results": results}), flush=True)


if __name__ == "__main__":
    main()
