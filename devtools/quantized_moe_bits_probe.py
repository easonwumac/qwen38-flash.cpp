#!/usr/bin/env python3
"""Compare MLX affine bit widths on the Qwen3.8 routed-MoE geometry.

This is a synthetic kernel feasibility probe, not a model-quality benchmark.  It
uses gather_qmm for the same gate/up/down dimensions and S*top-k work shape as
the verifier so an expensive model conversion is only justified when the lower
bit width is actually faster on the local GPU.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def quantized_bank(shape: tuple[int, ...], bits: int, key: mx.array):
    weight = mx.random.normal(shape=shape, key=key, dtype=mx.bfloat16) * 0.02
    packed = mx.quantize(weight, group_size=64, bits=bits)
    mx.eval(*packed)
    return packed


def main() -> None:
    args = parse_args()
    if args.rows < 1 or args.top_k < 1 or args.experts < args.top_k:
        raise SystemExit("require rows >= 1 and experts >= top-k >= 1")

    slots = args.rows * args.top_k
    keys = mx.random.split(mx.random.key(38), 5)
    x = mx.random.normal((slots, 1, 2560), key=keys[0], dtype=mx.bfloat16)
    route_weights = mx.full((args.rows, args.top_k, 1, 1), 1.0 / args.top_k)
    expert_ids = mx.array(
        [index % args.experts for index in range(slots)], dtype=mx.uint32
    )
    mx.eval(x, route_weights, expert_ids)

    outputs: dict[int, mx.array] = {}
    for bits in args.bits:
        gate = quantized_bank((args.experts, 640, 2560), bits, keys[1])
        up = quantized_bank((args.experts, 640, 2560), bits, keys[2])
        down = quantized_bank((args.experts, 2560, 640), bits, keys[3])

        def forward() -> mx.array:
            gate_y = mx.gather_qmm(
                x, *gate, rhs_indices=expert_ids, group_size=64, bits=bits
            )
            up_y = mx.gather_qmm(
                x, *up, rhs_indices=expert_ids, group_size=64, bits=bits
            )
            hidden = (gate_y * mx.sigmoid(gate_y)) * up_y
            down_y = mx.gather_qmm(
                hidden, *down, rhs_indices=expert_ids, group_size=64, bits=bits
            )
            return (down_y.reshape(args.rows, args.top_k, 1, 2560) * route_weights).sum(1)

        for _ in range(args.warmup):
            mx.eval(forward())
        samples = []
        output = forward()
        mx.eval(output)
        for _ in range(args.iterations):
            started = time.perf_counter()
            output = forward()
            mx.eval(output)
            samples.append((time.perf_counter() - started) * 1_000.0)
        outputs[bits] = output
        packed_bytes = sum(array.nbytes for bank in (gate, up, down) for array in bank)
        print(
            f"bits={bits} packed_mib={packed_bytes / 2**20:.2f} "
            f"median_ms={statistics.median(samples):.4f} "
            f"min_ms={min(samples):.4f} max_ms={max(samples):.4f}"
        )

    if 3 in outputs and 4 in outputs:
        a = outputs[3].astype(mx.float32).reshape(-1)
        b = outputs[4].astype(mx.float32).reshape(-1)
        cosine = mx.sum(a * b) / mx.sqrt(mx.sum(a * a) * mx.sum(b * b))
        relative_rmse = mx.sqrt(mx.mean(mx.square(a - b))) / mx.sqrt(mx.mean(mx.square(b)))
        mx.eval(cosine, relative_rmse)
        print(
            f"q3_vs_q4_cosine={cosine.item():.8f} "
            f"relative_rmse={relative_rmse.item():.6f}"
        )


if __name__ == "__main__":
    main()
