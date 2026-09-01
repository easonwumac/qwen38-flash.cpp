#!/usr/bin/env python3
"""Independent two-token PLE oracle for the retained Qwen3.8 checkpoint."""

import argparse
import json
import math
import os
import struct
from pathlib import Path

import mlx.core as mx


MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB


def splitmix64(value: int) -> int:
    value = (value + GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * MIX1) & MASK64
    value = ((value ^ (value >> 27)) * MIX2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= value // divisor:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def next_prime(value: int) -> int:
    value += 1
    while not is_prime(value):
        value += 1
    return value


def signed64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


class NgramRows:
    def __init__(self, config: dict):
        maximum = (1 << 63) - 1
        half_bound = max(1, (maximum // config["vocab_size"]) // 2)
        seed = config.get("seed") or 1234
        self.multipliers = [
            2 * (splitmix64((seed + GAMMA * (i + 1)) & MASK64) % half_bound) + 1
            for i in range(3)
        ]
        prime = config["ngram_vocab_size_base"] - 1
        self.sizes = []
        self.offsets = []
        offset = 0
        for _ in range(16):
            prime = next_prime(prime)
            self.sizes.append(prime)
            self.offsets.append(offset)
            offset += prime
        self.eos = config["eos_token_id"]
        self.previous = [self.eos, self.eos]
        self.segment_length = 0

    def __call__(self, token: int) -> list[int]:
        previous_one = self.previous[1] if self.segment_length >= 1 else self.eos
        previous_two = self.previous[0] if self.segment_length >= 2 else self.eos
        mixed = ((token * self.multipliers[0]) & MASK64) ^ (
            (previous_one * self.multipliers[1]) & MASK64
        )
        rows = [signed64(mixed) % self.sizes[i] + self.offsets[i] for i in range(8)]
        mixed ^= (previous_two * self.multipliers[2]) & MASK64
        rows += [signed64(mixed) % self.sizes[i] + self.offsets[i] for i in range(8, 16)]
        if token == self.eos:
            self.previous = [self.eos, self.eos]
            self.segment_length = 0
        else:
            self.previous = [self.previous[1], token]
            self.segment_length = min(2, self.segment_length + 1)
        return rows


def bf16(raw: bytes) -> float:
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw)[0] << 16))[0]


def gather_aos(fd: int, rows: list[int]) -> mx.array:
    output = []
    for row in rows:
        raw = os.pread(fd, 100, row * 100)
        if len(raw) != 100:
            raise RuntimeError("short n-gram AoS read")
        scales = [bf16(raw[80 + i * 2 : 82 + i * 2]) for i in range(5)]
        biases = [bf16(raw[90 + i * 2 : 92 + i * 2]) for i in range(5)]
        words = struct.unpack("<20I", raw[:80])
        for index in range(160):
            quantized = (words[index // 8] >> ((index % 8) * 4)) & 15
            group = index // 32
            output.append(quantized * scales[group] + biases[group])
    return mx.array(output, dtype=mx.float32).reshape(1, 1, 2560)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = {}

    def tensor(name):
        shard = index[name]
        if shard not in loaded:
            loaded[shard] = mx.load(str(args.model / shard))
        return loaded[shard][name]

    def qmm(x, name):
        return mx.quantized_matmul(
            x,
            tensor(name + ".weight"),
            scales=tensor(name + ".scales"),
            biases=tensor(name + ".biases"),
            transpose=True,
            group_size=64,
            bits=4,
        )

    def make_stream(token: int) -> mx.array:
        base = "language_model.model.embed_tokens"
        ids = mx.array([token], dtype=mx.int32)
        embedding = mx.dequantize(
            tensor(base + ".weight")[ids],
            tensor(base + ".scales")[ids],
            tensor(base + ".biases")[ids],
            group_size=64,
            bits=4,
        ).reshape(1, 1, 2560)
        return mx.tile(embedding, (1, 1, 4))

    prefix = "language_model.model.layers.1.ple"
    rows = NgramRows(config)
    fd = os.open(args.model / "ngram_table.bin.aos", os.O_RDONLY)
    convolution_state = mx.zeros((1, 9, 10240), dtype=mx.bfloat16)

    def grouped_norm(x, name):
        grouped = x.reshape(1, 1, 4, 2560)
        normalized = mx.fast.rms_norm(
            grouped, mx.ones((2560,), dtype=x.dtype), config["rms_norm_eps"]
        )
        return normalized * (tensor(name + ".weight").reshape(4, 2560) + 1)

    def step(stream, token):
        nonlocal convolution_state
        embedding = gather_aos(fd, rows(token)).astype(mx.bfloat16)
        key = grouped_norm(qmm(embedding, prefix + ".key_proj"), prefix + ".norm_key")
        query = grouped_norm(stream, prefix + ".norm_query")
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(2560)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        value = qmm(embedding, prefix + ".value_proj")
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(1, 1, 10240)
        normalized = grouped_norm(gated, prefix + ".norm_conv").reshape(1, 1, 10240)
        window = mx.concatenate((convolution_state, normalized), axis=1)
        convolution_state = window[:, -9:, :]
        convolution = mx.conv1d(
            window,
            tensor(prefix + ".conv1d.weight").swapaxes(1, 2),
            stride=1,
            padding=0,
            dilation=3,
            groups=10240,
        )
        return gated + convolution * mx.sigmoid(convolution)

    first = step(make_stream(9419), 9419)
    mx.eval(first, convolution_state)
    second = step(make_stream(11), 11)
    mx.eval(second, convolution_state)
    os.close(fd)
    first_values = first.astype(mx.float32).reshape(-1).tolist()
    second_values = second.astype(mx.float32).reshape(-1).tolist()
    print(
        json.dumps(
            {
                "first_checksum": sum(first_values),
                "second_checksum": sum(second_values),
                "first_l1": sum(abs(value) for value in first_values),
                "second_l1": sum(abs(value) for value in second_values),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
