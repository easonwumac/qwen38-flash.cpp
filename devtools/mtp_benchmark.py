#!/usr/bin/env python3
"""Reproduce and gate the retained high-acceptance Qwen3.8 MTP fixture."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


FIXTURE_PROMPT = """Complete this Python function efficiently and return only code:

def merge_sorted_unique(a, b):
    pass"""


@dataclass(frozen=True)
class Measurement:
    requested_tokens: int
    completion_tokens: int
    generation_tps: float
    generation_ms: float
    rounds: int
    proposed: int
    accepted: int
    proposed_by_position: tuple[int, int, int, int]
    accepted_by_position: tuple[int, int, int, int]
    fallbacks: int
    depth: int
    promotions: int
    demotions: int
    draft_ms: float
    verify_ms: float
    commit_ms: float
    wall_ms: float

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def parse_token_values(value: str) -> dict[int, float]:
    """Parse comma-separated TOKEN:VALUE pairs used by per-length gates."""
    result: dict[int, float] = {}
    if not value:
        return result
    for item in value.split(","):
        try:
            token_text, threshold_text = item.split(":", 1)
            tokens = int(token_text)
            threshold = float(threshold_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"expected TOKEN:VALUE, received {item!r}"
            ) from error
        if tokens <= 0 or threshold < 0:
            raise argparse.ArgumentTypeError("TOKEN must be positive and VALUE non-negative")
        result[tokens] = threshold
    return result


def parse_token_counts(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("token counts must be comma-separated integers") from error
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("token counts must be positive and unique")
    return result


def measurement_from_response(
    response: dict[str, Any], requested_tokens: int, wall_ms: float
) -> Measurement:
    try:
        usage = response["usage"]
        performance = response["performance"]
        mtp = performance["mtp"]

        def position_counts(name: str) -> tuple[int, int, int, int]:
            raw = mtp.get(name, [0, 0, 0, 0])
            if (
                not isinstance(raw, list)
                or len(raw) != 4
                or any(not isinstance(value, int) or value < 0 for value in raw)
            ):
                raise ValueError(f"mtp.{name} must contain four non-negative integers")
            return raw[0], raw[1], raw[2], raw[3]

        return Measurement(
            requested_tokens=requested_tokens,
            completion_tokens=int(usage["completion_tokens"]),
            generation_tps=float(performance["generation_tps"]),
            generation_ms=float(performance["generation_ms"]),
            rounds=int(mtp["rounds"]),
            proposed=int(mtp["proposed"]),
            accepted=int(mtp["accepted"]),
            proposed_by_position=position_counts("proposed_by_position"),
            accepted_by_position=position_counts("accepted_by_position"),
            fallbacks=int(mtp["fallbacks"]),
            depth=int(mtp["depth"]),
            promotions=int(mtp.get("promotions", 0)),
            demotions=int(mtp.get("demotions", 0)),
            draft_ms=float(mtp.get("draft_ms", 0.0)),
            verify_ms=float(mtp.get("verify_ms", 0.0)),
            commit_ms=float(mtp.get("commit_ms", 0.0)),
            wall_ms=wall_ms,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"response lacks valid performance telemetry: {error}") from error


def request_once(url: str, requested_tokens: int, timeout: float) -> Measurement:
    body = json.dumps(
        {
            "model": "qwen38-flash",
            "messages": [{"role": "user", "content": FIXTURE_PROMPT}],
            "max_tokens": requested_tokens,
            "temperature": 0,
            "thinking": False,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed: {error.reason}") from error
    wall_ms = 1000.0 * (time.perf_counter() - started)
    if not isinstance(payload, dict):
        raise ValueError("server response must be a JSON object")
    return measurement_from_response(payload, requested_tokens, wall_ms)


def summarize(samples: list[Measurement]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")
    tps = [sample.generation_tps for sample in samples]
    acceptance = [sample.acceptance for sample in samples]
    proposed_by_position = [
        sum(sample.proposed_by_position[position] for sample in samples)
        for position in range(4)
    ]
    accepted_by_position = [
        sum(sample.accepted_by_position[position] for sample in samples)
        for position in range(4)
    ]
    return {
        "samples": len(samples),
        "median_tps": statistics.median(tps),
        "min_tps": min(tps),
        "max_tps": max(tps),
        "median_acceptance": statistics.median(acceptance),
        "proposed_by_position": proposed_by_position,
        "accepted_by_position": accepted_by_position,
        "median_phase_ms": {
            "draft": statistics.median(sample.draft_ms for sample in samples),
            "verify": statistics.median(sample.verify_ms for sample in samples),
            "commit": statistics.median(sample.commit_ms for sample in samples),
        },
        "acceptance_by_position": [
            accepted / proposed if proposed else 0.0
            for accepted, proposed in zip(accepted_by_position, proposed_by_position)
        ],
        "paths": sorted(
            {
                f"{sample.rounds}:{sample.accepted}/{sample.proposed}:d{sample.depth}:"
                f"p{sample.promotions}/m{sample.demotions}"
                for sample in samples
            }
        ),
    }


def gate_failures(
    samples_by_tokens: dict[int, list[Measurement]],
    minimum_tps: dict[int, float],
    minimum_acceptance: float,
    require_mtp: bool,
    require_promotion: bool = False,
) -> list[str]:
    failures: list[str] = []
    for tokens, samples in samples_by_tokens.items():
        summary = summarize(samples)
        threshold = minimum_tps.get(tokens)
        if threshold is not None and summary["median_tps"] < threshold:
            failures.append(
                f"{tokens}: median {summary['median_tps']:.3f} tok/s < {threshold:.3f}"
            )
        if summary["median_acceptance"] < minimum_acceptance:
            failures.append(
                f"{tokens}: median acceptance {summary['median_acceptance']:.3f} "
                f"< {minimum_acceptance:.3f}"
            )
        for index, sample in enumerate(samples, start=1):
            if sample.completion_tokens != tokens:
                failures.append(
                    f"{tokens} sample {index}: completed {sample.completion_tokens} tokens"
                )
            if require_mtp and sample.proposed == 0:
                failures.append(f"{tokens} sample {index}: MTP did not engage")
            if require_promotion and sample.promotions == 0:
                failures.append(f"{tokens} sample {index}: MTP did not promote")
    for tokens in minimum_tps:
        if tokens not in samples_by_tokens:
            failures.append(f"no samples collected for {tokens}-token speed gate")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the retained deterministic Python MTP fixture"
    )
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--tokens", type=parse_token_counts, default=[128, 256])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--min-median-tps",
        type=parse_token_values,
        default={},
        metavar="TOKEN:TPS,...",
    )
    parser.add_argument("--min-acceptance", type=float, default=0.0)
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.samples <= 0 or args.timeout <= 0:
        parser.error("warmups must be non-negative; samples and timeout must be positive")
    if not 0.0 <= args.min_acceptance <= 1.0:
        parser.error("--min-acceptance must be between 0 and 1")

    samples_by_tokens: dict[int, list[Measurement]] = {}
    try:
        for tokens in args.tokens:
            for _ in range(args.warmups):
                request_once(args.url, tokens, args.timeout)
            samples_by_tokens[tokens] = [
                request_once(args.url, tokens, args.timeout) for _ in range(args.samples)
            ]
    except (RuntimeError, ValueError) as error:
        print(f"mtp_benchmark: {error}", file=sys.stderr)
        return 2

    report = {
        "fixture": "merge_sorted_unique",
        "sampling": {"temperature": 0, "thinking": False},
        "results": {
            str(tokens): {
                "summary": summarize(samples),
                "samples": [
                    {**asdict(sample), "acceptance": sample.acceptance}
                    for sample in samples
                ],
            }
            for tokens, samples in samples_by_tokens.items()
        },
    }
    failures = gate_failures(
        samples_by_tokens,
        args.min_median_tps,
        args.min_acceptance,
        args.require_mtp,
        args.require_promotion,
    )
    report["gate"] = {"passed": not failures, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
