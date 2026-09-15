#!/usr/bin/env python3
"""Generate IFBench responses while retaining qwen38 performance telemetry."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSON strings may legally contain Unicode line/paragraph separators.  Only
    # an ASCII newline delimits JSONL records.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--benchmark", default="IFBench single-turn OOD test")
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--model", default="qwen38-flash")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        default="xhigh",
        help="Thinking budget hint; matches the model chat-template default (xhigh).",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    args = parser.parse_args()
    if args.concurrency < 1 or args.concurrency > 4:
        parser.error("--concurrency must be between 1 and 4")
    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.no_thinking else 1.0

    raw_input = args.input.read_bytes()
    cases = load_jsonl(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.artifact.exists():
        prior = json.loads(args.artifact.read_text())
        # A response that stopped on its former length limit is not reusable if
        # a resumed run raises that limit. Re-running it avoids silently mixing
        # truncated and complete generations in one benchmark artifact.
        completed = {
            row["key"]: row
            for row in prior.get("cases", [])
            if not row.get("error") and row.get("finish_reason") != "length"
        }

    def generate(index: int, case: dict[str, Any]) -> dict[str, Any]:
        key = str(case["key"])
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if args.no_thinking:
            body["thinking"] = False
        else:
            body["reasoning_effort"] = args.reasoning_effort
            body.update(
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                frequency_penalty=args.frequency_penalty,
            )
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        row: dict[str, Any] = {
            "key": key,
            "prompt": case["prompt"],
            "case_index": index,
        }
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.load(response)
            message = payload["choices"][0]["message"]
            row.update(
                response=message.get("content") or "",
                reasoning_content=message.get("reasoning_content") or "",
                finish_reason=payload["choices"][0].get("finish_reason"),
                usage=payload.get("usage", {}),
                performance=payload.get("performance", {}),
                error=None,
            )
        except Exception as exc:
            row.update(response="", error=f"{type(exc).__name__}: {exc}")
        row["wall_seconds"] = time.monotonic() - started
        return row

    rows_by_key = dict(completed)
    pending = [
        (index, case)
        for index, case in enumerate(cases, 1)
        if str(case["key"]) not in completed
    ]
    evaluation_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(generate, index, case) for index, case in pending]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows_by_key[row["key"]] = row
            print(
                json.dumps(
                    {
                        "case": row["case_index"],
                        "key": row["key"],
                        "error": row.get("error"),
                        "completion_tokens": row.get("usage", {}).get("completion_tokens"),
                        "request_generation_tps": row.get("performance", {}).get(
                            "generation_tps"
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            rows = [
                rows_by_key[str(case["key"])]
                for case in cases
                if str(case["key"]) in rows_by_key
            ]
            artifact = {"cases": rows}
            args.artifact.parent.mkdir(parents=True, exist_ok=True)
            args.artifact.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")

    evaluation_wall_seconds = time.monotonic() - evaluation_started
    rows = [rows_by_key[str(case["key"])] for case in cases]

    response_rows = [{"prompt": row["prompt"], "response": row["response"]} for row in rows]
    write_jsonl(args.responses, response_rows)
    decode_rates = [
        float(row["performance"]["generation_tps"])
        for row in rows
        if not row.get("error") and row.get("performance", {}).get("generation_tps") is not None
    ]
    prefill_rates = [
        1000.0 * float(row["usage"]["prompt_tokens"]) / float(row["performance"]["prompt_ms"])
        for row in rows
        if not row.get("error") and float(row.get("performance", {}).get("prompt_ms", 0)) > 0
    ]
    prompt_ms = sum(float(row.get("performance", {}).get("prompt_ms", 0)) for row in rows)
    generation_denominator_ms = (
        sum(float(row.get("performance", {}).get("generation_ms", 0)) for row in rows)
        if args.concurrency == 1
        else max(0.0, evaluation_wall_seconds * 1000.0 - prompt_ms)
    )
    completion_tokens = sum(
        int(row.get("usage", {}).get("completion_tokens", 0)) for row in rows
    )
    summary = {
        "protocol": {
            "benchmark": args.benchmark,
            "temperature": temperature,
            "top_p": None if args.no_thinking else args.top_p,
            "top_k": None if args.no_thinking else args.top_k,
            "seed": None if args.no_thinking else args.seed,
            "frequency_penalty": None if args.no_thinking else args.frequency_penalty,
            "thinking": not args.no_thinking,
            "reasoning_effort": None if args.no_thinking else args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "dataset_sha256": hashlib.sha256(raw_input).hexdigest(),
            "dataset_rows": len(load_jsonl(args.input)),
            "evaluated_rows": len(rows),
            "concurrency": args.concurrency,
            "scheduling": "rolling" if args.concurrency > 1 else "serial",
        },
        "errors": sum(bool(row.get("error")) for row in rows),
        "completion_tokens": completion_tokens,
        "aggregate_decode_tps": (
            1000.0 * completion_tokens / generation_denominator_ms
            if generation_denominator_ms > 0
            else None
        ),
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "end_to_end_tps": (
            completion_tokens / evaluation_wall_seconds if evaluation_wall_seconds > 0 else None
        ),
        "decode_tps": {
            "median": statistics.median(decode_rates) if decode_rates else None,
            "p10": percentile(decode_rates, 0.1),
            "p90": percentile(decode_rates, 0.9),
        },
        "prefill_tps": {
            "median": statistics.median(prefill_rates) if prefill_rates else None,
            "p10": percentile(prefill_rates, 0.1),
            "p90": percentile(prefill_rates, 0.9),
        },
    }
    args.artifact.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
