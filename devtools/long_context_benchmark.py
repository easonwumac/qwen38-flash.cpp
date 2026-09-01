#!/usr/bin/env python3
"""Run a deterministic near-limit Qwen3.8 chat-completion benchmark."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


PROMPT_HEADER = "Summarize the following numbered notes in one concise sentence."
PROMPT_PAYLOAD = "alpha beta gamma delta epsilon zeta eta theta"


@dataclass(frozen=True)
class Measurement:
    lines: int
    prompt_bytes: int
    prompt_tokens: int
    completion_tokens: int
    prompt_ms: float
    prompt_tps: float
    generation_ms: float
    generation_tps: float
    wall_ms: float
    output_sha256: str
    output_preview: str


def build_prompt(lines: int) -> str:
    if lines <= 0:
        raise ValueError("lines must be positive")
    notes = "\n".join(
        f"{index:05d}: {PROMPT_PAYLOAD}" for index in range(lines)
    )
    return f"{PROMPT_HEADER}\n{notes}"


def measurement_from_response(
    response: dict[str, Any], lines: int, prompt_bytes: int, wall_ms: float
) -> Measurement:
    try:
        usage = response["usage"]
        performance = response["performance"]
        message = response["choices"][0]["message"]
        output = (message.get("content") or "") + (
            message.get("reasoning_content") or ""
        )
        prompt_tokens = int(usage["prompt_tokens"])
        prompt_ms = float(performance["prompt_ms"])
        if prompt_tokens <= 0 or prompt_ms <= 0:
            raise ValueError("prompt telemetry must be positive")
        return Measurement(
            lines=lines,
            prompt_bytes=prompt_bytes,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(usage["completion_tokens"]),
            prompt_ms=prompt_ms,
            prompt_tps=1000.0 * prompt_tokens / prompt_ms,
            generation_ms=float(performance["generation_ms"]),
            generation_tps=float(performance["generation_tps"]),
            wall_ms=wall_ms,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            output_preview=output[:80],
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"response lacks valid performance telemetry: {error}") from error


def request_once(url: str, lines: int, timeout: float) -> Measurement:
    prompt = build_prompt(lines)
    prompt_bytes = len(prompt.encode("utf-8"))
    body = json.dumps(
        {
            "model": "qwen38-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
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
    print(
        f"long-context: lines={lines} prompt_bytes={prompt_bytes} endpoint={endpoint}",
        file=sys.stderr,
        flush=True,
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
    except (http.client.RemoteDisconnected, TimeoutError, ConnectionError) as error:
        raise RuntimeError(f"request failed: {error}") from error
    wall_ms = 1000.0 * (time.perf_counter() - started)
    if not isinstance(payload, dict):
        raise ValueError("server response must be a JSON object")
    return measurement_from_response(payload, lines, prompt_bytes, wall_ms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--lines", type=int, default=16152)
    parser.add_argument("--min-prompt-tokens", type=int, default=258000)
    parser.add_argument("--max-prompt-tokens", type=int, default=259000)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if args.lines <= 0:
        parser.error("--lines must be positive")
    if (
        args.min_prompt_tokens <= 0
        or args.max_prompt_tokens < args.min_prompt_tokens
        or args.timeout <= 0
    ):
        parser.error("invalid token gate or timeout")

    try:
        measurement = request_once(args.url, args.lines, args.timeout)
    except (RuntimeError, ValueError) as error:
        print(f"long-context: {error}", file=sys.stderr)
        return 2
    failures: list[str] = []
    if not args.min_prompt_tokens <= measurement.prompt_tokens <= args.max_prompt_tokens:
        failures.append(
            f"prompt_tokens={measurement.prompt_tokens} outside "
            f"[{args.min_prompt_tokens},{args.max_prompt_tokens}]"
        )
    if measurement.completion_tokens != 1:
        failures.append(f"completion_tokens={measurement.completion_tokens}, expected 1")
    print(
        json.dumps(
            {
                "gate": {"passed": not failures, "failures": failures},
                "measurement": asdict(measurement),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
