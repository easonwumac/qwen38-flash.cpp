#!/usr/bin/env python3
"""Validate generated-turn RAM prefix reuse through the chat API."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("response must be an object")
    return payload


def assistant_turn(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload["choices"][0]["message"]
    turn: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    if message.get("reasoning_content") is not None:
        turn["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls") is not None:
        turn["tool_calls"] = message["tool_calls"]
    return turn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--padding-lines", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--require-first-cache", action="store_true")
    args = parser.parse_args()
    if args.padding_lines < 0 or args.timeout <= 0:
        parser.error("invalid padding or timeout")

    first_user = "Remember that the verification code is BLUE-4821. Reply only: remembered."
    if args.padding_lines:
        first_user += "\nContext:\n" + "\n".join(
            f"note {index}: stable local inference" for index in range(args.padding_lines)
        )
    common = {
        "model": "qwen38-flash",
        "temperature": 0,
        "thinking": False,
        "stream": False,
    }
    first = post(
        args.url,
        {**common, "messages": [{"role": "user", "content": first_user}], "max_tokens": 16},
        args.timeout,
    )
    messages = [
        {"role": "user", "content": first_user},
        assistant_turn(first),
        {"role": "user", "content": "What was the verification code? Reply only with the code."},
    ]
    second = post(
        args.url,
        {**common, "messages": messages, "max_tokens": 16},
        args.timeout,
    )
    first_perf = first["performance"]
    second_perf = second["performance"]
    first_cached = int(first_perf.get("cached_prompt_tokens", 0))
    cached = int(second_perf.get("cached_prompt_tokens", 0))
    output = second["choices"][0]["message"].get("content") or ""
    report = {
        "gate": {
            "passed": (
                cached > 0
                and "BLUE-4821" in output
                and (not args.require_first_cache or first_cached > 0)
            ),
            "failures": [
                reason
                for condition, reason in (
                    (cached == 0, "second turn did not reuse a cached prefix"),
                    (
                        args.require_first_cache and first_cached == 0,
                        "first turn did not restore a persistent prefix",
                    ),
                    ("BLUE-4821" not in output, "second turn lost the verification code"),
                )
                if condition
            ],
        },
        "first": {
            "prompt_tokens": first["usage"]["prompt_tokens"],
            "cached_prompt_tokens": first_cached,
            "prompt_ms": first_perf["prompt_ms"],
            "completion_tokens": first["usage"]["completion_tokens"],
        },
        "second": {
            "prompt_tokens": second["usage"]["prompt_tokens"],
            "cached_prompt_tokens": cached,
            "prompt_ms": second_perf["prompt_ms"],
            "completion_tokens": second["usage"]["completion_tokens"],
            "content": output,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
