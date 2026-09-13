#!/usr/bin/env python3
"""Exercise mixed request and recovery paths against a running server."""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


PROMPTS = (
    "Reply with exactly OK.",
    "Return a JSON object with the integer field value set to 7.",
    "用繁體中文回答：一加一是多少？",
    "Complete this Python expression with one line: sorted([3, 1, 2])",
)


@dataclass
class SoakCounters:
    completed: int = 0
    malformed_rejected: int = 0
    disconnects: int = 0


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: Any | None = None,
    timeout: float = 60.0,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            return response.status, body
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw}
        return error.code, body


def validate_ready(status: dict[str, Any]) -> None:
    if status.get("state") != "ready" or status.get("last_error"):
        raise RuntimeError(f"server did not recover: {status}")
    requests = status.get("requests")
    if not isinstance(requests, dict) or requests.get("active") != 0:
        raise RuntimeError(f"server has a stuck request: {status}")


def malformed_request(base_url: str, timeout: float) -> int:
    endpoint = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=b'{"messages":',
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(endpoint, timeout=timeout)
    except urllib.error.HTTPError as error:
        error.read()
        return error.code
    raise RuntimeError("malformed JSON was unexpectedly accepted")


def disconnect_stream(base_url: str, timeout: float) -> None:
    parsed = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=timeout
    )
    body = json.dumps(
        {
            "model": "qwen38-flash",
            "messages": [{"role": "user", "content": "Count upward forever."}],
            "max_tokens": 64,
            "temperature": 0,
            "thinking": False,
            "stream": True,
        }
    )
    connection.request(
        "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"stream request failed with HTTP {response.status}")
    if not response.readline():
        raise RuntimeError("stream returned no data before disconnect")
    connection.close()


def wait_until_idle(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        code, latest = request_json(base_url, "GET", "/v1/status", timeout=5.0)
        if code == 200 and latest.get("requests", {}).get("active") == 0:
            return latest
        time.sleep(0.05)
    raise RuntimeError(f"server did not become idle: {latest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-disconnect", action="store_true")
    args = parser.parse_args()
    if args.cycles <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("cycles, max-tokens, and timeout must be positive")

    counters = SoakCounters()
    started = time.monotonic()
    for cycle in range(args.cycles):
        for prompt in PROMPTS:
            code, response = request_json(
                args.url,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "qwen38-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "thinking": False,
                    "stream": False,
                },
                args.timeout,
            )
            if code != 200 or not response.get("choices"):
                raise RuntimeError(f"cycle {cycle} completion failed: {code} {response}")
            counters.completed += 1
        if malformed_request(args.url, args.timeout) != 400:
            raise RuntimeError("malformed request did not return HTTP 400")
        counters.malformed_rejected += 1

    if not args.skip_disconnect:
        disconnect_stream(args.url, args.timeout)
        counters.disconnects += 1
    status = wait_until_idle(args.url, args.timeout)
    validate_ready(status)
    print(
        json.dumps(
            {
                "passed": True,
                "elapsed_s": time.monotonic() - started,
                "completed": counters.completed,
                "malformed_rejected": counters.malformed_rejected,
                "disconnects": counters.disconnects,
                "requests": status["requests"],
                "tokens": status.get("tokens", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
