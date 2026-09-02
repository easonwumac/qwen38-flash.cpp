#!/usr/bin/env python3
"""Send a JSONL prompt suite to a running qwen38 server for calibration."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:11439/v1/completions")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.suite.read_text().splitlines() if line.strip()]
    cases = cases[args.offset :]
    if args.limit is not None:
        cases = cases[: args.limit]
    totals = {"rounds": 0, "proposed": 0, "accepted": 0, "tokens": 0}
    for index, case in enumerate(cases, 1):
        prompt = case.get("prompt", case.get("problem"))
        if prompt is None:
            raise ValueError(f"case {index} has neither prompt nor problem")
        requested = int(case.get("max_tokens", args.max_tokens))
        payload = json.dumps(
            {
                "prompt": str(prompt),
                "max_tokens": min(args.max_tokens, requested),
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            args.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        performance = result.get("performance", {})
        mtp = performance.get("mtp", {})
        usage = result.get("usage", {})
        totals["rounds"] += int(mtp.get("rounds", 0))
        totals["proposed"] += int(mtp.get("proposed", 0))
        totals["accepted"] += int(mtp.get("accepted", 0))
        totals["tokens"] += int(usage.get("completion_tokens", 0))
        print(
            json.dumps(
                {
                    "case": index,
                    "id": case.get("id", index),
                    "tps": performance.get("generation_tps"),
                    "rounds": mtp.get("rounds"),
                    "accepted": mtp.get("accepted"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(json.dumps({"summary": totals}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
