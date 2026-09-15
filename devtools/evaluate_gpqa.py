#!/usr/bin/env python3
"""Score GPQA generations with the OpenAI simple-evals answer parser."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ANSWER_PATTERN = re.compile(r"Answer[ \t]*:[ \t]*\$?([A-D])\$?", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSON strings may legally contain Unicode line/paragraph separators.  Only
    # an ASCII newline delimits JSONL records.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def wilson_interval(passed: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (passed / total + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        passed * (total - passed) / total**3 + z * z / (4.0 * total**2)
    ) / denominator
    return center - half, center + half


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_jsonl(args.input)
    generated = json.loads(args.artifact.read_text()).get("cases", [])
    by_key = {str(row["key"]): row for row in generated}
    results = []
    for case in cases:
        generated_row = by_key.get(str(case["key"]), {})
        response = str(generated_row.get("response") or "")
        match = ANSWER_PATTERN.search(response)
        extracted = match.group(1).upper() if match else None
        correct = extracted == case["correct_answer"]
        results.append(
            {
                "key": case["key"],
                "correct": correct,
                "expected": case["correct_answer"],
                "extracted": extracted,
                "parse_failure": extracted is None,
                "error": generated_row.get("error"),
                "domain": case.get("domain"),
                "subdomain": case.get("subdomain"),
            }
        )

    passed = sum(bool(row["correct"]) for row in results)
    low, high = wilson_interval(passed, len(results))
    summary = {
        "passed": passed,
        "total": len(results),
        "accuracy": passed / len(results) if results else None,
        "wilson_95": [low, high],
        "parse_failures": sum(bool(row["parse_failure"]) for row in results),
        "generation_errors": sum(bool(row["error"]) for row in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["generation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
