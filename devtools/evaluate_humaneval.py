#!/usr/bin/env python3
"""Evaluate HumanEval samples in isolated, network-denied subprocesses."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def check_program(problem: dict[str, Any], completion: str) -> str:
    return (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})\n"
    )


def sandbox_profile(writable: Path) -> str:
    escaped = str(writable.resolve()).replace('"', '\\"')
    return f"""(version 1)
(deny default)
(allow process*)
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*)
(allow file-write* (subpath \"{escaped}\"))
(deny network*)
"""


def evaluate_one(problem: dict[str, Any], completion: str, timeout: float) -> str:
    with tempfile.TemporaryDirectory(prefix="qwen38-humaneval-") as temporary:
        root = Path(temporary)
        program = root / "candidate.py"
        program.write_text(check_program(problem, completion))
        try:
            result = subprocess.run(
                [
                    "/usr/bin/sandbox-exec",
                    "-p",
                    sandbox_profile(root),
                    "/usr/bin/python3",
                    "-B",
                    "-I",
                    str(program),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "timed out"
        return "passed" if result.returncode == 0 else f"failed: exit {result.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    problems = {row["task_id"]: row for row in load_jsonl(args.problems)}
    samples = load_jsonl(args.samples)
    seen = {row["task_id"] for row in samples}
    missing = set(problems) - seen
    if missing:
        raise RuntimeError(f"missing {len(missing)} HumanEval tasks")

    results = []
    for index, sample in enumerate(samples, 1):
        task_id = sample["task_id"]
        status = evaluate_one(problems[task_id], sample["completion"], args.timeout)
        results.append({**sample, "result": status, "passed": status == "passed"})
        print(json.dumps({"case": index, "task_id": task_id, "result": status}), flush=True)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results)
    )
    passed = sum(row["passed"] for row in results)
    print(json.dumps({
        "pass@1": passed / len(results),
        "passed": passed,
        "total": len(results),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
