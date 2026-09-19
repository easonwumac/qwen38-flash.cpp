#!/usr/bin/env python3
"""Evaluate HumanEval samples in isolated, network-denied subprocesses."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import sys
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


def validate_samples(problems: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    if not problems:
        raise RuntimeError("empty HumanEval problem set")
    seen = {row["task_id"] for row in samples}
    missing = set(problems) - seen
    if missing:
        raise RuntimeError(f"missing {len(missing)} HumanEval tasks")
    if seen - set(problems) or len(seen) != len(samples):
        raise RuntimeError("unknown or duplicate HumanEval task IDs")


def complete_program_candidate(problem: dict[str, Any], raw: str) -> tuple[dict[str, Any], str]:
    from evalplus.sanitize import sanitize

    # Keep imports and helper definitions. Reattaching the benchmark's function
    # signature to a sanitized complete program changes what is being tested.
    return {**problem, "prompt": ""}, sanitize(raw, problem["entry_point"])


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
                    sys.executable,
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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--samples", type=Path, help="Function-body completion JSONL")
    inputs.add_argument(
        "--artifact", type=Path,
        help="Generation JSON with raw_completion; sanitize full programs with EvalPlus",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive")
    # The system Python path may be an Xcode launcher that cannot start inside
    # the sandbox. Fail before scoring rather than reporting 164 model failures.
    sentinel = {"prompt": "", "entry_point": "answer",
                "test": "def check(f):\n    assert f() == 42"}
    if evaluate_one(sentinel, "def answer(): return 42\n", args.timeout) != "passed":
        raise RuntimeError("HumanEval sandbox/interpreter preflight failed")

    problems = {row["task_id"]: row for row in load_jsonl(args.problems)}
    protocol = {"tests": "original HumanEval", "candidate": "function body"}
    if args.artifact is not None:
        from importlib.metadata import version
        samples = json.loads(args.artifact.read_text())["cases"]
        protocol.update(candidate="sanitized complete raw program",
                        sanitizer=f"EvalPlus {version('evalplus')}")
    else:
        samples = load_jsonl(args.samples)
    validate_samples(problems, samples)

    results = []
    for index, sample in enumerate(samples, 1):
        task_id = sample["task_id"]
        problem = problems[task_id]
        completion = sample.get("completion", "")
        if args.artifact is not None:
            problem, completion = complete_program_candidate(
                problem, sample.get("raw_completion", ""))
        status = "generation error" if sample.get("error") else evaluate_one(
            problem, completion, args.timeout)
        metadata = sample if args.samples is not None else {}
        results.append({**metadata, "task_id": task_id, "completion": completion,
                        "result": status, "passed": status == "passed"})
        print(json.dumps({"case": index, "task_id": task_id, "result": status}), flush=True)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results)
    )
    passed = sum(row["passed"] for row in results)
    print(json.dumps({
        "protocol": protocol,
        "pass@1": passed / len(results),
        "passed": passed,
        "total": len(results),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
