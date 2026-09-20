#!/usr/bin/env python3
"""Generate deterministic HumanEval completions through the native server."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def load_problems(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def trim_completion(text: str) -> str:
    """Apply the project's original conservative completion boundaries."""
    boundaries = [
        "<|endoftext|>", "</s>", "</parameter>", "</function>", "</tool_call>",
        "```", "\n\n\n", "\n\ndef ", "\n\nclass ",
    ]
    stops = [text.find(marker) for marker in boundaries if text.find(marker) >= 0]
    return text[: min(stops)] if stops else text


def evalplus_raw_trim(text: str) -> str:
    """Apply EvalPlus direct-completion stop strings."""
    boundaries = [
        "<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__",
        "\ndef main(", "\nprint(", "\ndef ", "\nclass ", "\nimport ",
        "\nfrom ", "\nassert ",
    ]
    stops = [text.find(marker) for marker in boundaries if text.find(marker) >= 0]
    return text[: min(stops)] if stops else text


def evalplus_nonthinking_prompt(task_prompt: str) -> str:
    """Render the EvalPlus chat protocol used for instruction-tuned models."""
    instruction = evalplus_chat_instruction(task_prompt)
    response = (
        "Below is a Python script with a self-contained function that solves the "
        "problem and passes corresponding tests:"
    )
    return (
        f"<|im_start|>user\n{instruction}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{response}\n```python\n"
    )


def evalplus_chat_instruction(task_prompt: str) -> str:
    """Return the user message for chat-only OpenAI-compatible servers."""
    return (
        "Please provide a self-contained Python script that solves the following "
        f"problem in a markdown code block:\n```\n{task_prompt.strip()}\n```"
    )


def solution_to_completion(text: str, entrypoint: str) -> str:
    """Extract an entry-point body so OpenAI HumanEval can append it."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = (blocks[-1] if blocks else text).strip("\n")
    signature = re.search(
        rf"(?m)^def\s+{re.escape(entrypoint)}\s*\([^\n]*\).*:\s*$", code,
    )
    if signature is None:
        return "\n" + code if code and not code[0].isspace() else code
    body: list[str] = []
    for line in code[signature.end():].lstrip("\n").splitlines():
        if line and not line[0].isspace():
            break
        body.append(line)
    return "\n" + "\n".join(body).rstrip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--model", default="qwen38-flash")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=(
            "raw",
            "chat-nonthinking",
            "evalplus-nonthinking",
            "chat-evalplus-nonthinking",
        ),
        default="raw",
    )
    parser.add_argument(
        "--stop-profile", choices=("project", "evalplus", "none"), default="project",
        help="Post-generation boundaries for raw/body modes.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    if args.limit is not None:
        problems = problems[: args.limit]

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.artifact.exists():
        previous = json.loads(args.artifact.read_text())
        completed = {
            row["task_id"]: row
            for row in previous.get("cases", [])
            if not row.get("error")
        }

    rows: list[dict[str, Any]] = []
    for index, problem in enumerate(problems, 1):
        task_id = str(problem["task_id"])
        if task_id in completed:
            rows.append(completed[task_id])
            continue
        if args.mode in ("raw", "evalplus-nonthinking"):
            endpoint = "/v1/completions"
            body = {
                "model": args.model,
                "prompt": (
                    problem["prompt"] if args.mode == "raw"
                    else evalplus_nonthinking_prompt(problem["prompt"])
                ),
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
        elif args.mode == "chat-evalplus-nonthinking":
            endpoint = "/v1/chat/completions"
            body = {
                "model": args.model,
                "messages": [{
                    "role": "user",
                    "content": evalplus_chat_instruction(problem["prompt"]),
                }],
                "temperature": 0,
                "reasoning_effort": "none",
                "max_tokens": args.max_tokens,
                "stream": False,
            }
        else:
            endpoint = "/v1/chat/completions"
            body = {
                "model": args.model,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Complete the Python function below. Return only the indented "
                        "function body: no Markdown fence, explanation, signature, or tests.\n\n"
                        + problem["prompt"]
                    ),
                }],
                "temperature": 0,
                "thinking": False,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
        request = urllib.request.Request(
            args.url.rstrip("/") + endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        row: dict[str, Any] = {"task_id": task_id}
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.load(response)
            choice = payload["choices"][0]
            raw = (
                choice["text"]
                if args.mode in ("raw", "evalplus-nonthinking")
                else choice["message"]["content"]
            )
            if args.mode in ("evalplus-nonthinking", "chat-evalplus-nonthinking"):
                completion = solution_to_completion(raw, str(problem["entry_point"]))
            elif args.stop_profile == "evalplus":
                completion = evalplus_raw_trim(raw)
            elif args.stop_profile == "none":
                completion = raw
            else:
                completion = trim_completion(raw)
            performance = payload.get("performance", {})
            if not performance and payload.get("metrics"):
                rate = payload["metrics"].get("request_latency", {}).get(
                    "stream_tokens_per_second"
                )
                if rate is not None:
                    performance = {"generation_tps": rate}
            row.update(
                completion=completion,
                raw_completion=raw,
                finish_reason=choice.get("finish_reason"),
                usage=payload.get("usage", {}),
                performance=performance,
                error=None,
            )
        except Exception as exc:
            row.update(completion="", error=f"{type(exc).__name__}: {exc}")
        row["wall_seconds"] = time.monotonic() - started
        rows.append(row)
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
        print(json.dumps({
            "case": index,
            "task_id": task_id,
            "error": row.get("error"),
            "tokens": row.get("usage", {}).get("completion_tokens"),
            "generation_tps": row.get("performance", {}).get("generation_tps"),
        }), flush=True)

    sample_rows = [
        {"task_id": row["task_id"], "completion": row["completion"]}
        for row in rows
    ]
    write_jsonl(args.samples, sample_rows)
    rates = [
        float(row["performance"]["generation_tps"])
        for row in rows
        if not row.get("error") and row.get("performance", {}).get("generation_tps") is not None
    ]
    summary = {
        "protocol": {
            "benchmark": "OpenAI HumanEval",
            "problems": len(problems),
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "completion_mode": args.mode,
            "stop_profile": args.stop_profile,
            "samples_per_problem": 1,
        },
        "errors": sum(bool(row.get("error")) for row in rows),
        "completion_tokens": sum(
            int(row.get("usage", {}).get("completion_tokens", 0)) for row in rows
        ),
        "median_decode_tps": statistics.median(rates) if rates else None,
    }
    args.artifact.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
