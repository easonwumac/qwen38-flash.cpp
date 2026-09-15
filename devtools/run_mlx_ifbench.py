#!/usr/bin/env python3
"""Generate IFBench responses directly through mlx-vlm reference models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mlx_vlm import batch_generate, load


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    raw_input = args.input.read_bytes()
    cases = load_jsonl(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]

    model, processor = load(args.model, strict=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    rows: list[dict[str, Any]] = []
    batch_stats: list[dict[str, Any]] = []

    def render(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    for start in range(0, len(cases), args.concurrency):
        batch = cases[start : start + args.concurrency]
        result = batch_generate(
            model,
            processor,
            prompts=[render(str(case["prompt"])) for case in batch],
            max_tokens=args.max_tokens,
            verbose=False,
        )
        stats = {
            "prompt_tps": result.stats.prompt_tps,
            "generation_tps": result.stats.generation_tps,
            "peak_memory_gb": result.stats.peak_memory,
        }
        batch_stats.append(stats)
        for case, response in zip(batch, result.texts):
            rows.append({
                "key": str(case["key"]),
                "prompt": case["prompt"],
                "response": response,
                "completion_tokens": len(
                    tokenizer.encode(response, add_special_tokens=False)
                ),
                "batch": len(batch_stats) - 1,
            })
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
        print(json.dumps({
            "completed": len(rows),
            "batch_size": len(batch),
            **stats,
        }), flush=True)

    response_rows = [
        {"prompt": row["prompt"], "response": row["response"]} for row in rows
    ]
    write_jsonl(args.responses, response_rows)
    summary = {
        "protocol": {
            "benchmark": "IFBench single-turn OOD test",
            "dataset_sha256": hashlib.sha256(raw_input).hexdigest(),
            "dataset_rows": len(load_jsonl(args.input)),
            "evaluated_rows": len(rows),
            "temperature": 0,
            "thinking": False,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "mtp": "off (no draft model configured)",
        },
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "peak_memory_gb": max(
            (stats["peak_memory_gb"] for stats in batch_stats), default=None
        ),
        "batches": batch_stats,
    }
    args.artifact.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
