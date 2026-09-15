#!/usr/bin/env python3
"""Generate IFBench responses directly through mlx-vlm reference models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import batch_generate, generate, load
from mlx_vlm.sample_utils import make_sampler


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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking-budget", type=int, default=2730)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.thinking and args.concurrency != 1:
        parser.error("thinking quality runs require --concurrency 1")

    raw_input = args.input.read_bytes()
    cases = load_jsonl(args.input)
    if args.offset < 0:
        parser.error("--offset must not be negative")
    cases = cases[args.offset:]
    if args.limit is not None:
        cases = cases[: args.limit]

    model, processor = load(args.model, strict=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    mx.random.seed(args.seed)
    rows: list[dict[str, Any]] = []
    batch_stats: list[dict[str, Any]] = []

    def render(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )

    def append_output(
        case: dict[str, Any], raw_response: str, batch: int,
    ) -> None:
        response = raw_response
        reasoning_content = ""
        thinking_closed = None
        if args.thinking:
            reasoning, marker, final = raw_response.partition("</think>")
            thinking_closed = bool(marker)
            reasoning_content = reasoning.removeprefix("<think>").strip()
            response = final.strip() if marker else ""
        rows.append({
            "key": str(case["key"]),
            "prompt": case["prompt"],
            "response": response,
            "reasoning_content": reasoning_content,
            "raw_response": raw_response,
            "thinking_closed": thinking_closed,
            "completion_tokens": len(
                tokenizer.encode(raw_response, add_special_tokens=False)
            ),
            "batch": batch,
        })

    if args.concurrency == 1:
        for index, case in enumerate(cases):
            mx.random.seed(args.seed)
            generation_options: dict[str, Any] = {}
            if args.thinking:
                generation_options["sampler"] = make_sampler(
                    temp=args.temperature, top_p=args.top_p, top_k=args.top_k,
                )
                generation_options.update(
                    thinking_budget=args.thinking_budget,
                    enable_thinking=True,
                )
            result = generate(
                model,
                processor,
                render(str(case["prompt"])),
                max_tokens=args.max_tokens,
                verbose=False,
                **generation_options,
            )
            stats = {
                "prompt_tps": result.prompt_tps,
                "generation_tps": result.generation_tps,
                "peak_memory_gb": result.peak_memory,
            }
            batch_stats.append(stats)
            append_output(case, result.text, index)
            args.artifact.parent.mkdir(parents=True, exist_ok=True)
            args.artifact.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
            print(json.dumps({"completed": len(rows), **stats}), flush=True)

    for start in range(0, len(cases), args.concurrency):
        if args.concurrency == 1:
            break
        batch = cases[start : start + args.concurrency]
        generation_options: dict[str, Any] = {}
        if args.thinking:
            generation_options["sampler"] = make_sampler(
                temp=args.temperature, top_p=args.top_p, top_k=args.top_k,
            )
        result = batch_generate(
            model,
            processor,
            prompts=[render(str(case["prompt"])) for case in batch],
            max_tokens=args.max_tokens,
            verbose=False,
            **generation_options,
        )
        stats = {
            "prompt_tps": result.stats.prompt_tps,
            "generation_tps": result.stats.generation_tps,
            "peak_memory_gb": result.stats.peak_memory,
        }
        batch_stats.append(stats)
        for case, raw_response in zip(batch, result.texts):
            append_output(case, raw_response, len(batch_stats) - 1)
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
            "offset": args.offset,
            "temperature": args.temperature if args.thinking else 0,
            "top_p": args.top_p if args.thinking else None,
            "top_k": args.top_k if args.thinking else None,
            "seed": args.seed if args.thinking else None,
            "thinking_budget": args.thinking_budget if args.thinking else None,
            "thinking": args.thinking,
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
