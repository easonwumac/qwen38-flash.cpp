#!/usr/bin/env python3
"""Run IFBench through mlx-lm while reproducing qwen38's forced thinking close."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


FORCED_THINKING_SUFFIX = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def split_thinking_output(text: str) -> tuple[str, str, bool]:
    reasoning, marker, final = text.partition("</think>")
    if not marker:
        return reasoning.removeprefix("<think>").strip(), "", False
    return (
        reasoning.removeprefix("<think>").strip(),
        final.strip(),
        True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=2730)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.thinking_budget < 1 or args.thinking_budget >= args.max_tokens:
        parser.error("--thinking-budget must be positive and below --max-tokens")

    raw_input = args.input.read_bytes()
    cases = load_jsonl(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]

    model, tokenizer = load(args.model, trust_remote_code=True)
    close_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    forced_suffix = tokenizer.encode(FORCED_THINKING_SUFFIX, add_special_tokens=False)
    if len(close_tokens) != 1:
        raise RuntimeError("Qwen thinking close marker must encode to one token")
    close_token = int(close_tokens[0])
    eos_tokens = set(getattr(tokenizer, "eos_token_ids", []))
    if not eos_tokens:
        eos = getattr(tokenizer, "eos_token_id", None)
        eos_tokens = {int(eos)} if eos is not None else set()
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)

    rows: list[dict[str, Any]] = []
    evaluation_started = time.monotonic()
    for case_index, case in enumerate(cases, 1):
        mx.random.seed(args.seed)
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(case["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = mx.array(tokenizer.encode(rendered, add_special_tokens=False))
        cache = make_prompt_cache(model)
        produced: list[int] = []
        forced = False
        thinking_closed = False
        finish_reason = "length"
        started = time.monotonic()
        first_token_at: float | None = None
        generator = generate_step(
            prompt,
            model,
            max_tokens=args.max_tokens,
            sampler=sampler,
            prompt_cache=cache,
        )

        def continue_after_forcing(include_last_token: bool) -> Any:
            nonlocal forced, thinking_closed, generator
            forced = True
            thinking_closed = True
            bridge = ([produced[-1]] if include_last_token else []) + forced_suffix
            produced.extend(forced_suffix)
            remaining = args.max_tokens - len(produced)
            generator.close()
            generator = generate_step(
                mx.array(bridge),
                model,
                max_tokens=remaining,
                sampler=sampler,
                prompt_cache=cache,
            )
            return generator

        while len(produced) < args.max_tokens:
            try:
                token_array, _ = next(generator)
            except StopIteration:
                break
            if first_token_at is None:
                first_token_at = time.monotonic()
            token = int(
                token_array.item() if hasattr(token_array, "item") else token_array
            )
            if token in eos_tokens:
                if not thinking_closed and len(produced) + len(forced_suffix) < args.max_tokens:
                    continue_after_forcing(False)
                    continue
                finish_reason = "stop"
                break
            produced.append(token)
            if token == close_token:
                thinking_closed = True
            if (
                not thinking_closed
                and len(produced) >= args.thinking_budget
                and len(produced) + len(forced_suffix) < args.max_tokens
            ):
                continue_after_forcing(True)

        generator.close()
        wall_seconds = time.monotonic() - started
        prompt_seconds = (
            first_token_at - started if first_token_at is not None else wall_seconds
        )
        generation_seconds = max(wall_seconds - prompt_seconds, 0.0)
        raw_response = tokenizer.decode(produced)
        reasoning, response, closed = split_thinking_output(raw_response)
        row = {
            "key": str(case["key"]),
            "prompt": case["prompt"],
            "case_index": case_index,
            "response": response,
            "reasoning_content": reasoning,
            "raw_response": raw_response,
            "finish_reason": finish_reason,
            "thinking_closed": closed,
            "thinking_budget_forced": forced,
            "usage": {
                "prompt_tokens": int(prompt.size),
                "completion_tokens": len(produced),
                "total_tokens": int(prompt.size) + len(produced),
            },
            "performance": {
                "prompt_tps": int(prompt.size) / prompt_seconds if prompt_seconds else None,
                "generation_tps": (
                    len(produced) / generation_seconds if generation_seconds else None
                ),
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            },
            "wall_seconds": wall_seconds,
            "error": None,
        }
        rows.append(row)
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(
            json.dumps({"cases": rows}, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "case": case_index,
                    "key": row["key"],
                    "completion_tokens": len(produced),
                    "forced": forced,
                    "thinking_closed": closed,
                    "generation_tps": row["performance"]["generation_tps"],
                }
            ),
            flush=True,
        )
        del cache
        mx.clear_cache()

    summary = {
        "protocol": {
            "benchmark": "IFBench single-turn OOD test",
            "dataset_sha256": hashlib.sha256(raw_input).hexdigest(),
            "dataset_rows": len(load_jsonl(args.input)),
            "evaluated_rows": len(rows),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "thinking": True,
            "thinking_budget": args.thinking_budget,
            "forced_suffix": FORCED_THINKING_SUFFIX,
            "max_tokens": args.max_tokens,
            "concurrency": 1,
            "mtp": "off",
        },
        "completion_tokens": sum(row["usage"]["completion_tokens"] for row in rows),
        "forced": sum(row["thinking_budget_forced"] for row in rows),
        "evaluation_wall_seconds": time.monotonic() - evaluation_started,
    }
    args.artifact.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    write_jsonl(
        args.responses,
        [{"prompt": row["prompt"], "response": row["response"]} for row in rows],
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
