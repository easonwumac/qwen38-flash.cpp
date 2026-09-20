#!/usr/bin/env python3
"""Run HumanEval directly through mlx-vlm for a reference-model control."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from mlx_vlm import batch_generate, generate, load

from run_humaneval import (
    evalplus_nonthinking_prompt,
    evalplus_raw_trim,
    load_problems,
    solution_to_completion,
    write_jsonl,
)


COMMON_STOPS = [
    "<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__",
    "\ndef main(", "\nprint(",
]
RAW_STOPS = COMMON_STOPS + [
    "\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ",
]
CHAT_STOPS = COMMON_STOPS + ["\n```\n", "<|im_end|>"]


class TextStoppingCriteria:
    """Stop on multi-token text without treating the last token as an EOS ID."""

    def __init__(self, tokenizer: Any, stops: list[str]) -> None:
        self.tokenizer = tokenizer
        self.stops = tuple(stops)
        self.tokens: list[int] = []
        self.matched: str | None = None

    def __call__(self, token: Any) -> bool:
        self.tokens.append(int(token))
        tail = self.tokenizer.decode(self.tokens[-64:], skip_special_tokens=False)
        for stop in self.stops:
            if stop in tail:
                self.matched = stop
                return True
        return False

    def remove_partial_stop(self, text: str) -> str:
        """mlx-vlm omits the final matching token; remove the retained prefix."""
        if self.matched is None:
            return text
        for length in range(len(self.matched), 0, -1):
            if text.endswith(self.matched[:length]):
                return text[:-length]
        return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-loader",
        choices=("mlx-vlm", "prism-hadamard"),
        default="mlx-vlm",
        help="Use the checkpoint's bundled Hadamard-aware loader when required.",
    )
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("raw", "evalplus-nonthinking"), required=True,
    )
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Continuous-batch width; 1 preserves per-request stop strings.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    problems = load_problems(args.problems)
    if args.limit is not None:
        problems = problems[: args.limit]
    if args.model_loader == "prism-hadamard":
        model_path = Path(args.model).resolve()
        sys.path.insert(0, str(model_path / "runtime"))
        from vision_artifact import load_vl_model

        model, processor, _ = load_vl_model(model_path)
    else:
        model, processor = load(args.model, strict=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    rows: list[dict[str, Any]] = []

    def prompt_for(problem: dict[str, Any]) -> str:
        if args.mode == "raw":
            return problem["prompt"].strip() + "\n"
        return evalplus_nonthinking_prompt(problem["prompt"])

    def append_row(
        problem: dict[str, Any], raw: str, metrics: dict[str, Any], index: int,
    ) -> None:
        completion = (
            evalplus_raw_trim(raw)
            if args.mode == "raw"
            else solution_to_completion(raw, str(problem["entry_point"]))
        )
        row = {
            "task_id": problem["task_id"],
            "completion": completion,
            "raw_completion": raw,
            **metrics,
        }
        rows.append(row)
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
        print(json.dumps({
            "case": index,
            "task_id": problem["task_id"],
            "tokens": row["generation_tokens"],
            "generation_tps": row["generation_tps"],
        }), flush=True)

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.concurrency == 1:
        for index, problem in enumerate(problems, 1):
            criteria = TextStoppingCriteria(
                tokenizer, RAW_STOPS if args.mode == "raw" else CHAT_STOPS,
            )
            result = generate(
                model,
                processor,
                prompt_for(problem),
                image=None,
                max_tokens=args.max_tokens,
                temperature=0.0,
                stopping_criteria=criteria,
                verbose=False,
            )
            append_row(problem, criteria.remove_partial_stop(result.text), {
                "finish_reason": result.finish_reason,
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "prompt_tps": result.prompt_tps,
                "generation_tps": result.generation_tps,
                "peak_memory_gb": result.peak_memory,
            }, index)
    else:
        for start in range(0, len(problems), args.concurrency):
            batch = problems[start : start + args.concurrency]
            result = batch_generate(
                model,
                processor,
                prompts=[prompt_for(problem) for problem in batch],
                max_tokens=args.max_tokens,
                verbose=False,
            )
            for offset, (problem, raw) in enumerate(zip(batch, result.texts), 1):
                generated_tokens = len(tokenizer.encode(raw, add_special_tokens=False))
                append_row(problem, raw, {
                    "finish_reason": "batch-stop-or-length",
                    "prompt_tokens": None,
                    "generation_tokens": generated_tokens,
                    "prompt_tps": result.stats.prompt_tps,
                    "generation_tps": result.stats.generation_tps,
                    "peak_memory_gb": result.stats.peak_memory,
                }, start + offset)

    write_jsonl(args.samples, [
        {"task_id": row["task_id"], "completion": row["completion"]}
        for row in rows
    ])
    rates = [float(row["generation_tps"]) for row in rows]
    summary = {
        "protocol": {
            "benchmark": "OpenAI HumanEval",
            "mode": args.mode,
            "problems": len(problems),
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "samples_per_problem": 1,
            "concurrency": args.concurrency,
            "mtp": "off (no draft model configured)",
        },
        "completion_tokens": sum(int(row["generation_tokens"]) for row in rows),
        "median_decode_tps": statistics.median(rates) if rates else None,
        "peak_memory_gb": max((float(row["peak_memory_gb"]) for row in rows), default=None),
    }
    args.artifact.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
