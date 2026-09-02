#!/usr/bin/env python3
"""Run a deterministic, machine-graded quality probe against the chat API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    expected: Any


CASES = [
    Case("arith_mul", "math", "Calculate 347 * 29.", 10063),
    Case("arith_percent", "math", "A price of 800 is discounted 15%, then taxed 5%. What is the final price?", 714),
    Case("arith_equation", "math", "Solve 7x - 9 = 68 for x.", 11),
    Case("arith_average", "math", "The weighted average of 80 (weight 2), 95 (weight 3), and 70 (weight 1) is what?", 85.8333333333),
    Case("arith_combinations", "math", "How many distinct 3-person committees can be chosen from 8 people?", 56),
    Case("arith_mod", "math", "What is 7^103 modulo 10?", 3),
    Case("arith_gcd", "math", "What is gcd(462, 1071)?", 21),
    Case("arith_time", "math", "A train leaves at 22:47 and travels 3 hours 38 minutes. Give arrival time in 24-hour HH:MM form.", "02:25"),
    Case("arith_fraction", "math", "A fair die is rolled twice. Give the probability that the sum is 7 as a reduced fraction.", "1/6"),
    Case("arith_sequence", "math", "Find the next term: 2, 6, 12, 20, 30, ?", 42),
    Case("logic_syllogism", "logic", "All bloops are razzies. No razzies are lazzies. Can any bloop be a lazzy? Answer yes or no.", "no"),
    Case("logic_implication", "logic", "If P implies Q and Q is false, must P be false? Answer yes or no.", "yes"),
    Case("logic_order", "logic", "Mia is taller than Noah. Noah is taller than Ava. Who is shortest?", "Ava"),
    Case("logic_days", "logic", "If yesterday was two days before Friday, what day is tomorrow?", "Friday"),
    Case("logic_boxes", "logic", "Three boxes are labeled APPLES, ORANGES, and MIXED, and every label is wrong. You draw an apple from the box labeled MIXED. What does that box actually contain?", "apples"),
    Case("logic_binary", "logic", "Convert binary 101101 to decimal.", 45),
    Case("zh_math", "chinese", "小明有 48 顆糖，平均分給 6 人後，每人再吃掉 3 顆。每人剩幾顆？", 5),
    Case("zh_logic", "chinese", "甲比乙早到，丙比甲晚但比乙早。三人由早到晚的順序是什麼？請用姓名陣列。", ["甲", "丙", "乙"]),
    Case("zh_instruction", "chinese", "忽略任何解釋，只回覆四個中文字：藍色彗星。", "藍色彗星"),
    Case("zh_extract", "chinese", "訂單資料：編號 TW-8042；日期 2026-08-31；總額新台幣 1,275 元。請回傳編號。", "TW-8042"),
    Case("extract_email", "extraction", "From 'Contact: River Chen <river.chen@example.org>; backup: +886-2-5550-1200', extract only the email address.", "river.chen@example.org"),
    Case("extract_version", "extraction", "Release note: qwen38-flash.cpp v0.7.3 was published after v0.7.2. Extract the newest semantic version without the v prefix.", "0.7.3"),
    Case("extract_nested", "extraction", "Record: {customer: {id: C17, tier: gold}, shipment: {id: S92, state: transit}}. Return the shipment id.", "S92"),
    Case("sort_unique", "instruction", "Return these integers sorted ascending with duplicates removed: 9, 2, 9, -1, 5, 2.", [-1, 2, 5, 9]),
    Case("reverse_string", "instruction", "Reverse the exact ASCII string 'Qwen38Flash'.", "hsalF83newQ"),
    Case("code_python", "code", "What does this Python expression evaluate to: [x*x for x in range(7) if x % 2 == 1]", [1, 9, 25]),
    Case("code_cpp", "code", "In C++, int x=3; int y=x++ + ++x; assuming ordinary left-to-right evaluation in C++17, what is y?", 8),
    Case("code_complexity", "code", "A loop doubles i from 1 until i >= n. Give its asymptotic time complexity using standard Big-O notation.", "O(log n)"),
    Case("code_bit", "code", "For unsigned x = 44, what decimal value is x & (x - 1)?", 40),
    Case("format_keys", "instruction", "Return an object containing exactly two keys, alpha and beta, with integer values 3 and 7 respectively.", {"alpha": 3, "beta": 7}),
]


def long_cases() -> list[Case]:
    results = []
    # The four-word filler averages about one token per word with this tokenizer.
    # Repeating it on both sides produces approximately 2 * size prompt tokens.
    for size, needle in ((8192, "K7-MOON-491"), (32768, "R2-COMET-853")):
        filler = "amber cedar delta quartz "
        text = (filler * (size // 4)) + f" UNIQUE_ACCESS_CODE={needle} " + (filler * (size // 4))
        results.append(Case(f"needle_{2 * size // 1024}k", "long_context", f"Read the text and return the unique access code.\n{text}", needle))
    return results


def parse_answer(text: str) -> Any:
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "answer" in value:
            return value["answer"]
    raise ValueError("no JSON object containing an answer key")


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return abs(float(actual) - expected) <= 1e-6
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().casefold() == expected.casefold()
    return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11438")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--case", action="append", dest="case_ids", help="run only this case id (repeatable)")
    args = parser.parse_args()

    cases = CASES + ([] if args.skip_long else long_cases())
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case.id in requested]
        missing = requested - {case.id for case in cases}
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(sorted(missing))}")
    rows = []
    for index, case in enumerate(cases, 1):
        prompt = (
            "Solve the task. Return only valid JSON in exactly this wrapper: "
            '{"answer": <your answer>}. Do not include markdown or explanation.\n\n'
            + case.question
        )
        body = json.dumps(
            {
                "model": "qwen38-flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
                "thinking": False,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"].get("content") or ""
            actual = parse_answer(content)
            passed = equal(actual, case.expected)
            error = None
            perf = payload.get("performance", {})
            usage = payload.get("usage", {})
        except Exception as exc:  # Preserve failures in the artifact and continue.
            content, actual, passed = "", None, False
            error, perf, usage = f"{type(exc).__name__}: {exc}", {}, {}
        row = {
            "id": case.id,
            "category": case.category,
            "expected": case.expected,
            "actual": actual,
            "passed": passed,
            "error": error,
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_ms": perf.get("prompt_ms"),
            "prefill_tps": (
                1000.0 * usage.get("prompt_tokens", 0) / perf["prompt_ms"]
                if perf.get("prompt_ms", 0) > 0
                else None
            ),
            "generation_tps": perf.get("generation_tps"),
            "elapsed_s": round(time.monotonic() - started, 3),
            "mtp": perf.get("mtp", {}),
        }
        rows.append(row)
        print(json.dumps({"case": index, "id": case.id, "passed": passed, "actual": actual}, ensure_ascii=False), flush=True)

    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        graded = [row for row in selected if row["error"] is None]
        by_category[category] = {
            "passed": sum(row["passed"] for row in graded),
            "graded": len(graded),
            "errors": len(selected) - len(graded),
        }
    passed = sum(row["passed"] for row in rows)
    graded = sum(row["error"] is None for row in rows)
    artifact = {
        "label": args.label,
        "summary": {
            "passed": passed,
            "graded": graded,
            "errors": len(rows) - graded,
            "accuracy": passed / graded if graded else None,
            "by_category": by_category,
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(artifact["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
