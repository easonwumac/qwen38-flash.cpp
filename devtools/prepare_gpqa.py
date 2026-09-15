#!/usr/bin/env python3
"""Prepare deterministic GPQA prompts using the OpenAI simple-evals contract."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


QUERY_TEMPLATE = """\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    with args.input.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    rng = random.Random(args.seed)
    rows = []
    for repeat in range(args.repeats):
        for index, row in enumerate(source_rows):
            answers = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            permutation = rng.sample(range(4), 4)
            choices = [answers[position] for position in permutation]
            correct_answer = "ABCD"[permutation.index(0)]
            prompt = QUERY_TEMPLATE.format(
                question=row["Question"],
                a=choices[0],
                b=choices[1],
                c=choices[2],
                d=choices[3],
            )
            record_id = row.get("Record ID") or str(index)
            rows.append(
                {
                    "key": f"{record_id}:r{repeat}",
                    "prompt": prompt,
                    "correct_answer": correct_answer,
                    "record_id": record_id,
                    "repeat": repeat,
                    "domain": row.get("High-level domain"),
                    "subdomain": row.get("Subdomain"),
                    "permutation": permutation,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(json.dumps({"source_rows": len(source_rows), "output_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
