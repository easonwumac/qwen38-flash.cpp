#!/usr/bin/env python3
"""Focused CLI validation for the IFBench runner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "devtools" / "run_ifbench.py"
sys.path.insert(0, str(ROOT / "devtools"))

import run_ifbench  # noqa: E402


class RunIfbenchCliTest(unittest.TestCase):
    def run_cli(self, value: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--input",
                    "/dev/null",
                    "--responses",
                    str(Path(directory) / "responses.jsonl"),
                    "--artifact",
                    str(Path(directory) / "artifact.json"),
                    "--chat-template-kwargs",
                    value,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_rejects_non_object_template_arguments(self) -> None:
        result = self.run_cli("[]")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must decode to a JSON object", result.stderr)

    def test_accepts_object_template_arguments(self) -> None:
        result = self.run_cli('{"enable_thinking":true}')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_both_reasoning_field_names(self) -> None:
        self.assertEqual(
            run_ifbench.extract_reasoning_content({"reasoning_content": "custom"}),
            "custom",
        )
        self.assertEqual(
            run_ifbench.extract_reasoning_content({"reasoning": "mlx"}),
            "mlx",
        )

    def test_normalizes_llama_cpp_timings(self) -> None:
        self.assertEqual(
            run_ifbench.extract_performance(
                {
                    "timings": {
                        "prompt_ms": 125.0,
                        "predicted_ms": 250.0,
                        "predicted_per_second": 40.0,
                    }
                }
            ),
            {
                "prompt_ms": 125.0,
                "generation_ms": 250.0,
                "generation_tps": 40.0,
            },
        )

    def test_prefers_native_performance_payload(self) -> None:
        native = {"generation_tps": 57.0}
        self.assertIs(
            run_ifbench.extract_performance(
                {"performance": native, "timings": {"predicted_per_second": 40.0}}
            ),
            native,
        )

    def test_normalizes_splash_metrics(self) -> None:
        self.assertEqual(
            run_ifbench.extract_performance(
                {
                    "metrics": {
                        "request_latency": {
                            "ttft_ms": 350.0,
                            "first_token_to_done_ms": 2500.0,
                            "stream_tokens_per_second": 50.0,
                        }
                    }
                }
            ),
            {
                "prompt_ms": 350.0,
                "generation_ms": 2500.0,
                "generation_tps": 50.0,
            },
        )

    def test_selects_cases_in_requested_key_order(self) -> None:
        cases = [{"key": "0"}, {"key": 10}, {"key": "20"}]
        self.assertEqual(
            [row["key"] for row in run_ifbench.select_cases_by_key(cases, "20,0")],
            ["20", "0"],
        )

    def test_rejects_missing_or_duplicate_keys(self) -> None:
        cases = [{"key": "0"}, {"key": "10"}]
        with self.assertRaisesRegex(ValueError, "unknown --keys: 20"):
            run_ifbench.select_cases_by_key(cases, "0,20")
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            run_ifbench.select_cases_by_key(cases, "0,0")


if __name__ == "__main__":
    unittest.main()
