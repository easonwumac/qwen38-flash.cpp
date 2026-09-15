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


if __name__ == "__main__":
    unittest.main()
