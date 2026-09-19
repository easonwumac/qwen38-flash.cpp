#!/usr/bin/env python3
"""Model-free argument gates: invalid probes must fail before GPU allocation."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


BUILD = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "build-release"
)


class RuntimeProbeCliTest(unittest.TestCase):
    def reject(self, executable, args, message, *, guarded=True, toggle=None):
        env = dict(os.environ)
        env.pop("QWEN38_MEMORY_GUARD", None)
        env.pop("QWEN38_QSA_BENCH_TOGGLE", None)
        if guarded:
            env["QWEN38_MEMORY_GUARD"] = "1"
        if toggle is not None:
            env["QWEN38_QSA_BENCH_TOGGLE"] = toggle
        result = subprocess.run(
            [str(BUILD / executable), "missing-model-for-cli-test", *args],
            env=env, text=True, capture_output=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)

    def test_runtime_requires_guard(self):
        self.reject("qwen38-runtime-ab-probe", ["QWEN38_TEST"],
                    "Usage under memory_guard", guarded=False)

    def test_runtime_switch_name(self):
        self.reject("qwen38-runtime-ab-probe", ["INVALID"],
                    "invalid runtime switch name")

    def test_runtime_output_bounds(self):
        for count in ("0", "4097"):
            with self.subTest(count=count):
                self.reject("qwen38-runtime-ab-probe", ["QWEN38_TEST", count],
                            "TOKENS must be 1..4096")

    def test_runtime_long_prompt_bounds(self):
        for lines in ("0", "8193"):
            with self.subTest(lines=lines):
                self.reject("qwen38-runtime-ab-probe", ["QWEN38_TEST", "128", lines],
                            "LINES must be 1..8192")

    def test_qsa_requires_guard(self):
        self.reject("qwen38-qsa-peak-probe", ["32768", "decode"],
                    "usage: guarded", guarded=False)

    def test_qsa_context_bounds(self):
        self.reject("qwen38-qsa-peak-probe", ["200000", "decode"],
                    "invalid bounded probe configuration")

    def test_qsa_mode(self):
        self.reject("qwen38-qsa-peak-probe", ["32768", "99"],
                    "invalid bounded probe configuration")

    def test_qsa_switch_name(self):
        self.reject("qwen38-qsa-peak-probe", ["32768", "decode"],
                    "invalid QSA benchmark switch", toggle="NOT_A_RUNTIME_SWITCH")


if __name__ == "__main__":
    unittest.main()
