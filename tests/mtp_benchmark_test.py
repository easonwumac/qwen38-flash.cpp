from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "devtools" / "mtp_benchmark.py"
SPEC = importlib.util.spec_from_file_location("mtp_benchmark", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import mtp_benchmark")
mtp_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mtp_benchmark
SPEC.loader.exec_module(mtp_benchmark)


def fixture_response(tokens: int = 128, tps: float = 65.0) -> dict:
    return {
        "usage": {"completion_tokens": tokens},
        "performance": {
            "generation_tps": tps,
            "generation_ms": 1000.0 * tokens / tps,
            "mtp": {
                "rounds": 37,
                "proposed": 103,
                "accepted": 89,
                "fallbacks": 0,
                "depth": 3,
            },
        },
    }


class MtpBenchmarkTest(unittest.TestCase):
    def test_parses_measurement_and_acceptance(self) -> None:
        result = mtp_benchmark.measurement_from_response(
            fixture_response(), requested_tokens=128, wall_ms=2200.0
        )
        self.assertEqual(result.accepted, 89)
        self.assertAlmostEqual(result.acceptance, 89 / 103)
        self.assertEqual(result.completion_tokens, 128)

    def test_summarizes_median_and_path(self) -> None:
        samples = [
            mtp_benchmark.measurement_from_response(
                fixture_response(tps=tps), requested_tokens=128, wall_ms=2200.0
            )
            for tps in (64.8, 65.2, 65.0)
        ]
        summary = mtp_benchmark.summarize(samples)
        self.assertEqual(summary["median_tps"], 65.0)
        self.assertEqual(summary["paths"], ["37:89/103:d3"])

    def test_gate_detects_speed_acceptance_and_missing_mtp(self) -> None:
        sample = mtp_benchmark.measurement_from_response(
            fixture_response(tps=64.0), requested_tokens=128, wall_ms=2200.0
        )
        failures = mtp_benchmark.gate_failures(
            {128: [sample]}, {128: 65.0}, minimum_acceptance=0.9, require_mtp=True
        )
        self.assertEqual(len(failures), 2)

    def test_rejects_incomplete_generation(self) -> None:
        sample = mtp_benchmark.measurement_from_response(
            fixture_response(tokens=127), requested_tokens=128, wall_ms=2200.0
        )
        failures = mtp_benchmark.gate_failures(
            {128: [sample]}, {}, minimum_acceptance=0.0, require_mtp=False
        )
        self.assertIn("128 sample 1: completed 127 tokens", failures)

    def test_parses_per_length_thresholds(self) -> None:
        self.assertEqual(
            mtp_benchmark.parse_token_values("128:64.5,256:66.5"),
            {128: 64.5, 256: 66.5},
        )


if __name__ == "__main__":
    unittest.main()
