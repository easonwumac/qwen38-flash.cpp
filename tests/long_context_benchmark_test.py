import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "devtools" / "long_context_benchmark.py"
SPEC = importlib.util.spec_from_file_location("long_context_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LongContextBenchmarkTest(unittest.TestCase):
    def test_build_prompt_is_deterministic(self) -> None:
        self.assertEqual(
            MODULE.build_prompt(2),
            "Summarize the following numbered notes in one concise sentence.\n"
            "00000: alpha beta gamma delta epsilon zeta eta theta\n"
            "00001: alpha beta gamma delta epsilon zeta eta theta",
        )
        with self.assertRaises(ValueError):
            MODULE.build_prompt(0)

    def test_measurement_parses_api_telemetry(self) -> None:
        response = {
            "usage": {"prompt_tokens": 258457, "completion_tokens": 1},
            "performance": {
                "prompt_ms": 2500000.0,
                "generation_ms": 200.0,
                "generation_tps": 5.0,
            },
            "choices": [
                {"message": {"reasoning_content": "", "content": "All"}}
            ],
        }
        measurement = MODULE.measurement_from_response(response, 16152, 900000, 2500250.0)
        self.assertEqual(measurement.prompt_tokens, 258457)
        self.assertAlmostEqual(measurement.prompt_tps, 103.3828, places=4)
        self.assertEqual(measurement.output_preview, "All")


if __name__ == "__main__":
    unittest.main()
