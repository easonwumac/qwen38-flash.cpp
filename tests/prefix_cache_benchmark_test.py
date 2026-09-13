import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "devtools" / "prefix_cache_benchmark.py"
SPEC = importlib.util.spec_from_file_location("prefix_cache_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrefixCacheBenchmarkTest(unittest.TestCase):
    def test_assistant_turn_preserves_reasoning_and_tools(self) -> None:
        payload = {"choices": [{"message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": "reason",
            "tool_calls": [{"id": "call_1"}],
        }}]}
        self.assertEqual(
            MODULE.assistant_turn(payload),
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "reason",
                "tool_calls": [{"id": "call_1"}],
            },
        )


if __name__ == "__main__":
    unittest.main()
