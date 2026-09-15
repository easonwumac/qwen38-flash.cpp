import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "devtools" / "run_humaneval.py"
SPEC = importlib.util.spec_from_file_location("run_humaneval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunHumanEvalTest(unittest.TestCase):
    def test_trim_completion_uses_first_boundary(self) -> None:
        text = "    return 1\n\ndef extra():\n    pass<|endoftext|>"
        self.assertEqual(MODULE.trim_completion(text), "    return 1")

    def test_trim_completion_preserves_plain_body(self) -> None:
        self.assertEqual(MODULE.trim_completion("    return value\n"), "    return value\n")

    def test_trim_completion_removes_accidental_protocol_suffix(self) -> None:
        text = "    return value\n</parameter>\n</tool_call>"
        self.assertEqual(MODULE.trim_completion(text), "    return value\n")


if __name__ == "__main__":
    unittest.main()
