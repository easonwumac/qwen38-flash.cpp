import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "devtools" / "evaluate_humaneval.py"
SPEC = importlib.util.spec_from_file_location("evaluate_humaneval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


PROBLEM = {
    "task_id": "test/0",
    "prompt": "def add(a, b):\n",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
    "entry_point": "add",
}


class EvaluateHumanEvalTest(unittest.TestCase):
    def test_sandbox_accepts_correct_completion(self) -> None:
        self.assertEqual(
            MODULE.evaluate_one(PROBLEM, "    return a + b\n", 2.0),
            "passed",
        )

    def test_sandbox_rejects_wrong_completion(self) -> None:
        self.assertTrue(
            MODULE.evaluate_one(PROBLEM, "    return a - b\n", 2.0).startswith("failed")
        )

    def test_network_is_denied(self) -> None:
        completion = "    import socket\n    socket.socket().connect(('example.com', 80))\n"
        self.assertTrue(MODULE.evaluate_one(PROBLEM, completion, 2.0).startswith("failed"))


if __name__ == "__main__":
    unittest.main()
