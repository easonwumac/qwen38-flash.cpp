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
    def test_samples_must_cover_each_task_once(self) -> None:
        problems = {PROBLEM["task_id"]: PROBLEM}
        MODULE.validate_samples(problems, [PROBLEM])
        for samples in ([], [PROBLEM, PROBLEM], [PROBLEM, {"task_id": "unknown"}]):
            with self.assertRaises(RuntimeError):
                MODULE.validate_samples(problems, samples)
        with self.assertRaises(RuntimeError):
            MODULE.validate_samples({}, [])

    @unittest.skipUnless(importlib.util.find_spec("evalplus"), "EvalPlus not installed")
    def test_full_program_preserves_import_and_helper(self) -> None:
        problem, completion = MODULE.complete_program_candidate(PROBLEM, '''```python
import math
def helper(value):
    return math.floor(value)
def add(a, b):
    return helper(a + b)
```''')
        self.assertEqual(problem["prompt"], "")
        self.assertIn("import math", completion)
        self.assertIn("def helper", completion)
        self.assertEqual(MODULE.evaluate_one(problem, completion, 2.0), "passed")

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

    def test_timeout_is_reported(self) -> None:
        self.assertEqual(MODULE.evaluate_one(PROBLEM, "    while True: pass\n", 0.1), "timed out")


if __name__ == "__main__":
    unittest.main()
