import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "devtools" / "evaluate_gpqa.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gpqa", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateGpqaTest(unittest.TestCase):
    def test_subset_artifact_scores_only_generated_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.jsonl"
            artifact = root / "artifact.json"
            output = root / "score.json"
            dataset.write_text(
                "\n".join(
                    [
                        json.dumps({"key": "a", "correct_answer": "A"}),
                        json.dumps({"key": "b", "correct_answer": "B"}),
                        json.dumps({"key": "c", "correct_answer": "C"}),
                    ]
                )
                + "\n"
            )
            artifact.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"key": "a", "response": "Answer: A", "error": None},
                            {"key": "c", "response": "Answer: D", "error": None},
                        ]
                    }
                )
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(dataset),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(output.read_text())["summary"]
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["dataset_total"], 3)
            self.assertEqual(summary["parse_failures"], 0)

    def test_answer_pattern_matches_simple_evals_contract(self) -> None:
        self.assertEqual(MODULE.ANSWER_PATTERN.search("Answer: $c$").group(1).upper(), "C")


if __name__ == "__main__":
    unittest.main()
