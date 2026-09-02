import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "devtools" / "quality_ab.py"
SPEC = importlib.util.spec_from_file_location("quality_ab", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QualityAbTest(unittest.TestCase):
    def test_parse_answer_tolerates_leading_text(self) -> None:
        self.assertEqual(MODULE.parse_answer('result: {"answer": [1, 2]} trailing'), [1, 2])

    def test_parse_answer_rejects_unwrapped_json(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_answer('{"result": 3}')

    def test_equal_handles_case_and_float_tolerance(self) -> None:
        self.assertTrue(MODULE.equal(" NO ", "no"))
        self.assertTrue(MODULE.equal(85.833333, 85.8333333333))
        self.assertFalse(MODULE.equal(85.8, 85.8333333333))

    def test_long_case_labels_match_approximate_token_scale(self) -> None:
        self.assertEqual([case.id for case in MODULE.long_cases()], ["needle_16k", "needle_64k"])


if __name__ == "__main__":
    unittest.main()
