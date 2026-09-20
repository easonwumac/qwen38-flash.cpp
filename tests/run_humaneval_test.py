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

    def test_evalplus_raw_trim_stops_at_top_level_import(self) -> None:
        text = "    return value\nimport os\n"
        self.assertEqual(MODULE.evalplus_raw_trim(text), "    return value")

    def test_evalplus_prompt_closes_thinking_and_prefills_code_fence(self) -> None:
        prompt = MODULE.evalplus_nonthinking_prompt("def answer():\n    pass")
        self.assertIn("<think>\n\n</think>", prompt)
        self.assertTrue(prompt.endswith("```python\n"))

    def test_evalplus_chat_instruction_has_no_rendered_template(self) -> None:
        prompt = MODULE.evalplus_chat_instruction("def answer():\n    pass")
        self.assertIn("```\ndef answer():\n    pass\n```", prompt)
        self.assertNotIn("<|im_start|>", prompt)
        self.assertNotIn("<think>", prompt)

    def test_solution_to_completion_extracts_fenced_function_body(self) -> None:
        text = "Here it is:\n```python\ndef answer():\n    return 42\n```"
        self.assertEqual(MODULE.solution_to_completion(text, "answer"), "\n    return 42")

    def test_solution_to_completion_accepts_body_only(self) -> None:
        self.assertEqual(
            MODULE.solution_to_completion("    return value\n", "answer"),
            "    return value",
        )


if __name__ == "__main__":
    unittest.main()
