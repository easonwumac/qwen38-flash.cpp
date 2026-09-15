import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "devtools" / "build_q8_ngram_aos.py"
SPEC = importlib.util.spec_from_file_location("build_q8_ngram_aos", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildQ8NgramAosTest(unittest.TestCase):
    def test_affine_q8_round_trip(self) -> None:
        values = np.linspace(-3.0, 5.0, MODULE.DIMENSION, dtype=np.float32)[None, :]
        source = MODULE.float_to_bf16(values)
        encoded = MODULE.quantize_rows(source)
        self.assertEqual(encoded.shape, (1, MODULE.Q8_ROW_BYTES))
        q = encoded[:, : MODULE.DIMENSION].reshape(1, MODULE.GROUPS, MODULE.GROUP_SIZE)
        scales = encoded[:, MODULE.DIMENSION : MODULE.DIMENSION + MODULE.GROUPS * 2]
        biases = encoded[:, MODULE.DIMENSION + MODULE.GROUPS * 2 :]
        scales = MODULE.bf16_to_float(scales.copy().view("<u2"))
        biases = MODULE.bf16_to_float(biases.copy().view("<u2"))
        restored = q.astype(np.float32) * scales[:, :, None] + biases[:, :, None]
        expected = MODULE.bf16_to_float(source).reshape(restored.shape)
        self.assertLess(float(np.max(np.abs(restored - expected))), 0.02)

    def test_constant_groups_are_exact(self) -> None:
        values = np.full((1, MODULE.DIMENSION), 2.0, dtype=np.float32)
        encoded = MODULE.quantize_rows(MODULE.float_to_bf16(values))
        self.assertTrue(np.all(encoded[:, : MODULE.DIMENSION] == 0))


if __name__ == "__main__":
    unittest.main()
