from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "devtools" / "build_lossless_qmeta.py"
SPEC = importlib.util.spec_from_file_location("build_lossless_qmeta", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load lossless qmeta builder")
QMETADATA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QMETADATA
SPEC.loader.exec_module(QMETADATA)


class PackedQmetaTest(unittest.TestCase):
    def test_lossless_round_trip_for_qwen_projection_widths(self) -> None:
        for bits in (9, 13, 16):
            for groups in (10, 40):
                values = np.arange(37 * groups, dtype=np.uint32).reshape(37, groups)
                values = (values * np.uint32(7919)) & np.uint32((1 << bits) - 1)
                packed = QMETADATA.pack_indices(values, groups, bits)
                unpacked = QMETADATA.unpack_indices(packed, groups, bits)
                np.testing.assert_array_equal(unpacked, values)

    def test_lossy_medoid_dictionary_is_bounded_and_deterministic(self) -> None:
        def pair(scale: float, bias: float) -> np.uint32:
            scale_bits = np.asarray([scale], dtype=np.float32).view(np.uint32)[0] >> 16
            bias_bits = np.asarray([bias], dtype=np.float32).view(np.uint32)[0] >> 16
            return np.uint32(scale_bits | (bias_bits << np.uint32(16)))

        values = np.asarray(
            [pair(1.0, 0.0)] * 100
            + [pair(2.0, 0.0)] * 80
            + [pair(3.0, 0.0)] * 2
            + [pair(20.0, 0.0)],
            dtype=np.uint32,
        )
        first = QMETADATA.lossy_medoid_dictionary(values, 2)
        second = QMETADATA.lossy_medoid_dictionary(values, 2)
        self.assertEqual(first[0].size, 2)
        self.assertEqual(first[1].shape, values.shape)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertGreater(first[2]["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
