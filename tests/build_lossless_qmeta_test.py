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
        for bits in (13, 16):
            for groups in (10, 40):
                values = np.arange(37 * groups, dtype=np.uint32).reshape(37, groups)
                values = (values * np.uint32(7919)) & np.uint32((1 << bits) - 1)
                packed = QMETADATA.pack_indices(values, groups, bits)
                unpacked = QMETADATA.unpack_indices(packed, groups, bits)
                np.testing.assert_array_equal(unpacked, values)


if __name__ == "__main__":
    unittest.main()
