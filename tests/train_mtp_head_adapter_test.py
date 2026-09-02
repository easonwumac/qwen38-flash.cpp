#!/usr/bin/env python3

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import train_mtp_head_adapter as trainer  # noqa: E402


class MtpCalibrationRowsTest(unittest.TestCase):
    def test_loads_rows_and_marks_only_live_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.bin"
            width = 3
            data = bytearray(trainer.HEADER.pack(trainer.MAGIC, 1, width))
            for depth, matched in ((1, 1), (2, 0), (3, 1)):
                data.extend(trainer.ROW_PREFIX.pack(depth, 10 + depth, 20 + depth, matched, 7))
                data.extend(np.asarray([depth, depth + 1, depth + 2], dtype="<f4").tobytes())
            path.write_bytes(data)
            rows = trainer.load_rows(path)
            self.assertEqual(rows.hidden.shape, (3, width))
            self.assertEqual(rows.active.tolist(), [True, True, False])
            self.assertEqual(rows.round_id.tolist(), [0, 0, 0])
            self.assertEqual(rows.request_id.tolist(), [0, 0, 0])

    def test_loads_v2_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.bin"
            width = 2
            data = bytearray(trainer.HEADER.pack(trainer.MAGIC_V2, 2, width))
            data.extend(trainer.ROW_PREFIX_V2.pack(1, 10, 10, 1, 7, 42))
            data.extend(np.asarray([1, 2], dtype="<f4").tobytes())
            path.write_bytes(data)
            rows = trainer.load_rows(path)
            self.assertEqual(rows.request_id.tolist(), [42])

    def test_loads_v3_target_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.bin"
            width = 2
            data = bytearray(trainer.HEADER.pack(trainer.MAGIC_V3, 3, width))
            data.extend(trainer.ROW_PREFIX_V2.pack(1, 10, 10, 1, 7, 42))
            data.extend(np.asarray([1, 2], dtype="<f4").tobytes())
            data.extend(np.asarray([3, 4], dtype="<f4").tobytes())
            path.write_bytes(data)
            rows = trainer.load_rows(path)
            self.assertEqual(rows.target_hidden.tolist(), [[3.0, 4.0]])

    def test_rejects_partial_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.bin"
            path.write_bytes(trainer.HEADER.pack(trainer.MAGIC, 1, 3) + b"x")
            with self.assertRaisesRegex(ValueError, "partial row"):
                trainer.load_rows(path)


if __name__ == "__main__":
    unittest.main()
