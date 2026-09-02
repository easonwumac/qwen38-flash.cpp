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
    def test_greedy_metrics_reports_selective_change_precision(self) -> None:
        count = 3
        rows = trainer.CalibrationRows(
            hidden=np.zeros((count, 2), dtype=np.float32),
            target_hidden=None,
            depth=np.ones(count, dtype=np.uint32),
            draft=np.asarray([1, 2, 3], dtype=np.uint32),
            target=np.asarray([1, 4, 5], dtype=np.uint32),
            matched=np.asarray([True, False, False]),
            position=np.arange(count, dtype=np.uint64),
            request_id=np.zeros(count, dtype=np.uint64),
            round_id=np.arange(count, dtype=np.uint64),
            active=np.ones(count, dtype=np.bool_),
        )
        logits = np.zeros((count, 6), dtype=np.float32)
        logits[0, 1] = 1
        logits[1, 4] = 1
        logits[2, 4] = 1
        metrics = trainer.greedy_metrics(logits, rows, np.arange(count))
        self.assertEqual(metrics["changed"], 2)
        self.assertEqual(metrics["repaired"], 1)
        self.assertEqual(metrics["wrong_to_wrong"], 1)
        self.assertEqual(metrics["repair_precision"], 0.5)

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
