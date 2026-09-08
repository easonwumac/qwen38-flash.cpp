"""Small CPU-side input validation checks; requires the developer MLX Python."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).resolve().parents[1] / "devtools/selective_expert_bits_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class InputValidation(unittest.TestCase):
    def check_header(self, header, message):
        data = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            path.write_bytes(struct.pack("<Q", len(data)) + data)
            with self.assertRaisesRegex(ValueError, message):
                probe.read_experts(path, ".experts.down_proj", [0], hashlib.sha256())

    def test_missing_projection(self):
        self.check_header({}, "exactly one")

    def test_wrong_geometry(self):
        self.check_header({"x.experts.down_proj": {
            "dtype": "BF16", "shape": [1, 1, 1], "data_offsets": [0, 2]}}, "geometry")

    def test_wrong_dtype(self):
        self.check_header({"x.experts.down_proj": {
            "dtype": "F16", "shape": [288, 2560, 640],
            "data_offsets": [0, 943718400]}}, "geometry")

    def test_bad_span(self):
        self.check_header({"x.experts.down_proj": {
            "dtype": "BF16", "shape": [288, 2560, 640],
            "data_offsets": [0, 1]}}, "byte span")

    def test_truncated_data(self):
        self.check_header({"x.experts.down_proj": {
            "dtype": "BF16", "shape": [288, 2560, 640],
            "data_offsets": [0, 943718400]}}, "truncated tensor")

    def test_oversized_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            path.write_bytes(struct.pack("<Q", 17 * 1024**2))
            with self.assertRaisesRegex(ValueError, "header size"):
                probe.read_experts(path, ".experts.down_proj", [0], hashlib.sha256())


if __name__ == "__main__":
    unittest.main()
