import importlib.util
import io
import json
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "devtools" / "build_bf16_ngram_aos.py"
SPEC = importlib.util.spec_from_file_location("build_bf16_ngram_aos", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildBf16NgramAosTest(unittest.TestCase):
    def test_tensor_header_and_copy_range(self) -> None:
        metadata = {"tensor": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]}}
        encoded = json.dumps(metadata).encode()
        source = io.BytesIO(struct.pack("<Q", len(encoded)) + encoded + b"abcd")
        start, decoded = MODULE.tensor_header(source)
        target = io.BytesIO()
        MODULE.copy_range(source, target, start, 4)
        self.assertEqual(decoded, metadata)
        self.assertEqual(target.getvalue(), b"abcd")


if __name__ == "__main__":
    unittest.main()
