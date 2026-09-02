#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "devtools"))
import build_training_view  # noqa: E402


class BuildTrainingViewTest(unittest.TestCase):
    def test_excludes_runtime_qmeta_without_copying_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model"
            output = root / "view"
            source.mkdir()
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "model-00001.safetensors",
                "model-mtp-q8.safetensors",
                "model-qmeta-lossless16.safetensors",
            ):
                (source / name).write_bytes(b"fixture")

            manifest = build_training_view.build_view(source, output)

            self.assertEqual(
                manifest["excluded"],
                ["model-mtp-q8.safetensors", "model-qmeta-lossless16.safetensors"],
            )
            self.assertTrue((output / "model-00001.safetensors").is_symlink())
            self.assertFalse((output / "model-mtp-q8.safetensors").exists())
            self.assertTrue((output / "mtp.safetensors").is_symlink())
            self.assertEqual(manifest["mtp_sidecar"], "model-mtp-q8.safetensors")
            self.assertFalse((output / "model-qmeta-lossless16.safetensors").exists())

    def test_refuses_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model"
            output = root / "view"
            source.mkdir()
            output.mkdir()
            (output / "keep").write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                build_training_view.build_view(source, output)


if __name__ == "__main__":
    unittest.main()
