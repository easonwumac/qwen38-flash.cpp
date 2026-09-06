import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "study", Path(__file__).resolve().parents[1] / "devtools" / "expert_trace_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def key(expert, layer=0):
    return f"language_model.model.layers.{layer}.mlp/{expert}"


class ReplayTest(unittest.TestCase):
    def test_reload_and_hot_hits(self):
        events = []
        for expert in [0, 1, 2, 1]:
            events.extend([("test", 0, key(expert), 1), ("test", 1, "", 0)])
        stats = study.replay(events, {key(0)}, 1)["test"]
        self.assertEqual(stats["hot_hits"], 1)
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["reloads"], 1)
        self.assertEqual(stats["evictions"], 2)

    def test_pinned_capacity(self):
        with self.assertRaisesRegex(ValueError, "leases"):
            study.replay([("test", 0, key(0), 1), ("test", 0, key(1), 1)], set(), 1)

    def test_global_borrows_across_layers(self):
        events = []
        for expert in [0, 1, 0, 1]:
            events.extend([("test", 0, key(expert), 1), ("test", 1, "", 0)])
        self.assertEqual(study.replay(events, set(), 2)["test"]["misses"], 2)
        self.assertEqual(study.replay(events, set(), 1, True)["test"]["misses"], 4)

    def test_ranking_does_not_train_on_heldout(self):
        events = [("warmup_decode", 0, key(0), 2764800), ("warmup_decode", 1, "", 0),
                  ("prompt1_decode", 0, key(287), 2764800), ("prompt1_decode", 1, "", 0)]
        result = study.analyze(events)
        self.assertEqual(result["layers"][0]["training_counts"][287], 0)
        self.assertEqual(result["layers"][0]["heldout_counts"][287], 1)
        config = next(c for c in result["configs"] if c["name"] == "ranked16g_dynamic8g")
        self.assertAlmostEqual(config["hot_gib"], 16, delta=0.003)


if __name__ == "__main__":
    unittest.main()
