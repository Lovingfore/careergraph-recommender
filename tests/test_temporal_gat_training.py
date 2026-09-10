from pathlib import Path
import tempfile
import unittest

from src.train_temporal_gat import build_training_plan


ROOT = Path(__file__).resolve().parents[1]


class TemporalGATTrainingTests(unittest.TestCase):
    def test_build_training_plan_is_deterministic_and_keeps_test_window(self):
        plan = build_training_plan(10)
        self.assertEqual(plan, {"train": [0, 1, 2, 3, 4, 5], "validation": [6, 7], "test": [8, 9]})

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch not installed in test environment")
    def test_run_training_writes_checkpoint_and_metrics(self):
        from src.train_temporal_gat import run_training

        with tempfile.TemporaryDirectory() as tmp:
            result = run_training(ROOT / "data" / "clean", Path(tmp), epochs=2, seed=7)
            self.assertIn("baseline", result)
            self.assertIn("temporal_gat", result)
            self.assertIn("test", result)
            self.assertTrue((Path(tmp) / "temporal_gat.pt").exists())
            self.assertTrue((Path(tmp) / "training_log.csv").exists())
            self.assertTrue((Path(tmp) / "evaluation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
