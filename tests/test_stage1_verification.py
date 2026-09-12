from pathlib import Path
import unittest

from src.verify_stage1 import verify_stage1


ROOT = Path(__file__).resolve().parents[1]


class Stage1VerificationTests(unittest.TestCase):
    def test_stage1_report_has_all_acceptance_sections(self):
        report = verify_stage1(ROOT, ROOT / "artifacts" / "topic17.sqlite3")
        self.assertEqual(set(report), {"data_integrity", "database", "feature_outputs", "loader", "bipartite_graph", "web_contract", "model", "web_api", "resume_dataset"})
        self.assertTrue(report["data_integrity"]["ok"])
        self.assertTrue(report["database"]["ok"])
        self.assertTrue(report["bipartite_graph"]["ok"])
        self.assertIn(report["model"]["status"], {"not_trained", "optional_dependency_unavailable", "trained"})
        self.assertTrue(report["web_api"]["ok"])
        self.assertTrue(report["resume_dataset"]["ok"])
        self.assertEqual(report["resume_dataset"]["rows"], 300)


if __name__ == "__main__":
    unittest.main()
