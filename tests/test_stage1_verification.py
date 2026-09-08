from pathlib import Path
import unittest

from src.verify_stage1 import verify_stage1


ROOT = Path(__file__).resolve().parents[1]


class Stage1VerificationTests(unittest.TestCase):
    def test_stage1_report_has_all_acceptance_sections(self):
        report = verify_stage1(ROOT, ROOT / "artifacts" / "topic17.sqlite3")
        self.assertEqual(set(report), {"data_integrity", "database", "feature_outputs", "loader", "web_contract"})
        self.assertTrue(report["data_integrity"]["ok"])
        self.assertTrue(report["database"]["ok"])


if __name__ == "__main__":
    unittest.main()
