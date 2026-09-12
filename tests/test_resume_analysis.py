import sqlite3
import unittest
from pathlib import Path

from src.database import get_counts
from src.resume_analysis import analyze_resume_text


ROOT = Path(__file__).resolve().parents[1]


class ResumeAnalysisTests(unittest.TestCase):
    def test_analyzes_chinese_resume_without_persisting(self):
        before = get_counts(ROOT / "artifacts" / "topic17.sqlite3")
        result = analyze_resume_text(
            "我熟悉 Python、数据库、机器学习和系统分析，参与过软件开发项目。",
            root=ROOT,
            current_job="15-1251.00",
            target_job="15-1252.00",
        )
        after = get_counts(ROOT / "artifacts" / "topic17.sqlite3")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["persisted"])
        self.assertGreaterEqual(result["skill_count"], 1)
        self.assertTrue(result["recognized_skills"])
        self.assertTrue(result["recommendations"])
        self.assertIn("skills", result["skill_gap"])
        self.assertIn("path", result["career_path"])
        self.assertEqual(before, after)

    def test_rejects_empty_resume_text(self):
        with self.assertRaises(ValueError):
            analyze_resume_text("  \n\t", root=ROOT, current_job="15-1251.00", target_job="15-1252.00")

    def test_auto_selects_current_job_when_omitted(self):
        result = analyze_resume_text("Python 编程 数据库", root=ROOT, target_job="15-1252.00")
        self.assertEqual(result["status"], "ok")
        self.assertIn(result["current_job"], result["occupations"])


if __name__ == "__main__":
    unittest.main()
