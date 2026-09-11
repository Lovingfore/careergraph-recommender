import json
from pathlib import Path
import unittest

from src.generate_chinese_resumes import generate_resumes


ROOT = Path(__file__).resolve().parents[1]


class ChineseResumeDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd

        cls.occupations = pd.read_csv(ROOT / "data" / "clean" / "occupations.csv", dtype=str)
        cls.occupation_skill = pd.read_csv(ROOT / "data" / "clean" / "occupation_skill.csv")
        cls.skills = pd.read_csv(ROOT / "data" / "clean" / "skills.csv", dtype=str)

    def test_generates_300_deterministic_chinese_resumes(self):
        first = generate_resumes(self.occupations, self.occupation_skill, self.skills, count=300, seed=20260911)
        second = generate_resumes(self.occupations, self.occupation_skill, self.skills, count=300, seed=20260911)
        self.assertEqual(len(first), 300)
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))
        required = {
            "resume_id", "user_id", "education", "major", "years_experience", "city",
            "current_job", "target_job", "skills", "skill_levels", "work_experience",
            "project_experience", "self_introduction", "resume_text", "source",
        }
        self.assertTrue(required <= set(first.columns))
        self.assertTrue(first["resume_text"].map(lambda value: any("\u4e00" <= char <= "\u9fff" for char in value)).all())
        skill_levels = json.loads(first.iloc[0]["skill_levels"])
        self.assertTrue(set(skill_levels) <= set(self.skills["skill_id"]))
        self.assertEqual(first["user_id"].nunique(), 300)
        self.assertEqual(first["resume_id"].nunique(), 300)

    def test_prepared_dataset_contains_300_resume_rows(self):
        import pandas as pd

        resumes = pd.read_csv(ROOT / "data" / "clean" / "resumes_zh.csv")
        summary = json.loads((ROOT / "data" / "clean" / "dataset_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(resumes), 300)
        self.assertEqual(summary["resumes"], 300)
        self.assertEqual(summary["users"], 300)
        self.assertTrue((resumes["source"] == "synthetic_chinese_resume").all())


if __name__ == "__main__":
    unittest.main()
