from pathlib import Path
import unittest

from src.recommendation_service import (
    career_path_for_user,
    forecast_for_user,
    recommend_for_user,
    skill_gap_for_user,
)


ROOT = Path(__file__).resolve().parents[1]


class RecommendationServiceTests(unittest.TestCase):
    def test_recommendations_are_ranked_and_limited(self):
        result = recommend_for_user("u001", top_k=3, root=ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(len(result["recommendations"]), 3)
        ranks = [item["rank"] for item in result["recommendations"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_missing_user_returns_error(self):
        result = recommend_for_user("missing", root=ROOT)
        self.assertEqual(result["status"], "error")

    def test_skill_gap_contains_named_skills(self):
        result = skill_gap_for_user("u001", "15-1244.00", root=ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["occupation_id"], "15-1244.00")
        self.assertTrue(result["skills"])
        self.assertIn("skill_name", result["skills"][0])

    def test_career_path_is_available_from_current_job(self):
        result = career_path_for_user("u001", "15-1252.00", root=ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["path"][0], "15-1251.00")
        self.assertEqual(result["path"][-1], "15-1252.00")

    def test_forecast_uses_explicit_baseline_status(self):
        result = forecast_for_user("u001", months=6, root=ROOT)
        self.assertIn(result["status"], {"baseline", "unavailable"})
        self.assertNotEqual(result["status"], "temporal_gat")


if __name__ == "__main__":
    unittest.main()
