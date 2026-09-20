from pathlib import Path
import unittest

from src.recommendation_service import (
    _select_path,
    career_path_for_user,
    forecast_for_user,
    recommend_for_user,
    skill_profile_for_user,
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
        self.assertIn("missing_skills", result["recommendations"][0])
        if result["recommendations"][-1]["missing_skills"]:
            self.assertIn("skill_name_zh", result["recommendations"][-1]["missing_skills"][0])

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
        self.assertEqual(result["strategy"], "fast")
        self.assertEqual(result["strategy_name"], "快速路线")

    def test_stable_path_and_invalid_strategy(self):
        stable = career_path_for_user("u001", "15-1252.00", root=ROOT, strategy="stable")
        invalid = career_path_for_user("u001", "15-1252.00", root=ROOT, strategy="unknown")
        self.assertEqual(stable["status"], "ok")
        self.assertEqual(stable["strategy_name"], "稳健路线")
        self.assertEqual(invalid["status"], "error")

    def test_path_strategies_have_distinct_objectives(self):
        graph = {
            "a": [("d", 0.2), ("b", 0.9)],
            "b": [("d", 0.9)],
        }
        self.assertEqual(_select_path(graph, "a", "d", 3, "fast"), (["a", "d"], 0.2))
        self.assertEqual(_select_path(graph, "a", "d", 3, "stable"), (["a", "b", "d"], 0.81))

    def test_skill_profile_contains_named_levels(self):
        result = skill_profile_for_user("u001", root=ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(len(result["skills"]), 3)
        self.assertIn("skill_name_zh", result["skills"][0])
        self.assertIn("level", result["skills"][0])

    def test_forecast_uses_explicit_baseline_status(self):
        result = forecast_for_user("u001", months=6, root=ROOT)
        self.assertIn(result["status"], {"baseline", "unavailable"})
        self.assertNotEqual(result["status"], "temporal_gat")


if __name__ == "__main__":
    unittest.main()
