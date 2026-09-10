from pathlib import Path
import os
import unittest


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.topic17_web.settings")


class WebSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import django

        django.setup()

    def test_summary_api_returns_counts_and_loader_status(self):
        from django.test import Client

        response = Client().get("/api/summary/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("counts", payload)
        self.assertIn("loader", payload)
        self.assertEqual(payload["counts"]["occupations"], 12)

    def test_index_page_is_reachable(self):
        from django.test import Client

        response = Client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CareerGraph Recommender", response.content)

    def test_recommendation_api_returns_rows(self):
        from django.test import Client

        response = Client().get("/api/recommend/?user_id=u001&top_k=3")
        self.assertEqual(response.status_code, 200)
        self.assertIn("recommendations", response.json())

    def test_skill_gap_api_returns_skills(self):
        from django.test import Client

        response = Client().get("/api/skill-gap/?user_id=u001&occupation_id=15-1244.00")
        self.assertEqual(response.status_code, 200)
        self.assertIn("skills", response.json())

    def test_career_path_api_returns_path(self):
        from django.test import Client

        response = Client().get("/api/career-path/?user_id=u001&occupation_id=15-1252.00")
        self.assertEqual(response.status_code, 200)
        self.assertIn("path", response.json())

    def test_forecast_api_returns_explicit_status(self):
        from django.test import Client

        response = Client().get("/api/forecast/?user_id=u001&months=6")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_recommendation_api_requires_user_id(self):
        from django.test import Client

        response = Client().get("/api/recommend/")
        self.assertEqual(response.status_code, 400)

    def test_bipartite_graph_is_viewable(self):
        from django.test import Client

        response = Client().get("/bipartite-graph.svg")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/svg+xml", response["Content-Type"])


if __name__ == "__main__":
    unittest.main()
