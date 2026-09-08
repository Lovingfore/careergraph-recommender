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


if __name__ == "__main__":
    unittest.main()
