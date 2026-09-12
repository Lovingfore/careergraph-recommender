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
        self.assertIn("上传简历".encode("utf-8"), response.content)
        self.assertIn(b"TemporalGAT", response.content)

    def test_model_info_api_returns_training_metadata(self):
        from django.test import Client

        response = Client().get("/api/model-info/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "trained")
        self.assertIn("epochs", payload)
        self.assertIn("sample_count", payload)
        self.assertIn("baseline", payload)
        self.assertIn("temporal_gat", payload)

    def test_occupations_api_returns_select_options(self):
        from django.test import Client

        response = Client().get("/api/occupations/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["occupations"]), 12)

    def test_resume_upload_is_dynamic_and_does_not_change_database(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import Client
        from src.database import get_counts

        db_path = ROOT / "artifacts" / "topic17.sqlite3"
        before = get_counts(db_path)
        upload = SimpleUploadedFile(
            "resume.txt",
            "我熟悉 Python、数据库、机器学习和系统分析，参与过软件开发项目。".encode("utf-8"),
            content_type="text/plain",
        )
        response = Client().post(
            "/api/resume-upload/",
            {"resume": upload, "current_job": "15-1251.00", "target_job": "15-1252.00"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["persisted"])
        self.assertTrue(payload["recommendations"])
        self.assertIn("career_path", payload)
        self.assertEqual(before, get_counts(db_path))

    def test_resume_upload_rejects_invalid_extension_and_empty_text(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import Client

        client = Client()
        invalid = SimpleUploadedFile("resume.exe", b"Python", content_type="application/octet-stream")
        empty = SimpleUploadedFile("resume.txt", b"  \n", content_type="text/plain")
        self.assertEqual(client.post("/api/resume-upload/", {"resume": invalid}).status_code, 400)
        self.assertEqual(client.post("/api/resume-upload/", {"resume": empty}).status_code, 400)

    def test_resume_upload_rejects_payload_over_one_megabyte(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import Client

        oversized = SimpleUploadedFile("resume.txt", b"a" * (1024 * 1024 + 1), content_type="text/plain")
        response = Client().post("/api/resume-upload/", {"resume": oversized})
        self.assertEqual(response.status_code, 400)

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
