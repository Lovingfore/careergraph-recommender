from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.database import initialize_database, load_csv_data, get_counts, query_user_profile


ROOT = Path(__file__).resolve().parents[1]


class DatabaseTests(unittest.TestCase):
    def test_database_has_six_core_tables_and_imported_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "topic17.sqlite3"
            initialize_database(db_path)
            counts = load_csv_data(db_path, ROOT / "data" / "clean")
            self.assertEqual(
                set(counts),
                {"occupations", "skills", "occupation_skill", "user_profiles", "user_skill_events", "job_transitions"},
            )
            self.assertGreater(counts["occupations"], 0)
            self.assertGreater(counts["user_skill_events"], 0)
            conn = sqlite3.connect(db_path)
            try:
                tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
            finally:
                conn.close()
            self.assertTrue(
                {"occupations", "skills", "occupation_skill", "user_profiles", "user_skill_events", "job_transitions"} <= tables
            )

    def test_query_returns_user_profile_and_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "topic17_query.sqlite3"
            initialize_database(db_path)
            load_csv_data(db_path, ROOT / "data" / "clean")
            profile = query_user_profile(db_path, "u001")
            self.assertEqual(profile["user_id"], "u001")
            self.assertGreater(profile["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
