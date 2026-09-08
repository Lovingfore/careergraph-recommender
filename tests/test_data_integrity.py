from pathlib import Path
import unittest

from src.data_integrity import validate_clean_dataset, validate_feature_outputs


ROOT = Path(__file__).resolve().parents[1]


class DatasetIntegrityTests(unittest.TestCase):
    def test_clean_tables_have_no_orphan_ids(self):
        result = validate_clean_dataset(ROOT / "data" / "clean")
        self.assertEqual(result["orphan_occupation_ids"], [])
        self.assertEqual(result["orphan_skill_ids"], [])
        self.assertEqual(result["probability_sum_errors"], [])

    def test_feature_outputs_cover_all_profile_targets(self):
        result = validate_feature_outputs(ROOT / "data" / "clean", ROOT / "data" / "processed")
        self.assertEqual(result["missing_target_jobs"], [])
        self.assertEqual(result["feature_occupation_ids"], result["skill_occupation_ids"])


if __name__ == "__main__":
    unittest.main()
