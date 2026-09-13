from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentScriptTests(unittest.TestCase):
    def test_cross_platform_entrypoints_exist(self):
        for name in ("deploy.ps1", "deploy.bat", "deploy.sh"):
            with self.subTest(name=name):
                self.assertTrue((PROJECT_ROOT / name).is_file())

    def test_windows_deployer_is_portable_and_exposes_expected_switches(self):
        content = (PROJECT_ROOT / "deploy.ps1").read_text(encoding="utf-8")
        self.assertIn("[switch]$TrainModel", content)
        self.assertIn("[switch]$StartWeb", content)
        self.assertIn("requirements.txt", content)
        self.assertIn("run_all.ps1", content)
        self.assertIn("-SkipInstall", content)
        self.assertNotRegex(content, re.compile(r"[A-Za-z]:\\Project_all\\"))
        self.assertNotIn("Lovin", content)

    def test_unix_deployer_uses_project_relative_paths_and_same_pipeline(self):
        content = (PROJECT_ROOT / "deploy.sh").read_text(encoding="utf-8")
        self.assertIn("--train-model", content)
        self.assertIn("--start-web", content)
        self.assertIn("requirements.txt", content)
        for step in (
            "src/prepare_dataset.py",
            "src/forecast_skills.py",
            "src/build_features.py",
            "src/evaluate_recommendation.py",
            "src/init_database.py",
            "src/bipartite_graph.py",
            "src/verify_stage1.py",
        ):
            self.assertIn(step, content)
        self.assertNotRegex(content, re.compile(r"[A-Za-z]:\\Project_all\\"))
        self.assertNotIn("Lovin", content)
        self.assertIn("SKIP_INSTALL == 1", content)

    def test_windows_double_click_wrapper_forwards_arguments(self):
        content = (PROJECT_ROOT / "deploy.bat").read_text(encoding="utf-8")
        self.assertIn("ExecutionPolicy Bypass", content)
        self.assertIn("deploy.ps1", content)
        self.assertIn("-StartWeb", content)
        self.assertIn("%*", content)

    def test_existing_pipeline_can_skip_dependency_installation(self):
        content = (PROJECT_ROOT / "run_all.ps1").read_text(encoding="utf-8")
        self.assertIn("[switch]$SkipInstall", content)
        self.assertIn("if (-not $SkipInstall)", content)


if __name__ == "__main__":
    unittest.main()
