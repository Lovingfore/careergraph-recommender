# 第17题第一周验收补齐实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有数据清洗流水线基础上补齐第一周要求的职位数据一致性、SQLite 数据库、PyTorch DataLoader、最小 Django 展示和时序图注意力模型骨架。

**Architecture:** 保留现有 CSV 作为可审计的数据交换层，新增共享的完整性检查、SQLite 导入/查询服务和 PyTorch 数据接口。Django 只调用共享服务，不重复解析 CSV；线性趋势预测保留为 baseline，TemporalGAT 只实现可前向传播的最小结构，训练调参留到第二周。

**Tech Stack:** Python 3.12, pandas, numpy, SQLite, PyTorch, Django 5, unittest, PowerShell.

**Spec:** `docs/superpowers/specs/2026-09-08-topic17-stage1-design.md`

## Global Constraints

- 不修改 `data/raw/onet_30_2` 原始文本和原始压缩包。
- `user_skill_events.csv` 和 `job_transitions.csv` 保留 `source=derived_teaching_data`，不得表述为真实调查数据。
- 职位主表、职位-技能关系、特征矩阵和用户目标职位必须使用同一套有效职位 ID。
- 数据库导入失败必须回滚并返回非零退出码。
- Web 只调用共享数据库和 DataLoader 服务，不复制 CSV 解析逻辑。
- 每个新增行为先写失败测试，再实现最小代码。

---

### Task 1: 建立测试入口与职位数据一致性约束

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_data_integrity.py`
- Create: `src/data_integrity.py`
- Modify: `src/prepare_dataset.py:19-32, 74-90, 158-190`

**Interfaces:**
- `validate_clean_dataset(data_dir: Path) -> dict`
- `validate_feature_outputs(data_dir: Path, processed_dir: Path) -> dict`
- `prepare_dataset.py` 生成数据后调用校验函数，失败时抛出 `ValueError`。

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_data_integrity -v`

Expected: FAIL because `src.data_integrity` does not exist and current generated outputs omit two occupation IDs.

- [ ] **Step 3: Write minimal implementation**

Implement CSV foreign-key checks, duplicate pair checks, value-range checks, transition probability sums, and target/feature coverage. Replace the hardcoded invalid O*NET skill codes with an explicit supported-code mapping; keep the display occupation names stable and reject a selected occupation with no skill rows. Call the checks at the end of dataset generation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python src/prepare_dataset.py` then `python src/build_features.py` then `python -m unittest tests.test_data_integrity -v`

Expected: PASS; all selected occupations have skill rows and all profile targets are candidate occupations.

- [ ] **Step 5: Commit**

No Git commit is possible because the supplied directory is not a Git repository; record the verified file changes in the final report.

### Task 2: Add SQLite schema, transactional importer, and query service

**Files:**
- Create: `src/database.py`
- Create: `src/init_database.py`
- Create: `tests/test_database.py`
- Modify: `requirements.txt`
- Modify: `run_all.ps1`

**Interfaces:**
- `SCHEMA_SQL: str`
- `initialize_database(db_path: Path) -> None`
- `load_csv_data(db_path: Path, data_dir: Path) -> dict[str, int]`
- `get_counts(db_path: Path) -> dict[str, int]`
- `query_user_profile(db_path: Path, user_id: str) -> dict`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import sqlite3
import unittest
from src.database import initialize_database, load_csv_data, get_counts, query_user_profile

ROOT = Path(__file__).resolve().parents[1]

class DatabaseTests(unittest.TestCase):
    def test_database_has_six_core_tables_and_imported_rows(self):
        db_path = ROOT / "work" / "test_topic17.sqlite3"
        initialize_database(db_path)
        counts = load_csv_data(db_path, ROOT / "data" / "clean")
        self.assertEqual(set(counts), {"occupations", "skills", "occupation_skill", "user_profiles", "user_skill_events", "job_transitions"})
        self.assertGreater(counts["occupations"], 0)
        self.assertGreater(counts["user_skill_events"], 0)
        with sqlite3.connect(db_path) as conn:
            tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
        self.assertTrue({"occupations", "skills", "occupation_skill", "user_profiles", "user_skill_events", "job_transitions"} <= tables)

    def test_query_returns_user_profile_and_events(self):
        db_path = ROOT / "work" / "test_topic17_query.sqlite3"
        initialize_database(db_path)
        load_csv_data(db_path, ROOT / "data" / "clean")
        profile = query_user_profile(db_path, "u001")
        self.assertEqual(profile["user_id"], "u001")
        self.assertGreater(profile["event_count"], 0)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_database -v`

Expected: FAIL because database functions do not exist.

- [ ] **Step 3: Write minimal implementation**

Create six tables with primary keys, foreign keys, indexes, and `PRAGMA foreign_keys=ON`. Import in dependency order inside one transaction; normalize CSV column names to the canonical schema; run post-import checks before commit. `init_database.py` accepts `--db-path` and `--data-dir`, initializes and loads the database, then prints JSON counts.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_database -v` and `python src/init_database.py --db-path artifacts/topic17.sqlite3 --data-dir data/clean`

Expected: PASS and a database file containing non-zero rows in all six tables.

- [ ] **Step 5: Commit**

No Git commit; keep the generated database under `artifacts/` as the stage deliverable.

### Task 3: Add PyTorch sequence DataLoader, graph tensors, and TemporalGAT smoke model

**Files:**
- Create: `src/data_loader.py`
- Create: `src/temporal_gat.py`
- Create: `tests/test_data_loader.py`
- Modify: `requirements.txt`
- Modify: `requirements-optional-models.txt`

**Interfaces:**
- `SkillSequenceDataset(events: DataFrame, skill_ids: list[str], sequence_length: int = 3)`
- `build_dataloader(data_dir: Path, batch_size: int = 2, sequence_length: int = 3)`
- `build_graph_tensors(data_dir: Path) -> dict`
- `TemporalGAT(num_skills: int, hidden_dim: int = 32, heads: int = 2)`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.data_loader import build_graph_tensors

ROOT = Path(__file__).resolve().parents[1]

class LoaderTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch not installed in test environment")
    def test_dataloader_batch_has_expected_shapes(self):
        from src.data_loader import build_dataloader
        loader, meta = build_dataloader(ROOT / "data" / "clean", batch_size=2, sequence_length=3)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["x"].shape), (2, 3, meta["num_skills"]))
        self.assertEqual(tuple(batch["y"].shape), (2, meta["num_skills"]))

    def test_graph_tensors_have_edges(self):
        graph = build_graph_tensors(ROOT / "data" / "clean")
        self.assertGreater(graph["occupation_skill_edge_index"].shape[1], 0)
        self.assertGreater(graph["transition_edge_index"].shape[1], 0)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_data_loader -v`

Expected: FAIL for missing loader functions; the PyTorch-specific test may be skipped if the dependency is not installed.

- [ ] **Step 3: Write minimal implementation**

Implement contiguous month windows from `user_skill_events.csv`, zero-fill missing skill/month combinations, and return dictionaries with `x`, `y`, and `user_id`. Build stable ID maps and `torch.long` edge indices with `torch.float32` edge weights. Implement TemporalGAT with a temporal GRU encoder followed by multi-head neighbor aggregation using the occupation-skill and transition edge lists; keep the forward signature `forward(x, edge_index=None, edge_weight=None) -> Tensor`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_data_loader -v`; if PyTorch is available, run a one-batch forward smoke test for `TemporalGAT`.

Expected: graph edge tests pass; DataLoader and model shape tests pass when PyTorch is installed, otherwise the test output clearly reports the dependency skip.

- [ ] **Step 5: Commit**

No Git commit; record whether the local environment had PyTorch available.

### Task 4: Add minimal Django Web page/API using shared services

**Files:**
- Create: `web/manage.py`
- Create: `web/topic17_web/__init__.py`
- Create: `web/topic17_web/settings.py`
- Create: `web/topic17_web/urls.py`
- Create: `web/topic17_web/wsgi.py`
- Create: `web/topic17_app/__init__.py`
- Create: `web/topic17_app/apps.py`
- Create: `web/topic17_app/urls.py`
- Create: `web/topic17_app/views.py`
- Create: `web/topic17_app/templates/topic17_app/index.html`
- Create: `tests/test_web.py`
- Modify: `run_all.ps1`

**Interfaces:**
- `GET /`
- `GET /api/summary/`
- Web views call `src.database.get_counts`, `src.database.query_user_profile`, and `src.data_loader.build_dataloader`.

- [ ] **Step 1: Write the failing test**

```python
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

    def test_index_page_is_reachable(self):
        from django.test import Client
        response = Client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Topic 17")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_web -v`

Expected: FAIL because the Django project does not exist.

- [ ] **Step 3: Write minimal implementation**

Create a Django project without a second ORM schema; use the shared SQLite query service. The API returns counts, integrity status, and either a DataLoader batch summary or a clear `torch_not_installed` status. The HTML page displays the same values in a simple table and includes the exact command needed to initialize the database.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_web -v` with Django installed and an initialized `artifacts/topic17.sqlite3`.

Expected: both routes return 200 and show database counts plus loader status.

- [ ] **Step 5: Commit**

No Git commit; keep the Django project under `web/`.

### Task 5: Align documentation and create the final verification command

**Files:**
- Modify: `README.md`
- Modify: `ALGORITHM_SELECTION.md`
- Modify: `run_all.ps1`
- Create: `src/verify_stage1.py`
- Create: `tests/test_stage1_verification.py`

**Interfaces:**
- `python src/verify_stage1.py --db-path artifacts/topic17.sqlite3`
- Verification JSON fields: `data_integrity`, `database`, `feature_outputs`, `loader`, `web_contract`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import unittest
from src.verify_stage1 import verify_stage1

ROOT = Path(__file__).resolve().parents[1]

class Stage1VerificationTests(unittest.TestCase):
    def test_stage1_report_has_all_acceptance_sections(self):
        report = verify_stage1(ROOT, ROOT / "artifacts" / "topic17.sqlite3")
        self.assertEqual(set(report), {"data_integrity", "database", "feature_outputs", "loader", "web_contract"})
        self.assertTrue(report["data_integrity"]["ok"])
        self.assertTrue(report["database"]["ok"])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_stage1_verification -v`

Expected: FAIL because the stage verifier does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement a compact verifier that runs integrity checks, inspects all six database counts, checks processed files and targets, probes DataLoader availability, and validates that Django URL modules exist. Update README with the canonical commands, database schema, synthetic-data warning, and Web launch command. Update algorithm documentation to call linear trend the baseline and TemporalGAT the selected final model. Update `run_all.ps1` to run preparation, feature generation, database initialization, and stage verification.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`; then `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`; then start Django with `python web/manage.py runserver 127.0.0.1:8000` and request `/api/summary/`.

Expected: all non-PyTorch tests pass; PyTorch tests pass when dependency is installed; the stage verifier reports database and data integrity as `ok=true`.

- [ ] **Step 5: Commit**

No Git commit; provide the final verification JSON and exact launch commands to the user.
