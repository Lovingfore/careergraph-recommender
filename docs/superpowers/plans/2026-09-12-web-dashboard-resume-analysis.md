# Web Dashboard and Resume Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a lightweight Django dashboard that exposes model/data status, supports dynamic recommendation and career-path planning, and analyzes uploaded text resumes in memory without persisting them.

**Architecture:** Keep the existing CSV/SQLite data pipeline as the canonical source. Add a pure-Python temporary resume analysis module that converts text into a skill vector and reuses the existing hybrid scoring formulas against the occupation/skill and transition CSVs. Add thin Django JSON endpoints and a no-build vanilla HTML/CSS/JavaScript dashboard; the upload endpoint never writes to canonical files or SQLite.

**Tech Stack:** Python 3.12, pandas, NumPy, Django, existing SQLite/CSV services, python-docx for the report, LibreOffice renderer for document QA.

**Spec:** `docs/superpowers/specs/2026-09-12-web-dashboard-resume-analysis-design.md`

## Global Constraints

- Upload only UTF-8 `.txt`, `.md`, or `.csv` files no larger than 1 MB.
- Uploaded text is analyzed in request memory only; the response must include `persisted=false` and no database/CSV write may occur.
- Do not add login, ORM tables, external CDN, microservices, or a frontend build system.
- Preserve all existing GET API contracts and the 28-test baseline.
- Label all metrics as deterministic synthetic teaching-data results; do not claim TemporalGAT beats the linear baseline.
- Use project-relative paths and keep PyTorch optional for the dashboard.

---

### Task 1: Add failing tests for temporary resume analysis and new Web contracts

**Files:**
- Create: `tests/test_resume_analysis.py`
- Modify: `tests/test_web.py`

**Interfaces:**
- Tests will require `src.resume_analysis.analyze_resume_text(text, root, current_job, target_job)` to return a JSON-safe result with `skill_count`, `recognized_skills`, `recommendations`, `skill_gap`, `career_path`, and `persisted=False`.
- Tests will require `GET /api/model-info/` and `GET /api/occupations/`.
- Tests will require `POST /api/resume-upload/` multipart behavior and unchanged SQLite counts.

- [ ] **Step 1: Write focused failing tests**

Add tests that upload a `SimpleUploadedFile("resume.txt", "我熟悉 Python、数据库、机器学习和系统分析".encode("utf-8"))`, assert HTTP 200, `status == "ok"`, `persisted is False`, non-empty recommendations, and `missing_skill_count`/career path fields. Record `get_counts()` before and after the request and assert equality. Add tests for `.exe`, empty text, and a 1 MB+ payload returning 400. Add model-info field assertions (`status`, `epochs`, `sample_count`, `baseline`, `temporal_gat`) and index text assertions for upload/model sections.

- [ ] **Step 2: Run only the new tests to verify the expected RED state**

Run:

```powershell
D:\python\python.exe -m unittest tests.test_resume_analysis tests.test_web -v
```

Expected: FAIL because `src.resume_analysis`, `/api/model-info/`, `/api/occupations/`, and `/api/resume-upload/` do not exist yet.

---

### Task 2: Implement pure in-memory resume parsing and dynamic scoring

**Files:**
- Create: `src/resume_analysis.py`
- Modify: `src/recommendation_service.py`
- Test: `tests/test_resume_analysis.py`

**Interfaces:**
- `analyze_resume_text(text: str, root: Path | None = None, current_job: str | None = None, target_job: str | None = None) -> dict[str, Any]` validates input, maps Chinese/English skill aliases to the 35 skill IDs, computes dynamic ranking/skill gaps/path, and returns `persisted: False`.
- `recommend_for_skill_vector(skill_levels: dict[str, float], current_job: str, target_job: str | None, root: Path | None = None, top_k: int = 5) -> dict[str, Any]` computes recommendation rows using the existing match/gap/growth/path formula.
- `skill_gap_for_vector(...)` and `career_path_for_jobs(...)` provide reusable pure helpers; no helper writes files.

- [ ] **Step 1: Implement alias mapping and validation**

Read `skills.csv`, map each O*NET English name plus `SKILL_NAMES_ZH` Chinese name to its skill ID, normalize case/whitespace, reject empty text, and assign a bounded level (0.75 for matched aliases, 0.10 for unmatched skills). Validate `current_job` and `target_job` against `occupations.csv`.

- [ ] **Step 2: Implement dynamic recommendation and gap calculations**

Load `job_feature_matrix.csv`, `occupation_skill.csv`, `occupations.csv`, and `transition_graph.json`. Reuse the documented formula `0.45 Match - 0.25 Gap + 0.15 Growth + 0.15 Path`; exclude the current job, sort by score, and include named jobs, `is_target`, missing counts, and path probabilities.

- [ ] **Step 3: Implement bounded career-path search**

Perform the existing maximum-three-hop breadth-first search over `transition_graph.json`; return `path`, `path_names`, `path_probability`, and `path_length`, or an empty path with `path_length=-1` when no route exists.

- [ ] **Step 4: Run the focused tests and refactor only after GREEN**

Run:

```powershell
D:\python\python.exe -m unittest tests.test_resume_analysis -v
```

Expected: PASS, including the assertion that no files/database counts change.

---

### Task 3: Add Django model/occupation/upload APIs

**Files:**
- Modify: `web/topic17_app/views.py`
- Modify: `web/topic17_app/urls.py`
- Test: `tests/test_web.py`

**Interfaces:**
- `model_info_api(request)` serves `GET /api/model-info/` with training metadata and `status=unavailable` if artifacts are absent.
- `occupations_api(request)` serves `GET /api/occupations/` with occupation IDs and names for select controls.
- `resume_upload_api(request)` serves `POST /api/resume-upload/`, enforces extension/size/UTF-8/empty-text checks, calls `analyze_resume_text`, and returns HTTP 400 for invalid input.

- [ ] **Step 1: Implement model-info and occupations read-only endpoints**

Read JSON/CSV files using `BASE_DIR`; never import torch. Add model information to `_summary()` so the dashboard can render it from one response as well.

- [ ] **Step 2: Implement multipart upload validation and response**

Use `request.FILES.get("resume")`, allow `.txt/.md/.csv`, cap bytes at `1_048_576`, decode with `utf-8-sig`, require non-empty stripped text, accept optional `current_job`/`target_job` form fields, and pass only the decoded string to `analyze_resume_text`.

- [ ] **Step 3: Add URL routes and run Web tests**

Run:

```powershell
D:\python\python.exe -m unittest tests.test_web -v
```

Expected: all existing and new Web tests pass.

---

### Task 4: Replace the minimal page with the visual dashboard

**Files:**
- Modify: `web/topic17_app/templates/topic17_app/index.html`
- Test: `tests/test_web.py`

**Interfaces:**
- The page must contain dataset/database metric cards, a model comparison card, a current-user query form, an upload form, current/target occupation selects, recommendation and gap tables, path display, and the bipartite graph.

- [ ] **Step 1: Build semantic layout and responsive CSS**

Use a single self-contained template with no CDN. Add cards for resumes/users/events, database table rows, model metrics, warning text that metrics use synthetic teaching data, and visually distinct result panels.

- [ ] **Step 2: Add vanilla JavaScript data loading**

Fetch `/api/occupations/` to populate selects, `/api/model-info/` for the model card, retain the existing user query actions, submit `FormData` to `/api/resume-upload/`, and render recommendation/gap/path/recognized-skills results as accessible tables and text blocks. Display API errors inline.

- [ ] **Step 3: Run Web smoke tests and manually request the page**

Run:

```powershell
D:\python\python.exe -m unittest tests.test_web -v
D:\python\python.exe -c "import os; os.environ['DJANGO_SETTINGS_MODULE']='web.topic17_web.settings'; import django; django.setup(); from django.test import Client; r=Client().get('/'); print(r.status_code, len(r.content))"
```

Expected: HTTP 200 and all Web tests pass.

---

### Task 5: Create the final Chinese Word report

**Files:**
- Create: `src/create_final_report.py`
- Create: `docs/reports/CareerGraph_Recommender_Final_Report.docx`
- Create: `docs/reports/final_report_render/` (render QA intermediates, not required for runtime)

**Interfaces:**
- `src/create_final_report.py` reads current dataset/model/evaluation files and writes a self-contained Chinese report with no hard-coded stale metrics.

- [ ] **Step 1: Generate report content from current artifacts**

Create a styled report with title/summary, architecture and data flow, 300-resume dataset and six SQLite table counts, feature engineering and bipartite graph, recommendation formula, TemporalGAT vs baseline table, Web API/UI walkthrough, upload no-persistence guarantee, test/acceptance evidence, startup commands, limitations, and future work. State that baseline outperforms the current small TemporalGAT on this teaching dataset.

- [ ] **Step 2: Render the DOCX and inspect every page**

Run the documents skill renderer:

```powershell
D:\python\python.exe C:\Users\Lovin\.codex\plugins\cache\openai-primary-runtime\documents\26.819.11345\skills\documents\render_docx.py docs\reports\CareerGraph_Recommender_Final_Report.docx --output_dir docs\reports\final_report_render --emit_pdf
```

Inspect every generated `page-*.png` with the image viewer; fix clipping, table overflow, missing Chinese glyphs, or awkward spacing and render again until clean.

- [ ] **Step 3: Add final report link to README and run document checks**

Add the report to the README deliverables and run a DOCX text/page-count check plus the full test suite.

---

### Task 6: Full verification and handoff

**Files:**
- Modify: `README.md`
- Test: `tests/`

- [ ] **Step 1: Run full test suite**

```powershell
D:\python\python.exe -m unittest discover -s tests -v
```

Expected: 0 failures.

- [ ] **Step 2: Run stage acceptance and verify non-persistence**

```powershell
D:\python\python.exe src\verify_stage1.py --db-path artifacts\topic17.sqlite3
D:\python\python.exe -c "import sqlite3; c=sqlite3.connect('artifacts/topic17.sqlite3'); print(c.execute('SELECT COUNT(*) FROM user_profiles').fetchone()[0]); c.close()"
```

Expected: all critical acceptance sections `ok=true`, 300 user profiles, and no upload-created rows.

- [ ] **Step 3: Review `git diff --check` and report exact artifacts**

Confirm no whitespace errors, list the dashboard routes and final DOCX path, and report model metrics with the baseline caveat.
