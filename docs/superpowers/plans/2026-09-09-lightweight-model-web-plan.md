# 轻量模型与推荐展示功能实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有课程项目上补齐一个轻量、可迁移的技能预测训练闭环和推荐展示接口。

**Architecture:** 保留现有 CSV、SQLite、Django 和混合推荐结构。新增一个可选 PyTorch 训练脚本和一个纯 Python 推荐服务；Django 只做参数解析、JSON 返回和简单页面展示，不引入 ORM 新表或复杂前端框架。

**Tech Stack:** Python 3.12、pandas、NumPy、SQLite、PyTorch（可选）、Django、unittest。

**Spec:** `docs/superpowers/specs/2026-09-09-lightweight-model-web-design.md`

## Global Constraints

- 不构建生产级用户系统、登录系统或复杂前端框架。
- 不引入 Redis、消息队列、微服务或独立图数据库。
- 不把教学数据当作真实用户调查或真实招聘市场效果。
- 不把未训练模型的输出当作正式预测结果。
- 所有路径通过项目根目录计算或命令行参数传入，不写死当前电脑用户名。
- 依赖拆分为基础依赖和可选模型依赖；没有 PyTorch 时网站仍能展示基础推荐与数据状态。

---

### Task 1: 训练与评价工具

**Files:**
- Create: `src/train_temporal_gat.py`
- Create: `tests/test_temporal_gat_training.py`
- Modify: `requirements-optional-models.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: `src.data_loader.build_dataloader`, `src.temporal_gat.TemporalGAT`, `data/clean/user_skill_events.csv`。
- Produces: `run_training(data_dir, artifact_dir, epochs=30, seed=42) -> dict`；命令行输出 `artifacts/models/temporal_gat.pt`、`training_log.csv`、`evaluation_summary.json`。

- [ ] **Step 1: Write the failing test**

  在 `tests/test_temporal_gat_training.py` 中增加一个 PyTorch 可用时运行的测试：构造临时输出目录，调用 `run_training(..., epochs=2)`，断言返回值含 `baseline`、`temporal_gat`、`test`，并断言模型和 JSON 文件存在；PyTorch 不可用时使用 `skipUnless`。

- [ ] **Step 2: Run test to verify it fails**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_temporal_gat_training -v`

  Expected: FAIL with `ModuleNotFoundError: No module named 'src.train_temporal_gat'`。

- [ ] **Step 3: Write minimal implementation**

  在 `src/train_temporal_gat.py` 中实现：固定 `random`、NumPy 和 PyTorch 种子；读取序列数据；按最后两个时间窗口做验证和测试，其余窗口训练；用 `MSELoss`、`Adam(lr=0.01)` 和命令行可覆盖的 epoch；记录每轮 loss；保存 `state_dict` 和包含模型参数的 checkpoint；计算线性趋势和 TemporalGAT 的 MAE/RMSE；PyTorch 缺失时抛出包含安装命令的明确错误。

- [ ] **Step 4: Run test to verify it passes**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_temporal_gat_training -v`

  Expected: PyTorch 已安装时 PASS；未安装时测试显示为 skipped，而其他测试不受影响。

- [ ] **Step 5: Commit**

  `git add src/train_temporal_gat.py tests/test_temporal_gat_training.py requirements-optional-models.txt README.md; git commit -m "feat: add lightweight temporal gat training"`

### Task 2: 纯 Python 推荐服务

**Files:**
- Create: `src/recommendation_service.py`
- Create: `tests/test_recommendation_service.py`

**Interfaces:**
- Consumes: `data/processed/recommendation_features.csv`、`data/processed/skill_forecast_6m.csv`、`src.database.query_user_profile`。
- Produces: `recommend_for_user(user_id, top_k=5) -> dict`、`skill_gap_for_user(user_id, occupation_id) -> dict`、`career_path_for_user(user_id, occupation_id) -> dict`、`forecast_for_user(user_id, months=6) -> dict`。

- [ ] **Step 1: Write the failing test**

  测试 `u001` 的推荐结果按 `rank` 升序且数量不超过 `top_k`；测试缺失用户返回 `status="error"`；测试技能缺口包含职位 ID 和技能列表；测试没有模型文件时预测返回 `status="baseline"` 或 `status="unavailable"`，不得伪造 `temporal_gat` 结果。

- [ ] **Step 2: Run test to verify it fails**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_recommendation_service -v`

  Expected: FAIL with `ModuleNotFoundError: No module named 'src.recommendation_service'`。

- [ ] **Step 3: Write minimal implementation**

  用 pandas 读取已有结果，集中处理用户/职位不存在、`top_k` 非法和空结果；缺口由 `gap_score`、`missing_skill_count` 和现有职位技能表生成；路径从 `transition_graph.json` 做最多三跳 BFS；预测默认返回已有线性 baseline 文件，只有检测到训练 checkpoint 时才标记为 `temporal_gat`。

- [ ] **Step 4: Run test to verify it passes**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_recommendation_service -v`

  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add src/recommendation_service.py tests/test_recommendation_service.py; git commit -m "feat: add portable recommendation service"`

### Task 3: Django API 与最小页面交互

**Files:**
- Modify: `web/topic17_app/views.py`
- Modify: `web/topic17_app/urls.py`
- Modify: `web/topic17_app/templates/topic17_app/index.html`
- Modify: `tests/test_web.py`

**Interfaces:**
- Consumes: `src.recommendation_service` 四个纯 Python 函数和现有 `_summary()`。
- Produces: `/api/recommend/`、`/api/skill-gap/`、`/api/career-path/`、`/api/forecast/`；页面提供用户 ID、职位 ID 输入和结果区域。

- [ ] **Step 1: Write the failing test**

  在 `tests/test_web.py` 增加 GET 请求测试：推荐接口返回 200 和 `recommendations`；缺口接口返回 200 和 `skills`；路径接口返回 200 和 `path`；预测接口返回 200 且包含 `status`；缺失参数返回 400。

- [ ] **Step 2: Run test to verify it fails**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_web -v`

  Expected: FAIL because the four routes are not defined。

- [ ] **Step 3: Write minimal implementation**

  在 views 中解析查询参数并将服务层异常转为 400/404 JSON；在 urls 中注册四个路径；在模板中加入原生表单、fetch 调用和二部图 SVG 的相对路径展示，不依赖外部 CDN。

- [ ] **Step 4: Run test to verify it passes**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_web -v`

  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add web tests/test_web.py; git commit -m "feat: expose recommendation APIs and demo UI"`

### Task 4: 文档、整体验收与部署说明

**Files:**
- Modify: `README.md`
- Create: `docs/reports/CareerGraph_Recommender_Week2_Report.md`
- Modify: `run_all.ps1`
- Modify: `src/verify_stage1.py`

**Interfaces:**
- Consumes: 任务 1-3 的输出文件、接口和测试结果。
- Produces: 一键运行说明、第二周阶段报告、模型状态验收字段和跨设备部署说明。

- [ ] **Step 1: Write the failing test**

  在 `tests/test_stage1_verification.py` 增加对验收 JSON 中 `model` 和 `web_api` 字段存在性的测试，并断言没有模型时状态为 `optional_dependency_unavailable` 或 `not_trained`。

- [ ] **Step 2: Run test to verify it fails**

  Run: `\.venv\Scripts\python.exe -m unittest tests.test_stage1_verification -v`

  Expected: FAIL because the new report fields are absent。

- [ ] **Step 3: Write minimal implementation**

  扩展 `verify_stage1.py` 检查模型文件、训练摘要和四个 Web 路由；更新 `run_all.ps1` 让基础流水线默认可运行、模型训练通过显式开关执行；README 写明 Windows/Linux 命令、可选 PyTorch、环境变量 `TOPIC17_DB_PATH` 和教学数据限制；报告记录完成项、测试证据和未完成项。

- [ ] **Step 4: Run test to verify it passes**

  Run: `\.venv\Scripts\python.exe -m unittest discover -s tests -v; \.venv\Scripts\python.exe src\verify_stage1.py --db-path artifacts\topic17.sqlite3`

  Expected: all mandatory tests pass, optional PyTorch tests are either passed or clearly skipped, and verification exits 0。

- [ ] **Step 5: Commit**

  `git add README.md docs/reports/CareerGraph_Recommender_Week2_Report.md run_all.ps1 src/verify_stage1.py tests/test_stage1_verification.py; git commit -m "docs: add week two verification and deployment notes"`

