# CareerGraph Recommender

基于技能演化图谱的动态职位推荐与职业发展路径规划系统。

## 本次交付内容

- `data/raw/README.md`：O*NET 30.2 原始数据下载和复现说明（原始大文件不纳入仓库）。
- `data/clean/`：筛选出的 12 个软件、数据和网络方向职位的清洗结果。
- `data/processed/`：职位向量、用户向量、用户-职位特征矩阵、转移图、Top-K 推荐和 6 个月技能预测。
- `src/prepare_dataset.py`：读取 O*NET 并完成清洗、归一化、教学数据生成。
- `src/build_features.py`：完成特征工程、混合推荐和职业路径特征。
- `src/forecast_skills.py`：用透明的线性趋势 baseline 预测未来 6 个月技能水平。
- `src/evaluate_recommendation.py`：计算 Precision@K、Recall@K 和 NDCG@K。
- `src/database.py`：提供六张核心 SQLite 表、事务导入、计数和用户画像查询服务。
- `src/init_database.py`：从 `data/clean/` 初始化 `artifacts/topic17.sqlite3`。
- `src/data_loader.py`：把用户技能事件整理为连续月窗口，并构造职位-技能/职位转移图边张量。
- `src/bipartite_graph.py`：从职位-技能关系构建带 `demand_weight` 权重的职位—技能二部图，并导出 JSON、CSV 和可视化文件。
- `src/temporal_gat.py`：时序 GRU + 多头邻居聚合的 TemporalGAT 最小可运行骨架，训练和调参安排在第二周。
- `src/train_temporal_gat.py`：轻量 TemporalGAT 训练、顺序切分、MAE/RMSE 评价和 checkpoint 输出。
- `src/recommendation_service.py`：可供命令行、Django 或未来其他 Web 框架复用的推荐、技能缺口、职业路径和技能预测服务。
- `src/verify_stage1.py`：输出第1周数据、数据库、特征、Loader 和 Web 契约验收 JSON。
- `web/`：不新增 ORM 表的最小 Django 展示，页面和 `/api/summary/` 共用上述服务。
- `docs/reports/CareerGraph_Recommender_Week1_Report.docx`：第一周阶段汇报，逐项对照计划表说明已完成任务、实现细节、验收证据和第二周边界。

运行后还会得到 `data/clean/clean_log.csv`，用于记录每一步清洗动作和行数变化；`data/processed/transition_graph.json` 保存职位转移图的可读版本。

基础流水线只需要 `requirements.txt`。如需执行 TemporalGAT 训练，再安装 `requirements-optional-models.txt`；受限网络环境可按设备选择 CPU-only PyTorch wheel。

## 数据说明

O*NET 职位和技能表是真实公开数据，来源为 O*NET 30.2 Database，2026 年 2 月发布，许可证为 CC BY 4.0：

<https://www.onetcenter.org/database.html>

为了让课程设计可运行且规模可控，本项目只抽取 12 个相关职位和 35 个通用技能。`user_skill_events.csv` 与 `job_transitions.csv` 是由选定 O*NET 职位向量和固定规则生成的确定性教学数据，不应在论文中表述为真实人员调查数据。后续如果拿到真实简历或职业转移数据，只需要替换这两个文件，特征工程接口不变。

## 一键运行

在项目根目录中执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/prepare_dataset.py
python src/forecast_skills.py
python src/build_features.py
python src/evaluate_recommendation.py
python src/init_database.py --db-path artifacts/topic17.sqlite3 --data-dir data/clean
python src/verify_stage1.py --db-path artifacts/topic17.sqlite3
```

轻量模型训练是可选步骤：

```powershell
python src/train_temporal_gat.py --data-dir data/clean --artifact-dir artifacts/models --epochs 30 --seed 42
# 或让一键脚本包含训练：
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -TrainModel
```

如果需要启动 Web 展示：

```powershell
python web/manage.py runserver 127.0.0.1:8000
```

然后访问 <http://127.0.0.1:8000/> 或 <http://127.0.0.1:8000/api/summary/>。

推荐演示接口：

- `/api/recommend/?user_id=u001&top_k=5`
- `/api/skill-gap/?user_id=u001&occupation_id=15-1244.00`
- `/api/career-path/?user_id=u001&occupation_id=15-1252.00`
- `/api/forecast/?user_id=u001&months=6`
- `/bipartite-graph.svg`

这些接口只依赖项目内的 CSV、SQLite 和相对路径，可复制到其他 Windows 或 Linux 设备后重新安装依赖运行。没有 PyTorch 时，基础推荐、技能缺口、职业路径和页面仍可使用；预测接口会明确标记线性趋势 baseline，不会伪装成深度模型输出。

SQLite 数据库包含 `occupations`、`skills`、`occupation_skill`、`user_profiles`、`user_skill_events` 和 `job_transitions` 六张核心表；导入启用外键约束并在一个事务内完成，失败会回滚。数据库仅由共享服务读写，Django 不重复解析 CSV。

## 核心特征

每个用户-候选职位组合生成以下特征：

- `match_score`：用户技能向量与职位需求向量的余弦相似度。
- `gap_score`：职位需求超过用户现有水平的加权技能缺口。
- `missing_skill_count`：需求较高但用户水平不足的技能数量。
- `direct_transition_probability`：当前职位到候选职位的直接转移概率。
- `path_probability`：职位转移图上最多三跳路径的最大概率乘积。
- `growth_score`：目标职位相对当前职位的平均技能要求提升。
- `estimated_training_hours`：依据缺口分数和缺口数量估算的补全时间。
- `recommendation_score`：`0.45 Match - 0.25 Gap + 0.15 Growth + 0.15 Path`。

## 职位—技能二部图

项目已从 `data/clean/occupation_skill.csv` 构建二部图：左侧为 12 个职位节点，右侧为 35 个技能节点，420 条边表示职位技能需求，边权为清洗后的 `demand_weight`。完整结构写入 `data/processed/bipartite/bipartite_graph.json`，边表写入 `bipartite_edges.csv`，统计摘要写入 `bipartite_graph_summary.json`。

运行命令：

```powershell
python src/bipartite_graph.py --data-dir data/clean --out-dir data/processed/bipartite --min-demand-weight 0.30
```

其中 `--min-demand-weight` 只控制可视化子图显示阈值，不会删除 JSON/CSV 中的完整 420 条边。当前阈值为 0.30 时显示 158 条高权重边；由于绘图库是可选依赖，当前环境会生成可直接在浏览器打开的 SVG 文件。

## 算法选择

第一版推荐采用“内容匹配 + 职位转移图”的混合推荐算法，优点是数据量小也能运行、结果容易解释，适合课程设计。技能预测保留线性趋势 baseline，同时在 `temporal_gat.py` 提供时序图注意力的最小前向接口；第二周再进行训练、损失设计和调参。这样可以先保证数据、特征、指标和 Web 接口闭环，再增加深度模型。

## 重要限制

该数据包用于课程项目原型和方法验证，不代表真实招聘市场规律，也不提供就业承诺。报告中应分别说明 O*NET 真实来源数据和本项目生成的教学数据。

## 阶段汇报

第一周“算法选择与数据库搭建”阶段汇报见 [`docs/reports/CareerGraph_Recommender_Week1_Report.docx`](docs/reports/CareerGraph_Recommender_Week1_Report.docx)。文档记录了数据清洗、SQLite 建库入库、图结构与时序特征、DataLoader、TemporalGAT 前向骨架、Django 验证结果，以及尚未提前宣称的训练与调参工作。

第二周轻量模型和 Web 展示补充说明见 [`docs/reports/CareerGraph_Recommender_Week2_Report.md`](docs/reports/CareerGraph_Recommender_Week2_Report.md)。
