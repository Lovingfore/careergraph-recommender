# 轻量模型与推荐展示功能设计

## 背景

项目第一周的数据清洗、SQLite 建库、特征工程、职位—技能二部图和基础 Django 展示已经完成。后续需要补足一个适合课程设计的最小闭环：技能预测模型可以训练和评价，推荐结果可以通过接口和页面展示，同时保留跨设备部署的灵活性。

## 目标

1. 在现有数据和代码结构上增加一个可重复运行的轻量 TemporalGAT 训练流程。
2. 用时间顺序切分数据，输出验证/测试 MAE、RMSE，并与线性趋势 baseline 并列。
3. 提供不依赖 Django ORM 的 JSON 服务接口：推荐、技能缺口、职业路径和技能预测。
4. 在现有 Django 页面上增加最小交互和二部图展示。
5. 保持本地、Windows、Linux 和未来容器部署的可迁移性。

## 非目标

- 不构建生产级用户系统、登录系统或复杂前端框架。
- 不引入 Redis、消息队列、微服务或独立图数据库。
- 不把教学数据当作真实用户调查或真实招聘市场效果。
- 不把未训练模型的输出当作正式预测结果。
- 不在本阶段进行大规模超参数搜索。

## 设计

### 1. 模型训练

- 新增 `src/train_temporal_gat.py`，默认读取 `data/clean/`，使用现有 `build_dataloader` 和 `TemporalGAT`。
- 固定随机种子，按时间顺序使用早期窗口训练、中间窗口验证、最后窗口测试；当样本量不足时返回清晰的可运行错误。
- 使用 MSELoss、Adam 和小规模固定 epoch 配置，支持命令行参数覆盖。
- 输出 `artifacts/models/temporal_gat.pt`、`artifacts/models/training_log.csv`、`artifacts/models/evaluation_summary.json`。
- 同一脚本计算线性趋势 baseline 的 MAE/RMSE，结果中标注数据来源为课程教学数据。

### 2. 推荐服务

- 新增 `src/recommendation_service.py`，只依赖 CSV/SQLite 和已有特征计算逻辑。
- 暴露以下纯 Python 函数，方便命令行、Django 或未来 FastAPI 复用：
  - `recommend_for_user(user_id, top_k=5)`
  - `skill_gap_for_user(user_id, occupation_id)`
  - `career_path_for_user(user_id, occupation_id)`
  - `forecast_for_user(user_id, months=6)`
- 模型文件不存在或 PyTorch 不可用时，技能预测接口返回 `status` 和原因，不返回伪造深度模型结果；推荐仍可使用已完成的混合推荐特征。

### 3. Django 接口与页面

- 保留现有 `/` 和 `/api/summary/`。
- 新增：
  - `/api/recommend/?user_id=u001&top_k=5`
  - `/api/skill-gap/?user_id=u001&occupation_id=15-1251.00`
  - `/api/career-path/?user_id=u001&occupation_id=15-1251.00`
  - `/api/forecast/?user_id=u001&months=6`
- 页面使用原生 HTML、少量 JavaScript 和相对路径请求，不绑定特定设备尺寸或浏览器插件。
- 二部图以静态 SVG/PNG 展示，文件来自已生成的 `data/processed/bipartite/`。

### 4. 测试与验收

- 先为训练脚本、服务函数和新接口写失败测试，再实现最小代码。
- PyTorch 不可用时，非模型测试仍必须通过；模型测试明确标记为可选依赖测试。
- 最终运行 `python -m unittest discover -s tests -v`、`python src/verify_stage1.py` 和一次训练/评价命令。
- 验收结果区分：基础流水线通过、模型训练通过、Web 接口通过、可选依赖缺失。

## 部署约束

- 所有路径通过项目根目录计算或命令行参数传入，不写死当前电脑用户名。
- 默认使用 SQLite 和本地文件，便于复制到其他设备运行。
- 依赖拆分为基础依赖和可选模型依赖；没有 PyTorch 时网站仍能展示基础推荐与数据状态。

