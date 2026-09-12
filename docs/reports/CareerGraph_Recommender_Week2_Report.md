# CareerGraph Recommender 第二周轻量补充报告

## 本周目标

在第一周“数据清洗—SQLite—特征工程—二部图—基础 Web”闭环的基础上，补齐一个适合课程设计的小型模型训练、推荐服务和跨设备展示闭环。

## 已完成任务

### 1. TemporalGAT 轻量训练

- 新增 `src/train_temporal_gat.py`。
- 使用现有 `SkillSequenceDataset` 生成连续 3 个月窗口，当前共 900 个样本、35 项技能；样本来自 300 条合成中文技术岗简历。
- 按时间顺序切分：训练 540 个窗口、验证 180 个窗口、测试 180 个窗口。
- 使用固定随机种子 42、MSELoss、Adam 和 30 轮小规模训练。
- 输出：
  - `artifacts/models/temporal_gat.pt`
  - `artifacts/models/training_log.csv`
  - `artifacts/models/evaluation_summary.json`

### 2. baseline 对照

本次测试使用的是确定性教学数据，不代表真实用户预测效果。

| 方法 | Test MAE | Test RMSE |
|---|---:|---:|
| 线性趋势 baseline | 0.017969 | 0.022436 |
| TemporalGAT（30 轮） | 0.116501 | 0.139666 |

当前小样本下 baseline 优于未充分调参的 TemporalGAT，因此项目不会宣称深度模型效果更好。该结果可作为后续增加真实数据和调参的依据。

### 3. 纯 Python 推荐服务

新增 `src/recommendation_service.py`，不依赖 Django ORM，可被命令行、Django 或未来 FastAPI 复用：

- `recommend_for_user(user_id, top_k=5)`：返回混合推荐排序；
- `skill_gap_for_user(user_id, occupation_id)`：返回需求技能、当前水平和缺口；
- `career_path_for_user(user_id, occupation_id)`：返回最多三跳职业路径；
- `forecast_for_user(user_id, months=6)`：返回线性趋势 baseline，并明确标记 `status=baseline`。

### 4. Django API 与页面

新增接口：

- `/api/model-info/`
- `/api/occupations/`
- `/api/resume-upload/`（UTF-8 `.txt/.md/.csv`，≤1 MB，仅内存分析，不落库）
- `/api/recommend/`
- `/api/skill-gap/`
- `/api/career-path/`
- `/api/forecast/`
- `/bipartite-graph.svg`

首页增加模型状态卡、数据库统计、基准用户查询区和“上传简历：即时分析（不落库）”区，可输入/选择当前职位与目标职位，查看动态推荐、技能缺口、补全时间和职业路径，并显示职位—技能二部图。页面不依赖外部 CDN 或固定设备插件。

### 5. 跨设备部署处理

- 所有数据路径从项目根目录计算，或通过 `TOPIC17_DB_PATH` 覆盖数据库位置；
- SQLite、CSV 和静态 SVG 保留在项目目录内；
- 基础依赖和 PyTorch 可选依赖分离；
- `run_all.ps1` 默认只执行基础流水线，使用 `-TrainModel` 才执行模型训练；
- 训练脚本同时支持模块方式和直接脚本方式启动。

## 验收证据

使用系统 Python（含 PyTorch 2.5.1）执行：

```text
Ran 37 tests
OK
```

阶段验收确认：

- 数据完整性通过；
- SQLite 六张核心表和行数通过；
- 特征输出覆盖全部职位；
- 二部图 12 个职位、35 个技能、420 条边通过；
- DataLoader 输出 `[2, 3, 35]` 通过；
- 模型 checkpoint、日志和评价摘要已生成；
- Django 首页、summary、推荐、缺口、路径、预测和二部图路由测试通过。
- 模型信息、职位字典和简历上传接口测试通过；上传前后 SQLite 六张表行数保持不变。
- 上传 POST 启用 Django CSRF 保护，并限制单文件 1 MB、请求体 2 MB；基准 CSV 哈希保持不变。

## 当前限制

1. 现有用户技能事件和职业转移仍是确定性教学数据。
2. 当前 TemporalGAT 只是小样本教学模型，测试结果不具备真实招聘市场泛化意义。
3. 还没有复杂用户登录、实时数据库、生产部署和大规模性能压测。
4. 当前版本已另行生成完整中文 Word 汇报：`CareerGraph_Recommender_Final_Report.docx`。

## 建议后续工作

优先准备真实或公开职业流动数据，再考虑增加训练样本、完善测试集和调参；如果只是完成课程演示，当前轻量闭环已经足够支撑推荐、缺口、路径、预测和二部图展示。
