# 第17题第一周验收补齐设计

## 目标

在现有 O*NET 数据清洗和推荐特征流水线基础上，补齐“四周进度安排.xlsx”第1周“算法选择与数据库搭建”阶段的可验收闭环：数据一致、数据库可查询、PyTorch DataLoader 可生成样本、Django 页面可调用并展示加载结果，同时保留现有线性趋势 baseline 作为第二周深度模型的对照。

## 范围与非目标

- 范围：修复职位编码一致性；增加 SQLite 建库/导入/查询；增加技能时序 Dataset/DataLoader 和图张量；增加最小 Django 展示入口；补充时序图注意力模型的最小结构与依赖；更新运行文档和验收检查。
- 非目标：不在本次实现完整训练调参、生产部署、真实招聘数据采集或大规模用户系统。
- 原始 O*NET 文本和原始压缩包只读保留，不修改。

## 总体架构

1. `src/prepare_dataset.py` 从 O*NET 原始文本生成统一的职位、技能、职位-技能关系，以及明确标记为教学/模拟来源的用户事件和转移记录。
2. `src/data_integrity.py` 提供跨表一致性检查；数据生成完成后必须保证所有外键和目标职位均可解析。
3. `src/database.py` 定义 SQLite schema、CSV 导入和查询服务；`src/init_database.py` 负责命令行建库和导入。
4. `src/data_loader.py` 将 `user_skill_events.csv` 构造成固定长度时序样本，将 `occupation_skill.csv` 构造成图节点/边张量；DataLoader 输出字段名稳定、可被 Web 服务复用。
5. `src/temporal_gat.py` 提供轻量时序图注意力网络结构；线性趋势预测继续作为 baseline，训练脚本不作为本次验收前提。
6. `web/` 是最小 Django 项目：使用同一个数据库查询服务和 DataLoader 服务，首页展示数据库统计与首个 batch 的概要，API 返回 JSON 便于验收。

## 数据规范

- 职位主键：`occupation_id`，统一使用原始 O*NET 中存在且有技能记录的编码；若 `.00` 主职业没有 `Skills.txt` 记录，使用显式映射或聚合策略，不产生孤立职位。
- 技能主键：`skill_id`；分类字段统一为 `skill_category`。
- `occupation_skill`：至少包含 `occupation_id`、`skill_id`、`importance`、`level`、`demand_weight`。
- `user_skill_events`：至少包含 `user_id`、`month`、`skill_id`、`level`、`source`。
- `job_transitions`：至少包含 `from_job`、`to_job`、`transition_count`、`transition_probability`、`source`。
- `user_profiles` 的 `current_job`、`target_job` 必须引用职位主表，且目标职位必须出现在职位特征矩阵中。
- `user_skill_events` 与 `job_transitions` 的模拟数据必须保留 `source=derived_teaching_data`，文档不得将其描述为真实调查数据。

## 数据库设计

SQLite 数据库包含六张核心表：

- `occupations(occupation_id PK, occupation_name, description)`
- `skills(skill_id PK, skill_name, skill_category)`
- `occupation_skill(occupation_id FK, skill_id FK, importance, level, importance_norm, level_norm, demand_weight, is_required, PRIMARY KEY(occupation_id, skill_id))`
- `user_profiles(user_id PK, current_job FK, target_job FK)`
- `user_skill_events(event_id PK, user_id FK, month, skill_id FK, level, source)`
- `job_transitions(transition_id PK, from_job FK, to_job FK, transition_count, transition_probability, source)`

导入过程使用事务，先清空并重建目标表，再按外键依赖顺序导入；导入结束执行行数、外键、概率和范围检查，任何失败均回滚并返回非零退出码。

## DataLoader 与模型接口

- `SkillSequenceDataset(events, sequence_length=6)`：按用户和技能组织连续月份，返回 `x:[sequence_length, num_skills]`、`y:[num_skills]`、`user_id`。
- `build_graph_tensors(occupation_skill, transitions)`：返回职位-技能边、转移边及其权重，节点 ID 映射可序列化保存。
- `build_dataloader(data_dir, batch_size=2, sequence_length=6)`：返回可迭代的 PyTorch DataLoader 和元数据。
- `TemporalGAT`：接收技能序列和图边索引，输出每个用户技能的下一阶段或六个月预测；本次只要求构造、前向传播和形状检查。

## Web 接口

- `GET /`：显示数据库六张表的计数、数据质量状态和首个 DataLoader batch 摘要。
- `GET /api/summary/`：返回 JSON `{counts, integrity, loader}`。
- Web 只调用共享的数据库查询和 DataLoader 函数，不复制 CSV 解析逻辑。

## 错误处理

- 原始编码缺失、外键断裂、重复联合键、技能值越界、转移概率不归一化时，数据准备或建库命令必须失败并说明具体表和字段。
- 未安装 PyTorch 时，数据清洗和数据库命令仍可运行；DataLoader、模型和 Web 的 loader 摘要接口返回明确的依赖错误，不伪造成功结果。
- Django 启动前自动检查数据库文件和核心表；缺失时提示先运行初始化命令。

## 测试与验收

测试先覆盖以下行为，再实现对应代码：

1. 重新生成数据后，职位主表与职位-技能表、特征矩阵、用户目标职位无孤立 ID。
2. SQLite 初始化后六张表存在，外键约束生效，查询能返回职位、技能和用户事件。
3. DataLoader 在安装 PyTorch 时能产生预期形状的 batch；无 PyTorch 时错误信息明确。
4. Django 测试客户端能访问 `/` 和 `/api/summary/`，并使用共享服务返回真实计数。
5. 时序图注意力模型能完成一次前向传播并返回 `[batch, num_skills]` 形状。
6. 原有四个流水线脚本的输出仍能生成，推荐评估不会把不存在的目标职位计入候选集合。

## 阶段验收结论标准

只有同时满足以下条件，才标记第一周阶段通过：

- 数据生成命令成功且无孤立职位/技能关系；
- SQLite 数据库可查询，六张核心表均有数据；
- DataLoader 可加载至少一个技能序列 batch；
- Django 页面/API 能展示数据库和 DataLoader 结果；
- 算法文档明确区分 baseline 与时序图注意力最终模型；
- README 和运行脚本能复现上述检查。
