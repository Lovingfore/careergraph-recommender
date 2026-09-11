# Web 可视化仪表盘与临时简历分析设计

## 目标

为 CareerGraph Recommender 增加一个可通过 Django Web 打开的中文可视化界面，展示数据集、SQLite 数据库、DataLoader 和模型训练状态；支持输入现有用户进行动态职位推荐、技能缺口和职业路径规划；支持上传 UTF-8 `.txt`、`.md` 或 `.csv` 简历并在内存中生成临时技能画像，不写入 SQLite、CSV 或模型训练集。

## 约束

- 课程设计规模保持轻量，继续使用 Django、SQLite、CSV 和现有 Python 服务。
- 不引入登录系统、微服务、图数据库或前端构建链。
- 上传文件最大 1 MB，只接受文本格式；不依赖外部 CDN。
- 上传数据只存在于一次 HTTP 请求的内存对象中，响应不返回持久化 token，也不改变 canonical 数据。
- 页面必须标注数据为确定性合成中文技术岗简历，TemporalGAT 指标仅为教学实验结果。

## 组件与数据流

1. `views.py` 从 `artifacts/models/evaluation_summary.json` 和 checkpoint 元信息读取模型卡片；从 CSV/SQLite 读取当前统计。
2. `POST /api/resume-upload/` 接收 multipart 文件，校验扩展名、大小和 UTF-8 文本，按 `skills.csv` 中文/英文技能关键词抽取临时技能向量。
3. 临时技能向量与职位需求矩阵计算每个候选职位的余弦匹配、缺口、成长和路径分数，返回推荐、缺口、路径与解析摘要；不调用 `user_profiles.csv` 的写入逻辑。
4. 现有 GET API 保持兼容；增加 `/api/model-info/` 与 `/api/occupations/` 用于页面卡片和下拉框。
5. 页面使用原生 HTML/CSS/JavaScript，展示指标卡、模型对比、数据库表格、上传结果、推荐表格、技能缺口表、职业路径和二部图。

## 错误处理

- 缺少文件、扩展名不支持、文件过大、空文本或无法解码返回 HTTP 400 和中文错误信息。
- 不存在的用户/职位沿用现有服务的结构化错误响应。
- 模型评估文件缺失时返回 `status=unavailable`，页面显示“未找到训练产物”，不伪造指标。

## 验收标准

- Web 首页可访问并显示数据、模型和上传入口。
- 模型信息 API 返回训练状态、epoch、样本切分、baseline 与 TemporalGAT MAE/RMSE。
- 上传一份包含“Python、数据库、机器学习”等关键词的文本简历后，返回 `status=ok`、`persisted=false`、推荐职位列表和职业路径；数据库计数保持不变。
- 非法扩展名、空文件和超大文件返回 400。
- 现有 28 个测试保持通过，并新增上传、模型信息和数据库不变测试。
- 汇报文档记录全部功能、数据规模、模型指标、验证证据、运行命令和限制。
