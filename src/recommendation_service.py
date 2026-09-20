"""命令行与 Django Web 共用的推荐、缺口、路径和预测业务服务。

服务层只读取离线 CSV/JSON 产物并返回 JSON 安全字典，不依赖 Django，也不写入
数据库；因此同一套透明计算可以被测试、脚本和 HTTP 接口共同复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.generate_chinese_resumes import OCCUPATION_NAMES_ZH, SKILL_NAMES_ZH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _root(root: Path | None) -> Path:
    """解析项目根目录；调用方可为测试或部署显式覆盖默认路径。"""

    return Path(root) if root is not None else PROJECT_ROOT


def _clean_dir(root: Path) -> Path:
    """返回可审计清洗 CSV 所在的 ``data/clean`` 目录。"""

    return _root(root) / "data" / "clean"


def _processed_dir(root: Path) -> Path:
    """返回特征、预测和转移图产物所在的 ``data/processed`` 目录。"""

    return _root(root) / "data" / "processed"


def _json_safe(value: Any) -> Any:
    """递归把 pandas/NumPy 标量转换为 JsonResponse 可序列化的原生值。"""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _profiles(root: Path) -> pd.DataFrame:
    """读取用户画像表，并强制 ID 字段按字符串保留。"""

    return pd.read_csv(_clean_dir(root) / "user_profiles.csv", dtype=str)


def _ensure_user(user_id: str, root: Path) -> pd.Series | None:
    """检查用户是否存在；存在时返回画像行，否则返回 ``None``。"""

    rows = _profiles(root)
    matches = rows[rows["user_id"].astype(str) == str(user_id)]
    return matches.iloc[0] if not matches.empty else None


def _latest_skill_levels(user_id: str, root: Path) -> dict[str, float]:
    """读取用户每项技能最近月份的水平，供推荐缺口和雷达画像复用。"""

    # 1. 读取完整技能事件表，再按 user_id 截取当前用户的数据副本，避免修改原始 DataFrame。
    events = pd.read_csv(_clean_dir(root) / "user_skill_events.csv")
    user_events = events[events["user_id"].astype(str) == str(user_id)].copy()
    if user_events.empty:
        return {}
    # 2. 将月份和水平强制转换为数值；无法解析的水平以 0 补齐，保证后续雷达图和缺口计算稳定。
    user_events["month"] = pd.to_numeric(user_events["month"], errors="coerce")
    user_events["level"] = pd.to_numeric(user_events["level"], errors="coerce").fillna(0.0)
    # 3. 先按月份升序，再对每个 skill_id 取最后一条，得到该用户每项技能的最新快照。
    latest = user_events.sort_values("month").groupby("skill_id", as_index=False).tail(1)
    # 4. 转换为 {技能ID: 最新水平}，作为推荐缺口、目标职位缺口和技能雷达图的统一数据源。
    return {
        str(record["skill_id"]): float(record["level"])
        for record in latest.to_dict("records")
    }


def _missing_skills_for_job(
    skill_levels: dict[str, float], occupation_id: str, root: Path
) -> list[dict[str, Any]]:
    """返回一个职位的正向技能缺口，并同时提供中英文技能名。"""

    # 1. 职位技能表提供需求权重，技能字典负责把 skill_id 转换为可读名称。
    clean = _clean_dir(root)
    demands = pd.read_csv(clean / "occupation_skill.csv")
    skills = pd.read_csv(clean / "skills.csv", dtype=str)
    names = skills.set_index("skill_id")["skill_name"].astype(str).to_dict()
    # 2. 只保留当前候选职位的技能需求，避免把其他职位的技能混入缺口结果。
    selected = demands[demands["occupation_id"].astype(str) == str(occupation_id)].copy()
    selected["demand_weight"] = pd.to_numeric(selected["demand_weight"], errors="coerce").fillna(0.0)
    # 3. 用 skill_id 映射用户最新水平；用户没有记录的技能按 0 处理，表示尚未掌握。
    selected["current_level"] = selected["skill_id"].astype(str).map(skill_levels).fillna(0.0)
    # 4. 缺口定义为 max(职位需求权重 - 用户当前水平, 0)，已达到要求的技能不会出现负缺口。
    selected["gap"] = (selected["demand_weight"] - selected["current_level"]).clip(lower=0.0)
    # 5. 过滤无缺口技能，并优先展示差距最大、需求权重更高的技能。
    selected = selected[selected["gap"] > 0].sort_values(["gap", "demand_weight"], ascending=False)
    rows: list[dict[str, Any]] = []
    # 6. 为前端职位卡片组装中英文名称、当前水平、需求水平和具体差距。
    for record in selected.to_dict("records"):
        skill_id = str(record["skill_id"])
        skill_name = str(names.get(skill_id, skill_id))
        rows.append({
            "skill_id": skill_id,
            "skill_name": skill_name,
            "skill_name_zh": SKILL_NAMES_ZH.get(skill_name, skill_name),
            "current_level": round(float(record["current_level"]), 6),
            "demand_weight": round(float(record["demand_weight"]), 6),
            "gap": round(float(record["gap"]), 6),
        })
    return rows


def recommend_for_user(user_id: str, top_k: int = 5, root: Path | None = None) -> dict[str, Any]:
    """读取离线混合排序结果，为一个已有用户返回前 ``top_k`` 个职位。"""

    root_path = _root(root)
    # 输入校验：限制 top_k 类型与范围，随后确认用户画像存在。
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 50:
        return {"status": "error", "error": "top_k must be an integer from 1 to 50"}
    profile = _ensure_user(user_id, root_path)
    if profile is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    # 文件读取：推荐分数已由 build_features.py 离线计算并落在 processed 目录。
    path = _processed_dir(root_path) / "recommendation_features.csv"
    if not path.exists():
        return {"status": "error", "error": "recommendation_features.csv is missing"}
    features = pd.read_csv(path)
    rows = features[features["user_id"].astype(str) == str(user_id)].sort_values("rank").head(top_k)
    if rows.empty:
        return {"status": "error", "error": f"no recommendations for user_id: {user_id}"}
    # 业务计算：按既有 rank 排序并截取请求数量，不在服务层重算或改写推荐权重。
    # 新增展示字段在这里统一补充，使离线推荐接口也能逐职位显示具体技能缺口。
    skill_levels = _latest_skill_levels(user_id, root_path)
    recommendations = []
    for record in rows.to_dict("records"):
        # 将 pandas/NumPy 标量先转为 JSON 安全类型，再读取候选职位 ID 和英文名称。
        recommendation = _json_safe(record)
        occupation_id = str(recommendation["candidate_job"])
        occupation_name = str(recommendation.get("candidate_job_name", occupation_id))
        # 中文职位名用于前端直接展示；找不到中文映射时安全回退到原英文名称。
        recommendation["candidate_job_name_zh"] = OCCUPATION_NAMES_ZH.get(occupation_name, occupation_name)
        # 对每个候选职位单独比较需求向量，不能复用目标职位的单一缺口结果。
        recommendation["missing_skills"] = _missing_skills_for_job(skill_levels, occupation_id, root_path)
        # 缺口数量以实时生成的明细长度为准，确保卡片计数和具体标签始终一致。
        recommendation["missing_skill_count"] = len(recommendation["missing_skills"])
        recommendations.append(recommendation)
    # JSON 安全输出：统一转换 NumPy/pandas 标量，供 Django 或命令行直接使用。
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "current_job": str(profile["current_job"]),
        "target_job": str(profile["target_job"]),
        "recommendations": recommendations,
    })


def skill_gap_for_user(user_id: str, occupation_id: str, root: Path | None = None) -> dict[str, Any]:
    """比较用户最新技能水平与目标职位需求，返回正向技能缺口。"""

    root_path = _root(root)
    # 输入校验：确认用户与目标职位都存在。
    profile = _ensure_user(user_id, root_path)
    if profile is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    # 文件读取：职位需求、职位字典、技能字典和用户事件均来自 clean CSV。
    clean = _clean_dir(root_path)
    occupations = pd.read_csv(clean / "occupations.csv", dtype=str)
    if str(occupation_id) not in set(occupations["occupation_id"].astype(str)):
        return {"status": "error", "error": f"unknown occupation_id: {occupation_id}"}
    # 业务计算：每项技能只取最新月份，并计算 max(需求权重 - 当前水平, 0)。
    levels = _latest_skill_levels(user_id, root_path)
    if not levels:
        return {"status": "error", "error": f"no skill events for user_id: {user_id}"}
    rows = _missing_skills_for_job(levels, str(occupation_id), root_path)
    occupation_name = occupations.set_index("occupation_id").loc[str(occupation_id), "occupation_name"]
    # JSON 安全输出：缺口按大小降序，附带可读技能与职位名称。
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "occupation_id": str(occupation_id),
        "occupation_name": str(occupation_name),
        "occupation_name_zh": OCCUPATION_NAMES_ZH.get(str(occupation_name), str(occupation_name)),
        "skills": rows,
        "missing_skill_count": len(rows),
    })


def _select_path(
    graph: dict[str, list[tuple[str, float]]],
    source: str,
    target: str,
    max_hops: int,
    strategy: str,
) -> tuple[list[str], float]:
    """枚举有限跳数内的无环路径，按快速或稳健目标选出一条。

    ``fast`` 先比较路径节点数，选择跳数最少的方案；跳数相同时选择概率更高者。
    ``stable`` 先比较沿途转移概率的乘积，选择累计概率最高的方案；概率相同时
    选择路径更短者。两种策略共享同一批候选路径，差别只在最终排序目标。
    """

    # 当前职位已经是目标职位时无需搜索，返回零跳路径和 100% 可行性。
    if source == target:
        return [source], 1.0
    # 每个候选项保存完整职位路径和沿途边概率的乘积。
    candidates: list[tuple[list[str], float]] = []

    def visit(node: str, current_path: list[str], probability: float) -> None:
        """深度优先枚举不超过 max_hops 的无环路径，并收集到达目标的方案。"""

        # current_path 包含起点，因此边数等于 len(current_path) - 1。
        if len(current_path) - 1 >= max_hops:
            return
        for next_job, edge_probability in graph.get(node, []):
            # 禁止回到当前路径中已经出现的职位，防止转移图中的环导致无限递归。
            if next_job in current_path:
                continue
            next_path = current_path + [next_job]
            # 路径可行性采用各条转移边概率连乘，任一低概率步骤都会降低整体稳定性。
            next_probability = probability * edge_probability
            if next_job == target:
                candidates.append((next_path, next_probability))
            else:
                visit(next_job, next_path, next_probability)

    # 从当前职位出发，初始概率为 1；搜索结束后统一执行策略选择。
    visit(source, [source], 1.0)
    if not candidates:
        return [], 0.0
    if strategy == "fast":
        # 快速路线：节点数越少越优；同长度下用负概率实现“概率越大越优”的次级排序。
        return min(candidates, key=lambda item: (len(item[0]), -item[1], item[0]))
    # 稳健路线：概率乘积越大越优；同概率下优先较短路径，职位序列用于保证结果确定性。
    return max(candidates, key=lambda item: (item[1], -len(item[0]), item[0]))


def career_path_for_user(
    user_id: str,
    occupation_id: str,
    root: Path | None = None,
    max_hops: int = 3,
    strategy: str = "fast",
) -> dict[str, Any]:
    """按快速（最少跳数）或稳健（最大概率）策略规划职业路径。"""

    root_path = _root(root)
    # 1. 输入校验：用户必须存在，策略只能是 fast/stable，目标职位必须属于职位字典。
    profile = _ensure_user(user_id, root_path)
    if profile is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    if strategy not in {"fast", "stable"}:
        return {"status": "error", "error": "strategy must be 'fast' or 'stable'"}
    occupations = pd.read_csv(_clean_dir(root_path) / "occupations.csv", dtype=str)
    valid_jobs = set(occupations["occupation_id"].astype(str))
    if str(occupation_id) not in valid_jobs:
        return {"status": "error", "error": f"unknown occupation_id: {occupation_id}"}
    current_job = str(profile["current_job"])
    # 2. 文件读取：使用 build_features.py 生成的职位转移邻接 JSON。
    graph_path = _processed_dir(root_path) / "transition_graph.json"
    if not graph_path.exists():
        return {"status": "error", "error": "transition_graph.json is missing"}
    raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    # 3. 把 JSON 边对象标准化为 {起点: [(终点, 概率)]}，供路径搜索函数直接遍历。
    graph = {
        str(source): [(str(edge["to_job"]), float(edge["probability"])) for edge in edges]
        for source, edges in raw_graph.items()
    }
    # 4. 两种策略共用同一图和最大跳数，仅由 strategy 决定最终选择目标。
    path, probability = _select_path(graph, current_job, str(occupation_id), max_hops, strategy)
    names = occupations.set_index("occupation_id")["occupation_name"].to_dict()
    # 5. JSON 输出同时返回策略标识、中文路径名称、累计概率和实际跳数，供前端解释结果。
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "current_job": current_job,
        "target_job": str(occupation_id),
        "strategy": strategy,
        "strategy_name": "快速路线" if strategy == "fast" else "稳健路线",
        "path": path,
        "path_names": [OCCUPATION_NAMES_ZH.get(str(names.get(job, job)), str(names.get(job, job))) for job in path],
        "path_probability": probability,
        "path_length": max(0, len(path) - 1) if path else -1,
    })


def skill_profile_for_user(user_id: str, root: Path | None = None) -> dict[str, Any]:
    """返回已有用户的最新技能画像，供前端雷达图使用。"""

    root_path = _root(root)
    # 1. 雷达图只允许查询已有用户；未知用户直接返回业务错误，不生成空画像。
    if _ensure_user(user_id, root_path) is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    # 2. 复用统一的最新技能快照，确保雷达图与推荐缺口使用同一时间点的数据。
    levels = _latest_skill_levels(user_id, root_path)
    if not levels:
        return {"status": "error", "error": f"no skill events for user_id: {user_id}"}
    # 3. 读取技能字典，为每个技能水平补充英文名和中文名。
    skills = pd.read_csv(_clean_dir(root_path) / "skills.csv", dtype=str)
    names = skills.set_index("skill_id")["skill_name"].astype(str).to_dict()
    rows = []
    # 4. 按水平从高到低排序；相同水平再按 skill_id 排序，保证雷达图数据顺序稳定可复现。
    for skill_id, level in sorted(levels.items(), key=lambda item: (-item[1], item[0])):
        skill_name = str(names.get(skill_id, skill_id))
        rows.append({
            "skill_id": skill_id,
            "skill_name": skill_name,
            "skill_name_zh": SKILL_NAMES_ZH.get(skill_name, skill_name),
            "level": round(float(level), 6),
        })
    # 5. 前端会从完整画像中选择最高的若干维度绘制 SVG 雷达图，因此接口保留全部技能。
    return _json_safe({"status": "ok", "user_id": str(user_id), "skills": rows})


def forecast_for_user(user_id: str, months: int = 6, root: Path | None = None) -> dict[str, Any]:
    """根据请求月份重算线性趋势预测，并明确标记为 baseline。"""

    root_path = _root(root)
    # 输入校验：预测区间限制为 1—60 个月，且用户必须存在。
    if not isinstance(months, int) or isinstance(months, bool) or months < 1 or months > 60:
        return {"status": "error", "error": "months must be an integer from 1 to 60"}
    if _ensure_user(user_id, root_path) is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    # 文件读取：复用离线文件中的最后水平与月度斜率，而非加载神经网络模型。
    path = _processed_dir(root_path) / "skill_forecast_6m.csv"
    if not path.exists():
        return {"status": "unavailable", "error": "skill_forecast_6m.csv is missing"}
    forecasts = pd.read_csv(path)
    rows = forecasts[forecasts["user_id"].astype(str) == str(user_id)].copy()
    if rows.empty:
        return {"status": "unavailable", "error": f"no forecasts for user_id: {user_id}"}
    # 业务计算：last_level + monthly_slope * months，并裁剪到 [0, 1]。
    rows["forecast_level"] = (
        pd.to_numeric(rows["last_level"], errors="coerce")
        + pd.to_numeric(rows["monthly_slope"], errors="coerce") * months
    ).clip(0.0, 1.0)
    result_rows = []
    for record in rows.to_dict("records"):
        result_rows.append({
            "skill_id": str(record["skill_id"]),
            "last_level": float(record["last_level"]),
            "monthly_slope": float(record["monthly_slope"]),
            "forecast_level": float(record["forecast_level"]),
        })
    # JSON 安全输出：model 固定为 linear_trend_baseline，避免误称 TemporalGAT 结果。
    return _json_safe({
        "status": "baseline",
        "user_id": str(user_id),
        "months": months,
        "model": "linear_trend_baseline",
        "data_note": "This forecast uses deterministic teaching data.",
        "forecasts": result_rows,
    })


__all__ = [
    "career_path_for_user",
    "forecast_for_user",
    "recommend_for_user",
    "skill_profile_for_user",
    "skill_gap_for_user",
]
