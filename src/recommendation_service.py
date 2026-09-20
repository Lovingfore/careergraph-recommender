"""命令行与 Django Web 共用的推荐、缺口、路径和预测业务服务。

服务层只读取离线 CSV/JSON 产物并返回 JSON 安全字典，不依赖 Django，也不写入
数据库；因此同一套透明计算可以被测试、脚本和 HTTP 接口共同复用。
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    # 业务计算：按既有 rank 排序并截取请求数量，不在服务层重算或改写权重。
    recommendations = []
    for record in rows.to_dict("records"):
        recommendations.append(_json_safe(record))
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
    occupation_skill = pd.read_csv(clean / "occupation_skill.csv")
    occupations = pd.read_csv(clean / "occupations.csv", dtype=str)
    skills = pd.read_csv(clean / "skills.csv", dtype=str)
    if str(occupation_id) not in set(occupations["occupation_id"].astype(str)):
        return {"status": "error", "error": f"unknown occupation_id: {occupation_id}"}
    events = pd.read_csv(clean / "user_skill_events.csv")
    # 业务计算：每项技能只取最新月份，并计算 max(需求权重 - 当前水平, 0)。
    user_events = events[events["user_id"].astype(str) == str(user_id)].copy()
    if user_events.empty:
        return {"status": "error", "error": f"no skill events for user_id: {user_id}"}
    user_events["month"] = pd.to_numeric(user_events["month"], errors="coerce")
    user_events = user_events.sort_values("month").groupby("skill_id", as_index=False).tail(1)
    levels = dict(zip(user_events["skill_id"].astype(str), pd.to_numeric(user_events["level"], errors="coerce")))
    selected = occupation_skill[occupation_skill["occupation_id"].astype(str) == str(occupation_id)].copy()
    selected["demand_weight"] = pd.to_numeric(selected["demand_weight"], errors="coerce").fillna(0.0)
    selected["current_level"] = selected["skill_id"].astype(str).map(levels).fillna(0.0)
    selected["gap"] = (selected["demand_weight"] - selected["current_level"]).clip(lower=0.0)
    selected = selected[selected["gap"] > 0].sort_values(["gap", "demand_weight"], ascending=False)
    names = skills.set_index("skill_id")["skill_name"].to_dict()
    rows = []
    for record in selected.to_dict("records"):
        rows.append({
            "skill_id": str(record["skill_id"]),
            "skill_name": str(names.get(str(record["skill_id"]), record["skill_id"])),
            "demand_weight": float(record["demand_weight"]),
            "current_level": float(record["current_level"]),
            "gap": float(record["gap"]),
        })
    occupation_name = occupations.set_index("occupation_id").loc[str(occupation_id), "occupation_name"]
    # JSON 安全输出：缺口按大小降序，附带可读技能与职位名称。
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "occupation_id": str(occupation_id),
        "occupation_name": str(occupation_name),
        "skills": rows,
        "missing_skill_count": len(rows),
    })


def career_path_for_user(user_id: str, occupation_id: str, root: Path | None = None, max_hops: int = 3) -> dict[str, Any]:
    """用广度优先搜索查找当前职位到目标职位最多三跳的最短可行路径。"""

    root_path = _root(root)
    # 输入校验：用户必须存在，目标职位必须属于职位字典。
    profile = _ensure_user(user_id, root_path)
    if profile is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    occupations = pd.read_csv(_clean_dir(root_path) / "occupations.csv", dtype=str)
    valid_jobs = set(occupations["occupation_id"].astype(str))
    if str(occupation_id) not in valid_jobs:
        return {"status": "error", "error": f"unknown occupation_id: {occupation_id}"}
    current_job = str(profile["current_job"])
    if current_job == str(occupation_id):
        path = [current_job]
        probability = 1.0
    else:
        # 文件读取：使用 build_features.py 生成的职位转移邻接 JSON。
        graph_path = _processed_dir(root_path) / "transition_graph.json"
        if not graph_path.exists():
            return {"status": "error", "error": "transition_graph.json is missing"}
        raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph = {
            str(source): [(str(edge["to_job"]), float(edge["probability"])) for edge in edges]
            for source, edges in raw_graph.items()
        }
        # 业务计算：BFS 保证先找到跳数最少的路径，沿途概率通过边概率连乘得到。
        queue: deque[tuple[str, list[str], float]] = deque([(current_job, [current_job], 1.0)])
        path = []
        probability = 0.0
        while queue:
            node, current_path, current_probability = queue.popleft()
            if len(current_path) - 1 >= max_hops:
                continue
            for next_job, edge_probability in graph.get(node, []):
                if next_job in current_path:
                    continue
                next_path = current_path + [next_job]
                next_probability = current_probability * edge_probability
                if next_job == str(occupation_id):
                    path, probability = next_path, next_probability
                    queue.clear()
                    break
                queue.append((next_job, next_path, next_probability))
            if path:
                break
    names = occupations.set_index("occupation_id")["occupation_name"].to_dict()
    # JSON 安全输出：同时返回职位 ID、可读名称、路径概率和实际跳数。
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "current_job": current_job,
        "target_job": str(occupation_id),
        "path": path,
        "path_names": [str(names.get(job, job)) for job in path],
        "path_probability": probability,
        "path_length": max(0, len(path) - 1) if path else -1,
    })


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
    "skill_gap_for_user",
]
