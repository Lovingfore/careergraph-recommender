"""Portable recommendation helpers shared by the command line and web demo."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else PROJECT_ROOT


def _clean_dir(root: Path) -> Path:
    return _root(root) / "data" / "clean"


def _processed_dir(root: Path) -> Path:
    return _root(root) / "data" / "processed"


def _json_safe(value: Any) -> Any:
    """Convert pandas/NumPy scalar values to values accepted by JsonResponse."""

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
    return pd.read_csv(_clean_dir(root) / "user_profiles.csv", dtype=str)


def _ensure_user(user_id: str, root: Path) -> pd.Series | None:
    rows = _profiles(root)
    matches = rows[rows["user_id"].astype(str) == str(user_id)]
    return matches.iloc[0] if not matches.empty else None


def recommend_for_user(user_id: str, top_k: int = 5, root: Path | None = None) -> dict[str, Any]:
    """Return the existing hybrid recommendation ranking for one user."""

    root_path = _root(root)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 50:
        return {"status": "error", "error": "top_k must be an integer from 1 to 50"}
    profile = _ensure_user(user_id, root_path)
    if profile is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    path = _processed_dir(root_path) / "recommendation_features.csv"
    if not path.exists():
        return {"status": "error", "error": "recommendation_features.csv is missing"}
    features = pd.read_csv(path)
    rows = features[features["user_id"].astype(str) == str(user_id)].sort_values("rank").head(top_k)
    if rows.empty:
        return {"status": "error", "error": f"no recommendations for user_id: {user_id}"}
    recommendations = []
    for record in rows.to_dict("records"):
        recommendations.append(_json_safe(record))
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "current_job": str(profile["current_job"]),
        "target_job": str(profile["target_job"]),
        "recommendations": recommendations,
    })


def skill_gap_for_user(user_id: str, occupation_id: str, root: Path | None = None) -> dict[str, Any]:
    """Describe skills whose occupation demand exceeds the user's latest level."""

    root_path = _root(root)
    profile = _ensure_user(user_id, root_path)
    if profile is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    clean = _clean_dir(root_path)
    occupation_skill = pd.read_csv(clean / "occupation_skill.csv")
    occupations = pd.read_csv(clean / "occupations.csv", dtype=str)
    skills = pd.read_csv(clean / "skills.csv", dtype=str)
    if str(occupation_id) not in set(occupations["occupation_id"].astype(str)):
        return {"status": "error", "error": f"unknown occupation_id: {occupation_id}"}
    events = pd.read_csv(clean / "user_skill_events.csv")
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
    return _json_safe({
        "status": "ok",
        "user_id": str(user_id),
        "occupation_id": str(occupation_id),
        "occupation_name": str(occupation_name),
        "skills": rows,
        "missing_skill_count": len(rows),
    })


def career_path_for_user(user_id: str, occupation_id: str, root: Path | None = None, max_hops: int = 3) -> dict[str, Any]:
    """Find the shortest feasible path from the user's current job to a target job."""

    root_path = _root(root)
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
        graph_path = _processed_dir(root_path) / "transition_graph.json"
        if not graph_path.exists():
            return {"status": "error", "error": "transition_graph.json is missing"}
        raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph = {
            str(source): [(str(edge["to_job"]), float(edge["probability"])) for edge in edges]
            for source, edges in raw_graph.items()
        }
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
    """Return transparent linear-trend forecasts with an explicit baseline label."""

    root_path = _root(root)
    if not isinstance(months, int) or isinstance(months, bool) or months < 1 or months > 60:
        return {"status": "error", "error": "months must be an integer from 1 to 60"}
    if _ensure_user(user_id, root_path) is None:
        return {"status": "error", "error": f"unknown user_id: {user_id}"}
    path = _processed_dir(root_path) / "skill_forecast_6m.csv"
    if not path.exists():
        return {"status": "unavailable", "error": "skill_forecast_6m.csv is missing"}
    forecasts = pd.read_csv(path)
    rows = forecasts[forecasts["user_id"].astype(str) == str(user_id)].copy()
    if rows.empty:
        return {"status": "unavailable", "error": f"no forecasts for user_id: {user_id}"}
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
