"""In-memory analysis for an uploaded resume.

The canonical CSV files and SQLite database are intentionally read-only here.
This module turns a short text resume into a bounded skill vector and applies
the same transparent hybrid scoring used by the offline recommendation job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.generate_chinese_resumes import SKILL_NAMES_ZH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else PROJECT_ROOT


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _load_context(root: Path) -> dict[str, Any]:
    clean = root / "data" / "clean"
    processed = root / "data" / "processed"
    skills = pd.read_csv(clean / "skills.csv", dtype=str)
    occupations = pd.read_csv(clean / "occupations.csv", dtype=str)
    occupation_skill = pd.read_csv(clean / "occupation_skill.csv")
    job_vectors = pd.read_csv(processed / "job_feature_matrix.csv")
    job_vectors = job_vectors.set_index("occupation_id")
    skill_ids = [str(skill_id) for skill_id in job_vectors.columns]
    names_en = skills.set_index("skill_id")["skill_name"].astype(str).to_dict()
    names_zh = {skill_id: SKILL_NAMES_ZH.get(name, name) for skill_id, name in names_en.items()}
    aliases: dict[str, str] = {}
    for skill_id, english_name in names_en.items():
        aliases[str(english_name).casefold()] = str(skill_id)
        aliases[names_zh.get(str(skill_id), str(english_name)).casefold()] = str(skill_id)
    by_name = {str(row.skill_name): str(row.skill_id) for row in skills.itertuples(index=False)}
    domain_aliases = {
        "python": by_name.get("Programming"),
        "编程": by_name.get("Programming"),
        "软件开发": by_name.get("Programming"),
        "数据库": by_name.get("Systems Analysis"),
        "机器学习": by_name.get("Mathematics"),
        "数据分析": by_name.get("Operations Analysis"),
        "网络": by_name.get("Troubleshooting"),
        "测试": by_name.get("Quality Control Analysis"),
    }
    aliases.update({key: value for key, value in domain_aliases.items() if value})
    transitions = pd.read_csv(clean / "job_transitions.csv")
    graph: dict[str, list[tuple[str, float]]] = {}
    for row in transitions.itertuples(index=False):
        graph.setdefault(str(row.from_job), []).append((str(row.to_job), float(row.transition_probability)))
    occupation_names = occupations.set_index("occupation_id")["occupation_name"].astype(str).to_dict()
    return {
        "skills": skills,
        "occupations": occupations,
        "occupation_skill": occupation_skill,
        "job_vectors": job_vectors,
        "skill_ids": skill_ids,
        "skill_names_en": {str(key): str(value) for key, value in names_en.items()},
        "skill_names_zh": {str(key): str(value) for key, value in names_zh.items()},
        "aliases": aliases,
        "graph": graph,
        "occupation_names": {str(key): str(value) for key, value in occupation_names.items()},
    }


def _path(graph: dict[str, list[tuple[str, float]]], source: str, target: str, max_hops: int = 3) -> tuple[list[str], float]:
    if source == target:
        return [source], 1.0
    best_path: list[str] = []
    best_probability = 0.0

    def visit(node: str, current_path: list[str], probability: float) -> None:
        nonlocal best_path, best_probability
        if len(current_path) - 1 >= max_hops:
            return
        for next_node, edge_probability in graph.get(node, []):
            if next_node in current_path:
                continue
            next_path = current_path + [next_node]
            next_probability = probability * edge_probability
            if next_node == target and next_probability > best_probability:
                best_path = next_path
                best_probability = next_probability
                continue
            visit(next_node, next_path, next_probability)

    visit(source, [source], 1.0)
    return best_path, best_probability


def _recommendations(context: dict[str, Any], skill_vector: dict[str, float], current_job: str, target_job: str | None, top_k: int) -> list[dict[str, Any]]:
    job_vectors: pd.DataFrame = context["job_vectors"]
    skill_ids = context["skill_ids"]
    user_vector = np.array([float(skill_vector.get(skill_id, 0.1)) for skill_id in skill_ids], dtype=float)
    current_vector = job_vectors.loc[current_job].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for candidate_job in job_vectors.index.astype(str):
        if candidate_job == current_job:
            continue
        candidate_vector = job_vectors.loc[candidate_job].to_numpy(dtype=float)
        gap_vector = np.maximum(candidate_vector - user_vector, 0.0)
        demand_sum = float(candidate_vector.sum()) or 1.0
        gap_score = float(np.dot(gap_vector, candidate_vector) / demand_sum)
        missing_count = int(((candidate_vector >= 0.28) & (user_vector < 0.28)).sum())
        match_score = _cosine(user_vector, candidate_vector)
        direct_probability = 0.0
        for next_job, probability in context["graph"].get(current_job, []):
            if next_job == candidate_job:
                direct_probability += probability
        path, path_probability = _path(context["graph"], current_job, candidate_job)
        growth_score = float(np.clip(0.5 + candidate_vector.mean() - current_vector.mean(), 0, 1))
        recommendation_score = 0.45 * match_score - 0.25 * gap_score + 0.15 * growth_score + 0.15 * path_probability
        rows.append({
            "candidate_job": candidate_job,
            "candidate_job_name": context["occupation_names"].get(candidate_job, candidate_job),
            "match_score": round(match_score, 6),
            "gap_score": round(gap_score, 6),
            "missing_skill_count": missing_count,
            "direct_transition_probability": round(direct_probability, 6),
            "path_probability": round(path_probability, 6),
            "path_length": len(path) - 1 if path else -1,
            "growth_score": round(growth_score, 6),
            "estimated_training_hours": round(40 + 320 * gap_score + 12 * missing_count, 2),
            "recommendation_score": round(recommendation_score, 6),
            "is_target": int(candidate_job == target_job),
        })
    rows.sort(key=lambda row: (-row["recommendation_score"], row["candidate_job"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:top_k]


def _skill_gap(context: dict[str, Any], skill_vector: dict[str, float], target_job: str) -> dict[str, Any]:
    table = context["occupation_skill"]
    selected = table[table["occupation_id"].astype(str) == target_job].copy()
    selected["current_level"] = selected["skill_id"].astype(str).map(skill_vector).fillna(0.1)
    selected["gap"] = (selected["demand_weight"] - selected["current_level"]).clip(lower=0.0)
    selected = selected[selected["gap"] > 0].sort_values(["gap", "demand_weight"], ascending=False)
    rows = []
    for record in selected.to_dict("records"):
        skill_id = str(record["skill_id"])
        rows.append({
            "skill_id": skill_id,
            "skill_name": context["skill_names_en"].get(skill_id, skill_id),
            "skill_name_zh": context["skill_names_zh"].get(skill_id, skill_id),
            "demand_weight": round(float(record["demand_weight"]), 6),
            "current_level": round(float(record["current_level"]), 6),
            "gap": round(float(record["gap"]), 6),
        })
    return {
        "status": "ok",
        "occupation_id": target_job,
        "occupation_name": context["occupation_names"].get(target_job, target_job),
        "skills": rows,
        "missing_skill_count": len(rows),
    }


def analyze_resume_text(
    text: str,
    root: Path | None = None,
    current_job: str | None = None,
    target_job: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Analyze uploaded text without writing any project state."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("简历文本不能为空")
    context = _load_context(_root(root))
    valid_jobs = set(context["occupations"]["occupation_id"].astype(str))
    if current_job and str(current_job) not in valid_jobs:
        raise ValueError(f"未知当前职位：{current_job}")
    if target_job and str(target_job) not in valid_jobs:
        raise ValueError(f"未知目标职位：{target_job}")

    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    skill_levels = {skill_id: 0.1 for skill_id in context["skill_ids"]}
    recognized_ids: set[str] = set()
    for alias, skill_id in sorted(context["aliases"].items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in normalized:
            recognized_ids.add(skill_id)
    for skill_id in recognized_ids:
        skill_levels[skill_id] = 0.75
    if current_job is None:
        vector = np.array([skill_levels[skill_id] for skill_id in context["skill_ids"]], dtype=float)
        current_job = max(
            valid_jobs,
            key=lambda job_id: _cosine(vector, context["job_vectors"].loc[job_id].to_numpy(dtype=float)),
        )
    current_job = str(current_job)
    recommendations = _recommendations(context, skill_levels, current_job, target_job, max(1, min(int(top_k), 50)))
    if target_job is None:
        target_job = recommendations[0]["candidate_job"] if recommendations else current_job
    target_job = str(target_job)
    gap = _skill_gap(context, skill_levels, target_job)
    path, path_probability = _path(context["graph"], current_job, target_job)
    return {
        "status": "ok",
        "persisted": False,
        "current_job": current_job,
        "current_job_name": context["occupation_names"].get(current_job, current_job),
        "target_job": target_job,
        "target_job_name": context["occupation_names"].get(target_job, target_job),
        "skill_count": len(recognized_ids),
        "recognized_skills": [
            {
                "skill_id": skill_id,
                "skill_name": context["skill_names_en"].get(skill_id, skill_id),
                "skill_name_zh": context["skill_names_zh"].get(skill_id, skill_id),
                "level": skill_levels[skill_id],
            }
            for skill_id in sorted(recognized_ids)
        ],
        "occupations": sorted(valid_jobs),
        "recommendations": recommendations,
        "skill_gap": gap,
        "career_path": {
            "status": "ok",
            "current_job": current_job,
            "target_job": target_job,
            "path": path,
            "path_names": [context["occupation_names"].get(job_id, job_id) for job_id in path],
            "path_probability": round(path_probability, 6),
            "path_length": len(path) - 1 if path else -1,
        },
        "data_note": "上传简历仅在本次请求内存中分析，不写入数据库或训练数据。",
    }


__all__ = ["SUPPORTED_EXTENSIONS", "analyze_resume_text"]
