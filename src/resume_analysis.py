"""对上传简历执行不落库的纯内存分析。

标准 CSV 与 SQLite 在本模块中始终只读。文本简历先被转换为有界技能向量，再复用
离线推荐的透明混合评分、技能缺口和职业路径计算；返回结果明确标记
``persisted=false``，不会改变训练数据或用户画像。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.generate_chinese_resumes import OCCUPATION_NAMES_ZH, SKILL_NAMES_ZH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}


def _root(root: Path | None) -> Path:
    """解析分析所需项目根目录，允许测试注入隔离数据目录。"""

    return Path(root) if root is not None else PROJECT_ROOT


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算技能向量余弦相似度，零向量时安全返回 0。"""

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _load_context(root: Path) -> dict[str, Any]:
    """一次性加载技能别名、职位向量、职位字典与转移图分析上下文。

    英文技能名、中文技能名和常用领域词被映射到统一 skill_id；职位矩阵来自
    离线特征产物，转移图来自 clean CSV，后续计算因此无需重复访问数据库。
    """

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


def _path(
    graph: dict[str, list[tuple[str, float]]],
    source: str,
    target: str,
    max_hops: int = 3,
    strategy: str = "stable",
) -> tuple[list[str], float]:
    """枚举最多三跳的无环路径，并按路线策略选取结果。

    上传简历产生的是临时用户画像，不能直接调用已有用户接口，因此在内存分析模块
    中执行同样的路线规则：快速路线优先最少跳数，稳健路线优先最大概率乘积。
    """

    # 起点和终点一致时不需要转移，零跳路径的累计概率定义为 1。
    if source == target:
        return [source], 1.0
    # 候选元素结构为 (职位ID路径, 沿途转移概率乘积)。
    candidates: list[tuple[list[str], float]] = []

    def visit(node: str, current_path: list[str], probability: float) -> None:
        """递归枚举未重复节点的候选路径。"""

        # 边数达到上限后停止扩展，防止生成超过页面设计范围的过长路线。
        if len(current_path) - 1 >= max_hops:
            return
        for next_node, edge_probability in graph.get(node, []):
            # 当前路径中已访问的职位不再进入，保证枚举的是无环路径。
            if next_node in current_path:
                continue
            next_path = current_path + [next_node]
            # 每增加一次职位转移，就把对应边概率乘入累计可行性。
            next_probability = probability * edge_probability
            if next_node == target:
                candidates.append((next_path, next_probability))
                continue
            visit(next_node, next_path, next_probability)

    # 从当前职位开始遍历，初始可行性为 100%。
    visit(source, [source], 1.0)
    if not candidates:
        return [], 0.0
    if strategy == "fast":
        # 快速路线：先选节点数最少者；同样长度时再选累计概率较高者。
        return min(candidates, key=lambda item: (len(item[0]), -item[1], item[0]))
    # 稳健路线：先选概率乘积最高者；同概率时偏向更短的学习/转岗链路。
    return max(candidates, key=lambda item: (item[1], -len(item[0]), item[0]))


def _recommendations(context: dict[str, Any], skill_vector: dict[str, float], current_job: str, target_job: str | None, top_k: int) -> list[dict[str, Any]]:
    """让临时技能向量复用离线推荐公式，生成当前请求的 Top-K 职位。

    总分仍为 ``0.45 Match - 0.25 Gap + 0.15 Growth + 0.15 Path``，仅用户向量
    来自本次上传文本，其余职位需求和转移数据与离线构建完全一致。
    """

    # 1. 按固定 skill_id 顺序把简历技能字典转换为数值向量，确保和职位矩阵列严格对齐。
    job_vectors: pd.DataFrame = context["job_vectors"]
    skill_ids = context["skill_ids"]
    user_vector = np.array([float(skill_vector.get(skill_id, 0.1)) for skill_id in skill_ids], dtype=float)
    # 当前职位向量用于计算候选职位相对当前岗位的成长幅度。
    current_vector = job_vectors.loc[current_job].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    # 2. 逐个候选职位计算评分；当前职位本身不需要作为推荐结果再次返回。
    for candidate_job in job_vectors.index.astype(str):
        if candidate_job == current_job:
            continue
        candidate_vector = job_vectors.loc[candidate_job].to_numpy(dtype=float)
        # 只保留“需求高于当前水平”的正向差值，再按职位需求加权形成整体缺口分数。
        gap_vector = np.maximum(candidate_vector - user_vector, 0.0)
        demand_sum = float(candidate_vector.sum()) or 1.0
        gap_score = float(np.dot(gap_vector, candidate_vector) / demand_sum)
        # 除聚合缺口分数外，再生成具体技能明细，供每张推荐卡片展示技能名称和差距。
        missing_skills = _skill_gap(context, skill_vector, candidate_job)["skills"]
        missing_count = len(missing_skills)
        # 余弦相似度表示简历技能结构与候选职位需求结构的匹配程度。
        match_score = _cosine(user_vector, candidate_vector)
        # 直接转移概率只统计当前职位到候选职位的一跳边，用于解释是否可直接转岗。
        direct_probability = 0.0
        for next_job, probability in context["graph"].get(current_job, []):
            if next_job == candidate_job:
                direct_probability += probability
        # 推荐排序中的路径分数沿用原有最大概率路径，路线策略只影响用户指定目标的最终展示路径。
        path, path_probability = _path(context["graph"], current_job, candidate_job)
        # 成长分数比较候选职位和当前职位的平均技能需求，并裁剪到合法的 [0, 1] 区间。
        growth_score = float(np.clip(0.5 + candidate_vector.mean() - current_vector.mean(), 0, 1))
        # 透明混合公式：匹配和成长提高分数，技能缺口降低分数，路径可行性提供转岗修正。
        recommendation_score = 0.45 * match_score - 0.25 * gap_score + 0.15 * growth_score + 0.15 * path_probability
        rows.append({
            "candidate_job": candidate_job,
            "candidate_job_name": context["occupation_names"].get(candidate_job, candidate_job),
            "candidate_job_name_zh": OCCUPATION_NAMES_ZH.get(
                context["occupation_names"].get(candidate_job, candidate_job),
                context["occupation_names"].get(candidate_job, candidate_job),
            ),
            "match_score": round(match_score, 6),
            "gap_score": round(gap_score, 6),
            "missing_skill_count": missing_count,
            "missing_skills": missing_skills,
            "direct_transition_probability": round(direct_probability, 6),
            "path_probability": round(path_probability, 6),
            "path_length": len(path) - 1 if path else -1,
            "growth_score": round(growth_score, 6),
            # 预计补全时间由 40 小时基础学习量、整体缺口强度和缺口技能数量共同估算。
            "estimated_training_hours": round(40 + 320 * gap_score + 12 * missing_count, 2),
            "recommendation_score": round(recommendation_score, 6),
            "is_target": int(candidate_job == target_job),
        })
    # 3. 综合分数降序排列；分数相同时按职位 ID 固定顺序，保证多次运行结果一致。
    rows.sort(key=lambda row: (-row["recommendation_score"], row["candidate_job"]))
    # 4. 排序完成后再写入从 1 开始的名次，并只返回请求的 Top-K。
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:top_k]


def _skill_gap(context: dict[str, Any], skill_vector: dict[str, float], target_job: str) -> dict[str, Any]:
    """比较临时技能向量与目标职位需求，列出所有正向缺口。"""

    # 1. 从完整职位技能关系表中筛选当前候选/目标职位的需求记录。
    table = context["occupation_skill"]
    selected = table[table["occupation_id"].astype(str) == target_job].copy()
    # 2. 根据 skill_id 映射本次简历临时技能水平；没有识别到的技能使用 0.10 保守基线。
    selected["current_level"] = selected["skill_id"].astype(str).map(skill_vector).fillna(0.1)
    # 3. 只计算正向差距，已满足或超过职位需求的技能不会被标记为缺口。
    selected["gap"] = (selected["demand_weight"] - selected["current_level"]).clip(lower=0.0)
    # 4. 过滤零缺口并按差距、需求权重降序，确保最需要补齐的技能优先展示。
    selected = selected[selected["gap"] > 0].sort_values(["gap", "demand_weight"], ascending=False)
    rows = []
    # 5. 输出中英文名称及三项数值，结构与已有用户缺口接口保持一致。
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
    strategy: str = "fast",
) -> dict[str, Any]:
    """分析上传文本并返回推荐、技能缺口和职业路径，不写入项目状态。

    数据链路：文本校验 → 技能别名匹配 → 已识别技能设为 0.75、其余为 0.10
    → 可选自动推断当前职位 → 推荐排序 → 确定目标职位 → 缺口/路径
    → 返回 ``persisted=false``。
    """

    # 1. 文本与职位参数校验：空文本、未知职位或未知路线策略立即返回清晰错误。
    if not isinstance(text, str) or not text.strip():
        raise ValueError("简历文本不能为空")
    context = _load_context(_root(root))
    valid_jobs = set(context["occupations"]["occupation_id"].astype(str))
    if current_job and str(current_job) not in valid_jobs:
        raise ValueError(f"未知当前职位：{current_job}")
    if target_job and str(target_job) not in valid_jobs:
        raise ValueError(f"未知目标职位：{target_job}")
    # fast 对应最少跳数，stable 对应最大累计概率；不允许静默回退，避免前端显示错误策略。
    if strategy not in {"fast", "stable"}:
        raise ValueError("路线策略必须是 fast 或 stable")

    # 技能识别：统一大小写和空白，按别名长度从长到短匹配，减少短词抢先命中。
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    skill_levels = {skill_id: 0.1 for skill_id in context["skill_ids"]}
    recognized_ids: set[str] = set()
    for alias, skill_id in sorted(context["aliases"].items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in normalized:
            recognized_ids.add(skill_id)
    # 临时向量：识别技能赋 0.75，未识别技能保留保守基线 0.10。
    for skill_id in recognized_ids:
        skill_levels[skill_id] = 0.75
    # 未指定当前职位时，选择与临时技能向量余弦相似度最高的职位。
    if current_job is None:
        vector = np.array([skill_levels[skill_id] for skill_id in context["skill_ids"]], dtype=float)
        current_job = max(
            valid_jobs,
            key=lambda job_id: _cosine(vector, context["job_vectors"].loc[job_id].to_numpy(dtype=float)),
        )
    current_job = str(current_job)
    # 4. 复用透明推荐公式；目标职位缺省时采用推荐首位，使后续缺口和路线都有明确终点。
    recommendations = _recommendations(context, skill_levels, current_job, target_job, max(1, min(int(top_k), 50)))
    if target_job is None:
        target_job = recommendations[0]["candidate_job"] if recommendations else current_job
    target_job = str(target_job)
    # 5. 目标职位缺口与路线分开计算：缺口解释“学什么”，路线解释“如何转到目标岗位”。
    gap = _skill_gap(context, skill_levels, target_job)
    path, path_probability = _path(context["graph"], current_job, target_job, strategy=strategy)
    # 返回值仅属于当前请求，persisted=False 是前端和调用方可检查的不落库声明。
    return {
        "status": "ok",
        "persisted": False,
        "current_job": current_job,
        "current_job_name": context["occupation_names"].get(current_job, current_job),
        "current_job_name_zh": OCCUPATION_NAMES_ZH.get(
            context["occupation_names"].get(current_job, current_job),
            context["occupation_names"].get(current_job, current_job),
        ),
        "target_job": target_job,
        "target_job_name": context["occupation_names"].get(target_job, target_job),
        "target_job_name_zh": OCCUPATION_NAMES_ZH.get(
            context["occupation_names"].get(target_job, target_job),
            context["occupation_names"].get(target_job, target_job),
        ),
        "skill_count": len(recognized_ids),
        # recognized_skills 仅包含文本中实际命中的技能，用于向用户解释简历识别结果。
        "recognized_skills": [
            {
                "skill_id": skill_id,
                "skill_name": context["skill_names_en"].get(skill_id, skill_id),
                "skill_name_zh": context["skill_names_zh"].get(skill_id, skill_id),
                "level": skill_levels[skill_id],
            }
            for skill_id in sorted(recognized_ids)
        ],
        # skill_profile 包含全部技能及其临时水平，并按水平降序排列，直接作为雷达图数据源。
        "skill_profile": [
            {
                "skill_id": skill_id,
                "skill_name": context["skill_names_en"].get(skill_id, skill_id),
                "skill_name_zh": context["skill_names_zh"].get(skill_id, skill_id),
                "level": skill_levels[skill_id],
            }
            for skill_id in sorted(
                context["skill_ids"],
                key=lambda item: (-skill_levels[item], context["skill_names_zh"].get(item, item)),
            )
        ],
        "occupations": sorted(valid_jobs),
        "recommendations": recommendations,
        "skill_gap": gap,
        # career_path 显式携带策略代码和中文名称，前端据此显示“快速路线/稳健路线”标签。
        "career_path": {
            "status": "ok",
            "current_job": current_job,
            "target_job": target_job,
            "strategy": strategy,
            "strategy_name": "快速路线" if strategy == "fast" else "稳健路线",
            "path": path,
            "path_names": [
                OCCUPATION_NAMES_ZH.get(
                    context["occupation_names"].get(job_id, job_id),
                    context["occupation_names"].get(job_id, job_id),
                )
                for job_id in path
            ],
            "path_probability": round(path_probability, 6),
            "path_length": len(path) - 1 if path else -1,
        },
        "data_note": "上传简历仅在本次请求内存中分析，不写入数据库或训练数据。",
    }


__all__ = ["SUPPORTED_EXTENSIONS", "analyze_resume_text"]
