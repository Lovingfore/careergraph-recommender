"""Build compact user-job features for Topic 17 recommendation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def max_path_probability(graph: dict[str, list[tuple[str, float]]], source: str, target: str, max_hops: int = 3) -> tuple[float, int]:
    if source == target:
        return 1.0, 0
    best_probability, best_hops = 0.0, max_hops + 1
    def visit(node: str, path: list[str], probability: float):
        nonlocal best_probability, best_hops
        if node == target and len(path) > 1:
            hops = len(path) - 1
            if probability > best_probability:
                best_probability, best_hops = probability, hops
            return
        if len(path) - 1 >= max_hops:
            return
        for next_node, edge_probability in graph.get(node, []):
            if next_node not in path:
                visit(next_node, path + [next_node], probability * edge_probability)

    visit(source, [source], 1.0)
    return best_probability, best_hops if best_probability else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "clean")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    occupations = pd.read_csv(args.data_dir / "occupations.csv")
    occ_skill = pd.read_csv(args.data_dir / "occupation_skill.csv")
    events = pd.read_csv(args.data_dir / "user_skill_events.csv")
    profiles = pd.read_csv(args.data_dir / "user_profiles.csv")
    transitions = pd.read_csv(args.data_dir / "job_transitions.csv")
    occupation_names = occupations.set_index("occupation_id")["occupation_name"].to_dict()
    skill_ids = sorted(occ_skill["skill_id"].unique())

    job_vectors = occ_skill.pivot_table(index="occupation_id", columns="skill_id", values="demand_weight", fill_value=0).reindex(columns=skill_ids, fill_value=0)
    job_vectors.to_csv(args.out_dir / "job_feature_matrix.csv", encoding="utf-8-sig")
    latest = events.sort_values("month").groupby(["user_id", "skill_id"], as_index=False).tail(1)
    user_vectors = latest.pivot_table(index="user_id", columns="skill_id", values="level", fill_value=0).reindex(columns=skill_ids, fill_value=0)
    user_vectors.to_csv(args.out_dir / "user_feature_matrix.csv", encoding="utf-8-sig")

    graph: dict[str, list[tuple[str, float]]] = {}
    for row in transitions.itertuples(index=False):
        graph.setdefault(row.from_job, []).append((row.to_job, float(row.transition_probability)))
    (args.out_dir / "transition_graph.json").write_text(
        json.dumps({key: [{"to_job": target, "probability": probability} for target, probability in values] for key, values in graph.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    job_ids = list(job_vectors.index)
    for profile in profiles.to_dict("records"):
        user_vec = user_vectors.loc[profile["user_id"]].to_numpy(dtype=float)
        current_vec = job_vectors.loc[profile["current_job"]].to_numpy(dtype=float)
        for job_id in job_ids:
            if job_id == profile["current_job"]:
                continue
            job_vec = job_vectors.loc[job_id].to_numpy(dtype=float)
            gap_vector = np.maximum(job_vec - user_vec, 0)
            demand_sum = float(job_vec.sum()) or 1.0
            gap_score = float(np.dot(gap_vector, job_vec) / demand_sum)
            missing_count = int(((job_vec >= 0.28) & (user_vec < 0.28)).sum())
            match_score = cosine(user_vec, job_vec)
            direct_probability = float(transitions.loc[(transitions["from_job"] == profile["current_job"]) & (transitions["to_job"] == job_id), "transition_probability"].sum())
            path_probability, path_length = max_path_probability(graph, profile["current_job"], job_id)
            growth_score = float(np.clip(0.5 + job_vec.mean() - current_vec.mean(), 0, 1))
            estimated_hours = float(round(40 + 320 * gap_score + 12 * missing_count, 2))
            recommendation_score = float(0.45 * match_score - 0.25 * gap_score + 0.15 * growth_score + 0.15 * path_probability)
            rows.append({
                "user_id": profile["user_id"],
                "current_job": profile["current_job"],
                "current_job_name": occupation_names.get(profile["current_job"], profile["current_job"]),
                "target_job": profile["target_job"],
                "target_job_name": occupation_names.get(profile["target_job"], profile["target_job"]),
                "candidate_job": job_id,
                "candidate_job_name": occupation_names.get(job_id, job_id),
                "match_score": round(match_score, 6),
                "gap_score": round(gap_score, 6),
                "missing_skill_count": missing_count,
                "direct_transition_probability": round(direct_probability, 6),
                "path_probability": round(path_probability, 6),
                "path_length": path_length,
                "growth_score": round(growth_score, 6),
                "estimated_training_hours": estimated_hours,
                "recommendation_score": round(recommendation_score, 6),
                "is_target": int(job_id == profile["target_job"]),
            })
    features = pd.DataFrame(rows)
    features["rank"] = features.groupby("user_id")["recommendation_score"].rank(method="first", ascending=False).astype(int)
    features.sort_values(["user_id", "rank"]).to_csv(args.out_dir / "recommendation_features.csv", index=False, encoding="utf-8-sig")
    features[features["rank"] <= 5].sort_values(["user_id", "rank"]).to_csv(args.out_dir / "topk_recommendations.csv", index=False, encoding="utf-8-sig")

    dictionary = pd.DataFrame([
        ["match_score", "余弦相似度", "用户技能向量与职位需求向量的相似度"],
        ["gap_score", "加权技能缺口", "目标职位要求超过用户当前水平的差距"],
        ["missing_skill_count", "缺口技能数量", "需求权重较高但用户水平不足的技能个数"],
        ["path_probability", "路径可行性", "职位转移图上最多三跳路径的最大概率乘积"],
        ["growth_score", "职业提升空间", "目标职位平均要求相对当前职位的归一化提升"],
        ["estimated_training_hours", "预计补全时间", "40 + 320*gap_score + 12*missing_skill_count"],
        ["recommendation_score", "推荐总分", "0.45 Match - 0.25 Gap + 0.15 Growth + 0.15 Path"],
    ], columns=["feature", "name", "definition"])
    dictionary.to_csv(args.out_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(features)} user-job feature rows to {args.out_dir}")


if __name__ == "__main__":
    main()
