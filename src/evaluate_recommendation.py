"""使用 Precision、Recall 和 NDCG 评估离线推荐排序结果。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def evaluate_at_k(features: pd.DataFrame, k: int) -> dict[str, float]:
    """计算所有用户的 Precision@K、Recall@K 与 NDCG@K 均值。

    ``is_target=1`` 的候选职位视为相关结果；Precision@K 衡量前 K 位命中比例，
    Recall@K 表示目标职位是否进入前 K 位，NDCG@K 根据命中排名给予对数折损。
    """

    precision, recall, ndcg = [], [], []
    for _, group in features.groupby("user_id"):
        ranked = group.sort_values("rank").head(k)
        hits = int(ranked["is_target"].sum())
        precision.append(hits / k)
        recall.append(float(hits > 0))
        target_rows = ranked.loc[ranked["is_target"] == 1]
        if target_rows.empty:
            ndcg.append(0.0)
        else:
            rank = int(target_rows.iloc[0]["rank"])
            ndcg.append(1.0 / math.log2(rank + 1))
    return {
        "k": k,
        "precision_at_k": round(sum(precision) / len(precision), 6),
        "recall_at_k": round(sum(recall) / len(recall), 6),
        "ndcg_at_k": round(sum(ndcg) / len(ndcg), 6),
    }


def main():
    """在 K=1、3、5 三个截断位置评估排序并写出汇总 CSV。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "recommendation_features.csv")
    parser.add_argument("--out-file", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "evaluation_summary.csv")
    args = parser.parse_args()
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.feature_file)
    # 多个 K 同时展示首位准确度与更宽推荐列表的召回变化，便于课程实验比较。
    result = pd.DataFrame([evaluate_at_k(features, 1), evaluate_at_k(features, 3), evaluate_at_k(features, 5)])
    result.to_csv(args.out_file, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
