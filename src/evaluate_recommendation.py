"""Evaluate the small recommendation extract with ranking metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def evaluate_at_k(features: pd.DataFrame, k: int) -> dict[str, float]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "recommendation_features.csv")
    parser.add_argument("--out-file", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "evaluation_summary.csv")
    args = parser.parse_args()
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.feature_file)
    result = pd.DataFrame([evaluate_at_k(features, 1), evaluate_at_k(features, 3), evaluate_at_k(features, 5)])
    result.to_csv(args.out_file, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
