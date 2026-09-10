"""Lightweight, reproducible training and evaluation for the TemporalGAT skeleton."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_training_plan(num_samples: int) -> dict[str, list[int]]:
    """Split ordered samples into train/validation/test partitions."""

    if num_samples < 3:
        raise ValueError("at least 3 ordered samples are required")
    train_count = max(1, int(num_samples * 0.6))
    validation_count = max(1, int(num_samples * 0.2))
    if train_count + validation_count >= num_samples:
        validation_count = 1
        train_count = num_samples - 2
    train_end = train_count
    validation_end = train_end + validation_count
    indices = list(range(num_samples))
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def _require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is not installed; install requirements-optional-models.txt") from exc
    return torch, nn


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA is optional
        torch.cuda.manual_seed_all(seed)


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    error = predictions - targets
    return {
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 6),
    }


def _linear_baseline(x: np.ndarray) -> np.ndarray:
    """Extrapolate one step from each skill's short history."""

    if x.shape[1] == 1:
        return np.clip(x[:, -1, :], 0.0, 1.0)
    # The loader returns [samples, time, skills].  An endpoint slope keeps
    # this baseline vectorized and avoids a dependency on sklearn.
    slopes = (x[:, -1, :] - x[:, 0, :]) / float(x.shape[1] - 1)
    return np.clip(x[:, -1, :] + slopes, 0.0, 1.0)


def _collect_dataset(data_dir: Path):
    try:
        from src.data_loader import SkillSequenceDataset, _read_clean
    except ModuleNotFoundError:  # Support ``python src/train_temporal_gat.py``.
        from data_loader import SkillSequenceDataset, _read_clean

    skills, events, _ = _read_clean(Path(data_dir))
    skill_ids = sorted(skills["skill_id"].astype(str).unique().tolist())
    dataset = SkillSequenceDataset(events, skill_ids, sequence_length=3)
    samples = [dataset[index] for index in range(len(dataset))]
    if len(samples) < 3:
        raise ValueError("at least 3 ordered skill windows are required for training")
    return samples, skill_ids


def run_training(
    data_dir: Path,
    artifact_dir: Path,
    epochs: int = 30,
    seed: int = 42,
    hidden_dim: int = 16,
    heads: int = 2,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """Train a small TemporalGAT and return chronological evaluation metrics."""

    torch, nn = _require_torch()
    try:
        from src.data_loader import build_graph_tensors
        from src.temporal_gat import TemporalGAT
    except ModuleNotFoundError:  # Support direct script execution.
        from data_loader import build_graph_tensors
        from temporal_gat import TemporalGAT

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    _set_seed(seed, torch)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    samples, skill_ids = _collect_dataset(Path(data_dir))
    plan = build_training_plan(len(samples))
    x_all = torch.stack([sample["x"] for sample in samples]).float()
    y_all = torch.stack([sample["y"] for sample in samples]).float()
    graph = build_graph_tensors(Path(data_dir))
    edge_index = graph["skill_edge_index"]
    edge_weight = graph["skill_edge_weight"]

    model = TemporalGAT(num_skills=len(skill_ids), hidden_dim=hidden_dim, heads=heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    log_rows: list[dict[str, float | int]] = []

    def predict(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            predictions = model(x_all[indices], edge_index=edge_index, edge_weight=edge_weight)
        return predictions.detach().cpu().numpy(), y_all[indices].cpu().numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_all[plan["train"]], edge_index=edge_index, edge_weight=edge_weight)
        loss = criterion(predictions, y_all[plan["train"]])
        loss.backward()
        optimizer.step()
        validation_predictions, validation_targets = predict(plan["validation"])
        validation = _metrics(validation_predictions, validation_targets)
        log_rows.append({"epoch": epoch, "train_loss": round(float(loss.item()), 6), "validation_mae": validation["mae"], "validation_rmse": validation["rmse"]})

    test_predictions, test_targets = predict(plan["test"])
    baseline_predictions = _linear_baseline(x_all[plan["test"]].cpu().numpy())
    result: dict[str, Any] = {
        "status": "trained",
        "seed": seed,
        "epochs": epochs,
        "sample_count": len(samples),
        "split": {key: len(value) for key, value in plan.items()},
        "baseline": _metrics(baseline_predictions, test_targets),
        "temporal_gat": _metrics(test_predictions, test_targets),
        "test": _metrics(test_predictions, test_targets),
        "data_note": "Evaluation uses deterministic teaching data, not real user outcomes.",
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_skills": len(skill_ids),
            "skill_ids": skill_ids,
            "hidden_dim": hidden_dim,
            "heads": heads,
            "seed": seed,
        },
        artifact_dir / "temporal_gat.pt",
    )
    pd.DataFrame(log_rows).to_csv(artifact_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    (artifact_dir / "evaluation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "clean")
    parser.add_argument("--artifact-dir", type=Path, default=root / "artifacts" / "models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_training(args.data_dir, args.artifact_dir, epochs=args.epochs, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
