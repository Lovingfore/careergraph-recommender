"""PyTorch-ready temporal skill sequences and graph edge tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Keep the data preparation layer importable without the optional model stack.
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for SkillSequenceDataset")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is not installed; install requirements-optional-models.txt")
    return torch


class SkillSequenceDataset(Dataset):
    """Contiguous user skill windows with the following month as target.

    Each sample contains ``sequence_length`` months of zero-filled skill
    vectors and a target vector for the immediately following month.
    """

    def __init__(self, events: pd.DataFrame, skill_ids: list[str], sequence_length: int = 3):
        _require_torch()
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        required = {"user_id", "month", "skill_id", "level"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"events is missing columns: {sorted(missing)}")
        self.skill_ids = list(skill_ids)
        self.skill_to_index = {skill_id: index for index, skill_id in enumerate(self.skill_ids)}
        self.sequence_length = sequence_length
        self.samples: list[tuple[str, np.ndarray, np.ndarray]] = []

        for user_id, user_events in events.groupby("user_id", sort=True):
            user_events = user_events.copy()
            user_events["month"] = pd.to_numeric(user_events["month"], errors="raise").astype(int)
            user_events["level"] = pd.to_numeric(user_events["level"], errors="raise").astype(float)
            if user_events.empty:
                continue
            min_month, max_month = int(user_events["month"].min()), int(user_events["month"].max())
            vectors: dict[int, np.ndarray] = {}
            for month in range(min_month, max_month + 1):
                vectors[month] = np.zeros(len(self.skill_ids), dtype=np.float32)
            for row in user_events.itertuples(index=False):
                skill_index = self.skill_to_index.get(str(row.skill_id))
                if skill_index is not None:
                    vectors[int(row.month)][skill_index] = float(np.clip(row.level, 0.0, 1.0))
            for start in range(min_month, max_month - sequence_length + 1):
                x = np.stack([vectors[month] for month in range(start, start + sequence_length)], axis=0)
                y = vectors[start + sequence_length].copy()
                self.samples.append((str(user_id), x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user_id, x, y = self.samples[index]
        return {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "user_id": user_id,
        }


def _read_clean(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    skills = pd.read_csv(data_dir / "skills.csv")
    events = pd.read_csv(data_dir / "user_skill_events.csv")
    occupation_skill = pd.read_csv(data_dir / "occupation_skill.csv")
    return skills, events, occupation_skill


def build_dataloader(data_dir: Path, batch_size: int = 2, sequence_length: int = 3, shuffle: bool = False):
    """Build a deterministic DataLoader and metadata from clean CSV tables."""

    _require_torch()
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    skills, events, _ = _read_clean(Path(data_dir))
    skill_ids = sorted(skills["skill_id"].astype(str).unique().tolist())
    dataset = SkillSequenceDataset(events, skill_ids, sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    metadata = {
        "num_skills": len(skill_ids),
        "skill_ids": skill_ids,
        "sequence_length": sequence_length,
        "num_samples": len(dataset),
        "batch_size": batch_size,
    }
    return loader, metadata


def _edge_tensor(index: list[tuple[int, int]], weights: list[float]):
    _require_torch()
    if index:
        edge_index = torch.tensor(index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    return edge_index, edge_weight


def build_graph_tensors(data_dir: Path) -> dict[str, Any]:
    """Build stable occupation-skill, transition, and skill co-occurrence edges."""

    _require_torch()
    data_dir = Path(data_dir)
    occupations = pd.read_csv(data_dir / "occupations.csv")
    skills = pd.read_csv(data_dir / "skills.csv")
    occupation_skill = pd.read_csv(data_dir / "occupation_skill.csv")
    transitions = pd.read_csv(data_dir / "job_transitions.csv")
    occupation_ids = sorted(occupations["occupation_id"].astype(str).unique().tolist())
    skill_ids = sorted(skills["skill_id"].astype(str).unique().tolist())
    occupation_to_index = {value: index for index, value in enumerate(occupation_ids)}
    skill_to_index = {value: index for index, value in enumerate(skill_ids)}
    job_to_index = occupation_to_index

    occupation_edges: list[tuple[int, int]] = []
    occupation_weights: list[float] = []
    for row in occupation_skill.itertuples(index=False):
        if row.occupation_id in occupation_to_index and row.skill_id in skill_to_index:
            occupation_edges.append((occupation_to_index[row.occupation_id], skill_to_index[row.skill_id]))
            occupation_weights.append(float(row.demand_weight))

    transition_edges: list[tuple[int, int]] = []
    transition_weights: list[float] = []
    for row in transitions.itertuples(index=False):
        if row.from_job in job_to_index and row.to_job in job_to_index:
            transition_edges.append((job_to_index[row.from_job], job_to_index[row.to_job]))
            transition_weights.append(float(row.transition_probability))

    # A skill-only projection is useful to TemporalGAT, whose nodes are skills.
    # Connect skills co-required by each occupation, weighted by demand.
    skill_pair_weights: dict[tuple[int, int], float] = {}
    for _, group in occupation_skill.groupby("occupation_id", sort=True):
        pairs = list(group[["skill_id", "demand_weight"]].itertuples(index=False, name=None))
        for source_id, source_weight in pairs:
            for target_id, target_weight in pairs:
                if source_id == target_id:
                    continue
                key = (skill_to_index[source_id], skill_to_index[target_id])
                skill_pair_weights[key] = max(skill_pair_weights.get(key, 0.0), float(min(source_weight, target_weight)))
    skill_edges = list(skill_pair_weights)
    skill_weights = [skill_pair_weights[key] for key in skill_edges]

    occupation_edge_index, occupation_edge_weight = _edge_tensor(occupation_edges, occupation_weights)
    transition_edge_index, transition_edge_weight = _edge_tensor(transition_edges, transition_weights)
    skill_edge_index, skill_edge_weight = _edge_tensor(skill_edges, skill_weights)
    return {
        "occupation_ids": occupation_ids,
        "skill_ids": skill_ids,
        "occupation_id_to_index": occupation_to_index,
        "skill_id_to_index": skill_to_index,
        "occupation_skill_edge_index": occupation_edge_index,
        "occupation_skill_edge_weight": occupation_edge_weight,
        "transition_edge_index": transition_edge_index,
        "transition_edge_weight": transition_edge_weight,
        "skill_edge_index": skill_edge_index,
        "skill_edge_weight": skill_edge_weight,
    }


__all__ = ["SkillSequenceDataset", "build_dataloader", "build_graph_tensors"]
