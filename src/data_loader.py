"""将 clean CSV 转换为 PyTorch 时序样本和图边张量。

PyTorch 是可选的模型依赖：没有它时仍可导入本模块，但真正构造
``Dataset``、``DataLoader`` 或张量会给出清晰错误，而不会静默地产生错误
类型。输入主要来自 ``user_skill_events.csv``、技能/职位字典和两类图边表。
"""

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
    """确认可选 PyTorch 已安装，并返回模块对象供调用方使用。"""
    if torch is None:
        raise RuntimeError("PyTorch is not installed; install requirements-optional-models.txt")
    return torch


class SkillSequenceDataset(Dataset):
    """按用户构造连续月份技能窗口，并以紧随其后的月份作为监督目标。

    输入事件至少包含 ``user_id``、``month``、``skill_id``、``level``。每位
    用户的事件会按月份补齐，再按固定技能字典编码为三维序列结构
    ``[样本, 时间, 技能]``：每个样本的 ``x`` 是连续
    ``sequence_length`` 个月的 ``[时间, 技能]`` 窗口，``y`` 是下一月的
    ``[技能]`` 向量；缺失月份或技能以 0 填充，等级裁剪到 [0, 1]。
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

        # 先按用户、月份建立稠密向量，确保中间缺失月份不会改变窗口长度。
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
            # 滑动窗口的最后一个输入月为 start+sequence_length-1，下一月
            # start+sequence_length 作为 y，因此不会跨用户或跨空洞拼接。
            for start in range(min_month, max_month - sequence_length + 1):
                x = np.stack([vectors[month] for month in range(start, start + sequence_length)], axis=0)
                y = vectors[start + sequence_length].copy()
                self.samples.append((str(user_id), x, y))

    def __len__(self) -> int:
        """返回可用的连续窗口数量。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回一个样本：``x`` 为 ``[时间, 技能]``，``y`` 为 ``[技能]``，并附用户 ID。"""
        user_id, x, y = self.samples[index]
        return {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "user_id": user_id,
        }


def _read_clean(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取技能字典、用户技能事件和职位-技能关系三张 clean 表。"""
    data_dir = Path(data_dir)
    skills = pd.read_csv(data_dir / "skills.csv")
    events = pd.read_csv(data_dir / "user_skill_events.csv")
    occupation_skill = pd.read_csv(data_dir / "occupation_skill.csv")
    return skills, events, occupation_skill


def build_dataloader(data_dir: Path, batch_size: int = 2, sequence_length: int = 3, shuffle: bool = False):
    """构造 ``DataLoader`` 及其 metadata，并保留技能 ID 的稳定排序。

    metadata 记录技能数量/ID、窗口长度、样本数和 batch 大小，训练与验收
    可据此解释批量张量形状；默认不打乱，便于时序数据复现和顺序切分。
    """

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
    """将边列表编码为 ``edge_index=[2, edge_count]`` 和 float32 权重。"""
    _require_torch()
    if index:
        edge_index = torch.tensor(index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    return edge_index, edge_weight


def build_graph_tensors(data_dir: Path) -> dict[str, Any]:
    """根据职位/技能 ID 映射构造三类稳定图张量。

    职位-技能边和职位转移边分别使用各自的整数 ID 映射，权重为
    ``demand_weight`` 与 ``transition_probability``；同时将同一职位共同
    要求的技能投影成技能-技能边。每类 ``edge_index`` 都是长整型
    ``[2, edge_count]``，对应的 ``edge_weight`` 是 float32 一维张量。
    """

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

    # TemporalGAT 的节点是技能，因此把每个职位内共同要求的技能投影成
    # 有向技能边；边权取两端需求权重的较小值，保留共同需求的保守强度。
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
