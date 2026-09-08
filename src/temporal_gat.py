"""Minimal TemporalGAT-compatible model skeleton for the second-week work."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None


if nn is not None:

    class TemporalGAT(nn.Module):
        """Encode per-skill histories, aggregate graph neighbours, predict next levels.

        ``x`` has shape ``[batch, time, num_skills]``.  ``edge_index`` is a
        skill-to-skill graph with shape ``[2, edges]``; graph inputs are
        optional so the model can also serve as a temporal-only baseline.
        """

        def __init__(self, num_skills: int, hidden_dim: int = 32, heads: int = 2):
            super().__init__()
            if num_skills < 1 or hidden_dim < 1 or heads < 1:
                raise ValueError("num_skills, hidden_dim, and heads must be positive")
            if hidden_dim % heads:
                raise ValueError("hidden_dim must be divisible by heads")
            self.num_skills = num_skills
            self.hidden_dim = hidden_dim
            self.heads = heads
            self.temporal = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
            head_dim = hidden_dim // heads
            self.heads_proj = nn.ModuleList([nn.Linear(hidden_dim, head_dim) for _ in range(heads)])
            self.output = nn.Linear(hidden_dim, 1)

        def _aggregate(self, node_features: Tensor, edge_index: Optional[Tensor], edge_weight: Optional[Tensor]) -> Tensor:
            if edge_index is None or edge_index.numel() == 0:
                return node_features
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape [2, edges]")
            batch_size, num_nodes, hidden_dim = node_features.shape
            source, target = edge_index.to(node_features.device)
            valid = (source >= 0) & (source < num_nodes) & (target >= 0) & (target < num_nodes)
            source, target = source[valid], target[valid]
            if edge_weight is None:
                weights = torch.ones(source.shape[0], dtype=node_features.dtype, device=node_features.device)
            else:
                weights = edge_weight.to(node_features.device, dtype=node_features.dtype)[valid]
            aggregate = node_features.clone()
            degree = torch.ones(num_nodes, dtype=node_features.dtype, device=node_features.device)
            for index in range(source.shape[0]):
                aggregate[:, target[index], :] += node_features[:, source[index], :] * weights[index]
                degree[target[index]] += weights[index]
            return aggregate / degree.view(1, -1, 1).clamp_min(1e-6)

        def forward(self, x: Tensor, edge_index: Optional[Tensor] = None, edge_weight: Optional[Tensor] = None) -> Tensor:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim != 3 or x.shape[-1] != self.num_skills:
                raise ValueError(f"x must have shape [batch, time, {self.num_skills}]")
            batch_size, _, num_skills = x.shape
            # Treat each skill as a one-dimensional temporal sequence.
            temporal_input = x.transpose(1, 2).reshape(batch_size * num_skills, x.shape[1], 1)
            _, hidden = self.temporal(temporal_input)
            hidden = hidden[-1].reshape(batch_size, num_skills, self.hidden_dim)
            hidden = self._aggregate(hidden, edge_index, edge_weight)
            projected = torch.cat([projection(hidden) for projection in self.heads_proj], dim=-1)
            return torch.sigmoid(self.output(projected).squeeze(-1))

else:

    class TemporalGAT:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed; install requirements-optional-models.txt")


__all__ = ["TemporalGAT"]
