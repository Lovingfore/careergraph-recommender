"""TemporalGAT 教学骨架的轻量、可复现实验训练与评估入口。

数据窗口按数据集当前的确定性样本顺序消费，先训练再验证和测试；产物包括
checkpoint、训练日志和评价 JSON。当前样本顺序先按用户分组、再按用户内月份
排列，因此这不是严格的全局时间留出。评价对象是确定性合成教学数据，结果
用于流程验收，不应解释为真实招聘市场效果。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_training_plan(num_samples: int) -> dict[str, list[int]]:
    """按当前样本列表顺序切分 60/20/20 的训练、验证、测试索引。

    此函数只保证不随机打乱；当前数据集先按 ``user_id`` 分组，再在每个用户内
    按月份生成窗口，所以不能把该切分解释为严格的全局时间留出。样本很少时
    仍保证三份数据各至少一条；返回值只包含索引，不复制样本内容。
    """

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
    """导入训练所需的可选 PyTorch/nn，并将缺失转成可读的运行时错误。"""
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is not installed; install requirements-optional-models.txt") from exc
    return torch, nn


def _set_seed(seed: int, torch: Any) -> None:
    """同时固定 Python、NumPy、PyTorch（含 CUDA）的随机种子以复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA is optional
        torch.cuda.manual_seed_all(seed)


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """计算并四舍五入 MAE 与 RMSE，供训练日志和评价摘要复用。"""
    error = predictions - targets
    return {
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 6),
    }


def _linear_baseline(x: np.ndarray) -> np.ndarray:
    """按每项技能历史端点斜率外推一步，作为透明的线性基线。"""

    if x.shape[1] == 1:
        return np.clip(x[:, -1, :], 0.0, 1.0)
    # The loader returns [samples, time, skills].  An endpoint slope keeps
    # this baseline vectorized and avoids a dependency on sklearn.
    slopes = (x[:, -1, :] - x[:, 0, :]) / float(x.shape[1] - 1)
    return np.clip(x[:, -1, :] + slopes, 0.0, 1.0)


def _collect_dataset(data_dir: Path):
    """读取 clean 事件并收集 ``SkillSequenceDataset`` 的全部窗口样本。"""
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
    """按确定性样本顺序执行 TemporalGAT 训练，并返回评价指标与产物摘要。"""

    # 1) 先检查可选依赖，避免已经创建目录或写文件后才发现无法训练。
    torch, nn = _require_torch()
    try:
        from src.data_loader import build_graph_tensors
        from src.temporal_gat import TemporalGAT
    except ModuleNotFoundError:  # Support direct script execution.
        from data_loader import build_graph_tensors
        from temporal_gat import TemporalGAT

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    # 2) 固定随机性、准备产物目录，并从事件表收集时序窗口。
    _set_seed(seed, torch)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    samples, skill_ids = _collect_dataset(Path(data_dir))
    # 3) 按当前样本列表原序切分（不随机）；x/y 保持
    # [样本, 时间, 技能] 与 [样本, 技能]。当前列表主要按用户分组。
    plan = build_training_plan(len(samples))
    x_all = torch.stack([sample["x"] for sample in samples]).float()
    y_all = torch.stack([sample["y"] for sample in samples]).float()
    graph = build_graph_tensors(Path(data_dir))
    edge_index = graph["skill_edge_index"]
    edge_weight = graph["skill_edge_weight"]

    # 4) 初始化模型、优化器和损失，训练阶段只使用 train 索引。
    model = TemporalGAT(num_skills=len(skill_ids), hidden_dim=hidden_dim, heads=heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    log_rows: list[dict[str, float | int]] = []

    def predict(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """关闭梯度后批量收集指定索引的预测与目标 NumPy 数组。"""
        model.eval()
        with torch.no_grad():
            predictions = model(x_all[indices], edge_index=edge_index, edge_weight=edge_weight)
        return predictions.detach().cpu().numpy(), y_all[indices].cpu().numpy()

    # 5) 每轮更新训练集参数，再立即在验证集评估并记录日志。
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

    # 6) 训练完成后只在测试切片上做一次最终比较，同时计算线性基线。
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
    # 7) 持久化 checkpoint、逐轮 CSV 日志和 JSON 摘要，供 Web/验收读取。
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
    """解析 CLI 参数并打印训练 JSON；成功返回 0，解析或训练失败返回非零。

    argparse 参数错误通常以退出码 2 结束；训练异常不在此处吞掉，由解释器传播
    并产生非零退出码，便于自动化脚本识别失败。
    """
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
