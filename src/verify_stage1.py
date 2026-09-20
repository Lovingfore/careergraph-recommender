"""Topic 17 第一阶段交付物的组合式验收检查。

脚本把 clean 数据、SQLite 六张表、特征产物、可选 DataLoader、二部图、
Web 文件/API、模型产物和中文简历数据集分别检查，最后输出可机器读取的
JSON；它只读取并报告状态，不负责修复或重建产物。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from data_integrity import validate_clean_dataset, validate_feature_outputs
    from database import get_counts
    from data_loader import build_dataloader
except ModuleNotFoundError:  # Support ``import src.verify_stage1``.
    from src.data_integrity import validate_clean_dataset, validate_feature_outputs
    from src.database import get_counts
    from src.data_loader import build_dataloader


EXPECTED_TABLES = ("occupations", "skills", "occupation_skill", "user_profiles", "user_skill_events", "job_transitions")


def _database_report(db_path: Path) -> dict[str, Any]:
    """验收 SQLite 文件可读且六张核心表均非空，返回行数和缺失表。"""
    try:
        counts = get_counts(db_path)
        missing = [table for table in EXPECTED_TABLES if counts.get(table, 0) <= 0]
        return {"ok": not missing, "counts": counts, "missing_or_empty_tables": missing}
    except Exception as exc:
        return {"ok": False, "counts": {}, "error": str(exc)}


def _loader_report(root: Path) -> dict[str, Any]:
    """尝试构造 DataLoader，报告样本/技能/窗口元数据和首批张量形状。"""
    try:
        loader, metadata = build_dataloader(root / "data" / "clean", batch_size=2, sequence_length=3)
        first_batch = next(iter(loader), None)
        return {
            "ok": True,
            "status": "available",
            "num_samples": metadata["num_samples"],
            "num_skills": metadata["num_skills"],
            "sequence_length": metadata["sequence_length"],
            "first_batch_shape": list(first_batch["x"].shape) if first_batch is not None else [],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "optional_dependency_unavailable",
            "error": str(exc),
        }


def _web_contract_report(root: Path) -> dict[str, Any]:
    """检查 Django 启动文件、模板和约定的页面/API/图谱路由是否存在。"""
    required = [
        root / "web" / "manage.py",
        root / "web" / "topic17_web" / "settings.py",
        root / "web" / "topic17_web" / "urls.py",
        root / "web" / "topic17_web" / "wsgi.py",
        root / "web" / "topic17_app" / "views.py",
        root / "web" / "topic17_app" / "urls.py",
        root / "web" / "topic17_app" / "templates" / "topic17_app" / "index.html",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    route_text = ""
    if not missing:
        route_text = (root / "web" / "topic17_app" / "urls.py").read_text(encoding="utf-8")
    required_routes = [
        'path("", index',
        'path("api/summary/", summary_api',
        'path("api/model-info/", model_info_api',
        'path("api/occupations/", occupations_api',
        'path("api/resume-upload/", resume_upload_api',
        'path("api/recommend/", recommendation_api',
        'path("api/skill-gap/", skill_gap_api',
        'path("api/career-path/", career_path_api',
        'path("api/forecast/", forecast_api',
        'path("bipartite-graph.svg", bipartite_graph',
    ]
    routes_ok = all(route in route_text for route in required_routes)
    return {"ok": not missing and routes_ok, "required_files": [str(path.relative_to(root)) for path in required], "missing_files": missing, "routes_ok": routes_ok}


def _model_report(root: Path) -> dict[str, Any]:
    """检查 TemporalGAT checkpoint 与评价 JSON，区分已训练和可选依赖缺失。"""
    model_dir = root / "artifacts" / "models"
    checkpoint = model_dir / "temporal_gat.pt"
    evaluation = model_dir / "evaluation_summary.json"
    if checkpoint.exists() and evaluation.exists():
        try:
            summary = json.loads(evaluation.read_text(encoding="utf-8"))
            return {
                "ok": summary.get("status") == "trained",
                "status": "trained" if summary.get("status") == "trained" else "not_trained",
                "checkpoint": str(checkpoint.relative_to(root)),
                "evaluation": str(evaluation.relative_to(root)),
                "test": summary.get("test", {}),
            }
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "status": "not_trained", "error": str(exc)}
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return {"ok": False, "status": "optional_dependency_unavailable", "error": "PyTorch is not installed; install requirements-optional-models.txt"}
    return {"ok": False, "status": "not_trained", "error": "run src/train_temporal_gat.py to create a checkpoint"}


def _web_api_report(root: Path) -> dict[str, Any]:
    """核对 views.py 中 API 视图名称与 urls.py 中公开路径的对应关系。"""
    views_path = root / "web" / "topic17_app" / "views.py"
    urls_path = root / "web" / "topic17_app" / "urls.py"
    if not views_path.exists() or not urls_path.exists():
        return {"ok": False, "error": "Web API source files are missing"}
    views_text = views_path.read_text(encoding="utf-8")
    urls_text = urls_path.read_text(encoding="utf-8")
    required_views = ["model_info_api", "occupations_api", "resume_upload_api", "recommendation_api", "skill_gap_api", "career_path_api", "forecast_api", "bipartite_graph"]
    required_routes = ["api/model-info/", "api/occupations/", "api/resume-upload/", "api/recommend/", "api/skill-gap/", "api/career-path/", "api/forecast/", "bipartite-graph.svg"]
    ok = all(name in views_text for name in required_views) and all(route in urls_text for route in required_routes)
    return {
        "ok": ok,
        "routes": [
            "/api/model-info/",
            "/api/occupations/",
            "/api/resume-upload/",
            "/api/recommend/",
            "/api/skill-gap/",
            "/api/career-path/",
            "/api/forecast/",
            "/bipartite-graph.svg",
        ],
        "missing_views": [name for name in required_views if name not in views_text],
        "missing_routes": [route for route in required_routes if route not in urls_text],
    }


def _resume_dataset_report(root: Path) -> dict[str, Any]:
    """验收中文简历 CSV 的 300 行、中文文本、来源字段和唯一 ID。"""
    path = root / "data" / "clean" / "resumes_zh.csv"
    if not path.exists():
        return {"ok": False, "rows": 0, "error": "resumes_zh.csv is missing"}
    try:
        import pandas as pd

        resumes = pd.read_csv(path)
        chinese_rows = resumes["resume_text"].astype(str).map(
            lambda value: any("\u4e00" <= char <= "\u9fff" for char in value)
        )
        source_ok = resumes["source"].astype(str).eq("synthetic_chinese_resume").all()
        unique_ok = resumes["resume_id"].nunique() == len(resumes) and resumes["user_id"].nunique() == len(resumes)
        return {
            "ok": len(resumes) == 300 and bool(chinese_rows.all()) and bool(source_ok) and unique_ok,
            "rows": int(len(resumes)),
            "chinese_text_rows": int(chinese_rows.sum()),
            "source": "synthetic_chinese_resume",
            "unique_ids": unique_ok,
        }
    except Exception as exc:
        return {"ok": False, "rows": 0, "error": str(exc)}


def _bipartite_graph_report(root: Path) -> dict[str, Any]:
    """重新构造二部图并检查节点、边及孤立节点计数是否满足阶段要求。"""
    try:
        from bipartite_graph import build_bipartite_graph
    except ModuleNotFoundError:
        from src.bipartite_graph import build_bipartite_graph
    try:
        graph = build_bipartite_graph(root / "data" / "clean", min_demand_weight=0.30)
        return {
            "ok": graph["summary"]["occupation_count"] > 0
            and graph["summary"]["skill_count"] > 0
            and graph["summary"]["edge_count"] > 0
            and graph["summary"]["isolated_occupation_count"] == 0
            and graph["summary"]["isolated_skill_count"] == 0,
            "occupation_count": graph["summary"]["occupation_count"],
            "skill_count": graph["summary"]["skill_count"],
            "edge_count": graph["summary"]["edge_count"],
            "display_edge_count": graph["filtered_edge_count"],
            "isolated_occupation_count": graph["summary"]["isolated_occupation_count"],
            "isolated_skill_count": graph["summary"]["isolated_skill_count"],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def verify_stage1(root: Path, db_path: Path) -> dict[str, Any]:
    """组合数据、数据库、特征、加载器、图谱、Web、模型和简历验收结果。"""
    root = Path(root)
    clean_dir = root / "data" / "clean"
    processed_dir = root / "data" / "processed"
    # 各子检查隔离异常并保留结构化错误；单项失败不会阻止其余证据生成。
    try:
        data_integrity = validate_clean_dataset(clean_dir)
    except Exception as exc:
        data_integrity = {"ok": False, "error": str(exc)}
    try:
        feature_outputs = validate_feature_outputs(clean_dir, processed_dir)
    except Exception as exc:
        feature_outputs = {"ok": False, "error": str(exc)}
    return {
        "data_integrity": data_integrity,
        "database": _database_report(Path(db_path)),
        "feature_outputs": feature_outputs,
        "loader": _loader_report(root),
        "bipartite_graph": _bipartite_graph_report(root),
        "web_contract": _web_contract_report(root),
        "model": _model_report(root),
        "web_api": _web_api_report(root),
        "resume_dataset": _resume_dataset_report(root),
    }


def main() -> int:
    """解析数据库路径，打印验收 JSON，并以 0/1 表示关键项是否全部通过。"""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify Topic 17 first-week acceptance criteria")
    parser.add_argument("--db-path", type=Path, default=root / "artifacts" / "topic17.sqlite3")
    args = parser.parse_args()
    report = verify_stage1(root, args.db_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 模型和 DataLoader 可因可选依赖而 unavailable；关键退出码只聚合
    # 数据、数据库、特征、图谱、Web 契约/API 和简历数据集这些硬性项。
    critical = [report["data_integrity"], report["database"], report["feature_outputs"], report["bipartite_graph"], report["web_contract"], report["web_api"], report["resume_dataset"]]
    return 0 if all(section.get("ok", False) for section in critical) else 1


if __name__ == "__main__":
    raise SystemExit(main())
