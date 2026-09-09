"""Acceptance checks for the Topic 17 first-week deliverable."""

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
    try:
        counts = get_counts(db_path)
        missing = [table for table in EXPECTED_TABLES if counts.get(table, 0) <= 0]
        return {"ok": not missing, "counts": counts, "missing_or_empty_tables": missing}
    except Exception as exc:
        return {"ok": False, "counts": {}, "error": str(exc)}


def _loader_report(root: Path) -> dict[str, Any]:
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
    routes_ok = 'path("", index' in route_text and 'path("api/summary/", summary_api' in route_text
    return {"ok": not missing and routes_ok, "required_files": [str(path.relative_to(root)) for path in required], "missing_files": missing, "routes_ok": routes_ok}


def _bipartite_graph_report(root: Path) -> dict[str, Any]:
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
    root = Path(root)
    clean_dir = root / "data" / "clean"
    processed_dir = root / "data" / "processed"
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
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify Topic 17 first-week acceptance criteria")
    parser.add_argument("--db-path", type=Path, default=root / "artifacts" / "topic17.sqlite3")
    args = parser.parse_args()
    report = verify_stage1(root, args.db_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    critical = [report["data_integrity"], report["database"], report["feature_outputs"], report["bipartite_graph"], report["web_contract"]]
    return 0 if all(section.get("ok", False) for section in critical) else 1


if __name__ == "__main__":
    raise SystemExit(main())
