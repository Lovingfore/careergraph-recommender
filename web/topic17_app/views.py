"""Web views backed by the shared database and DataLoader services."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from src.database import get_counts, query_user_profile
from src.data_loader import build_dataloader
from src.recommendation_service import (
    career_path_for_user,
    forecast_for_user,
    recommend_for_user,
    skill_gap_for_user,
)


BASE_DIR = Path(__file__).resolve().parents[2]


def _db_path() -> Path:
    return Path(os.environ.get("TOPIC17_DB_PATH", str(BASE_DIR / "artifacts" / "topic17.sqlite3")))


def _summary() -> dict[str, Any]:
    db_path = _db_path()
    try:
        counts = get_counts(db_path)
        database_status: dict[str, Any] = {"ok": True}
    except Exception as exc:
        counts = {}
        database_status = {"ok": False, "error": str(exc)}

    try:
        loader, metadata = build_dataloader(BASE_DIR / "data" / "clean", batch_size=2, sequence_length=3)
        first_batch = next(iter(loader), None)
        loader_status: dict[str, Any] = {
            "status": "ok",
            "num_samples": metadata["num_samples"],
            "num_skills": metadata["num_skills"],
            "sequence_length": metadata["sequence_length"],
            "first_batch_shape": list(first_batch["x"].shape) if first_batch is not None else [],
        }
    except Exception as exc:
        loader_status = {"status": "torch_not_installed_or_unavailable", "error": str(exc)}

    profile: dict[str, Any] | None = None
    if database_status["ok"]:
        try:
            profile = query_user_profile(db_path, "u001")
            # Events are useful in the API but unnecessarily large for the page.
            profile = {key: value for key, value in profile.items() if key != "events"}
        except Exception:
            profile = None
    return {
        "counts": counts,
        "database": database_status,
        "loader": loader_status,
        "sample_profile": profile,
        "db_path": str(db_path),
    }


def summary_api(request):
    return JsonResponse(_summary())


def _required_query(request, *names: str) -> dict[str, str] | JsonResponse:
    values = {name: str(request.GET.get(name, "")).strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        return JsonResponse({"status": "error", "error": f"missing query parameter(s): {', '.join(missing)}"}, status=400)
    return values


def _service_response(result: dict) -> JsonResponse:
    return JsonResponse(result, status=200 if result.get("status") in {"ok", "baseline"} else 400)


def recommendation_api(request):
    values = _required_query(request, "user_id")
    if isinstance(values, JsonResponse):
        return values
    try:
        top_k = int(request.GET.get("top_k", "5"))
    except ValueError:
        top_k = 0
    return _service_response(recommend_for_user(values["user_id"], top_k=top_k, root=BASE_DIR))


def skill_gap_api(request):
    values = _required_query(request, "user_id", "occupation_id")
    if isinstance(values, JsonResponse):
        return values
    return _service_response(skill_gap_for_user(values["user_id"], values["occupation_id"], root=BASE_DIR))


def career_path_api(request):
    values = _required_query(request, "user_id", "occupation_id")
    if isinstance(values, JsonResponse):
        return values
    return _service_response(career_path_for_user(values["user_id"], values["occupation_id"], root=BASE_DIR))


def forecast_api(request):
    values = _required_query(request, "user_id")
    if isinstance(values, JsonResponse):
        return values
    try:
        months = int(request.GET.get("months", "6"))
    except ValueError:
        months = 0
    return _service_response(forecast_for_user(values["user_id"], months=months, root=BASE_DIR))


def bipartite_graph(request):
    graph_path = BASE_DIR / "data" / "processed" / "bipartite" / "bipartite_graph.svg"
    if not graph_path.exists():
        return HttpResponse("bipartite graph is not generated", status=404, content_type="text/plain; charset=utf-8")
    return HttpResponse(graph_path.read_text(encoding="utf-8"), content_type="image/svg+xml")


def index(request):
    return render(request, "topic17_app/index.html", {"summary": _summary()})
