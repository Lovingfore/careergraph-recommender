"""Web views backed by the shared database and DataLoader services."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from django.http import JsonResponse
from django.shortcuts import render

from src.database import get_counts, query_user_profile
from src.data_loader import build_dataloader


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


def index(request):
    return render(request, "topic17_app/index.html", {"summary": _summary()})
