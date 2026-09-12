"""Web views backed by the shared database and DataLoader services."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from src.database import get_counts, query_user_profile
from src.data_loader import build_dataloader
from src.generate_chinese_resumes import OCCUPATION_NAMES_ZH
from src.resume_analysis import SUPPORTED_EXTENSIONS, analyze_resume_text
from src.recommendation_service import (
    career_path_for_user,
    forecast_for_user,
    recommend_for_user,
    skill_gap_for_user,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MAX_RESUME_BYTES = 1024 * 1024


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
        "model": _model_info(),
        "sample_profile": profile,
        "db_path": str(db_path),
    }


def _model_info() -> dict[str, Any]:
    """Read model metadata without requiring the optional PyTorch package."""

    evaluation_path = BASE_DIR / "artifacts" / "models" / "evaluation_summary.json"
    checkpoint_path = BASE_DIR / "artifacts" / "models" / "temporal_gat.pt"
    if not evaluation_path.exists():
        return {
            "status": "unavailable",
            "error": "未找到模型评估文件，请先运行训练命令。",
            "checkpoint_exists": checkpoint_path.exists(),
        }
    try:
        summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "error": f"模型评估文件不可读：{exc}", "checkpoint_exists": checkpoint_path.exists()}
    return {
        "status": "trained" if summary.get("status") == "trained" and checkpoint_path.exists() else "unavailable",
        "epochs": summary.get("epochs"),
        "sample_count": summary.get("sample_count"),
        "split": summary.get("split", {}),
        "baseline": summary.get("baseline", {}),
        "temporal_gat": summary.get("temporal_gat", {}),
        "data_note": summary.get("data_note", ""),
        "checkpoint_exists": checkpoint_path.exists(),
        "evaluation_file": str(evaluation_path.relative_to(BASE_DIR)),
    }


def summary_api(request):
    return JsonResponse(_summary())


def model_info_api(request):
    if request.method != "GET":
        return JsonResponse({"status": "error", "error": "仅支持 GET 请求"}, status=405)
    return JsonResponse(_model_info())


def occupations_api(request):
    if request.method != "GET":
        return JsonResponse({"status": "error", "error": "仅支持 GET 请求"}, status=405)
    path = BASE_DIR / "data" / "clean" / "occupations.csv"
    try:
        import pandas as pd

        occupations = pd.read_csv(path, dtype=str).sort_values("occupation_id")
        rows = [
            {
                "occupation_id": str(row.occupation_id),
                "occupation_name": str(row.occupation_name),
                "occupation_name_zh": OCCUPATION_NAMES_ZH.get(str(row.occupation_name), str(row.occupation_name)),
            }
            for row in occupations.itertuples(index=False)
        ]
    except Exception as exc:
        return JsonResponse({"status": "error", "error": f"职位字典不可用：{exc}"}, status=500)
    return JsonResponse({"status": "ok", "occupations": rows})


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


def resume_upload_api(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "error": "仅支持 POST 请求"}, status=405)
    upload = request.FILES.get("resume")
    if upload is None:
        return JsonResponse({"status": "error", "error": "请上传简历文件"}, status=400)
    suffix = Path(upload.name or "").suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        return JsonResponse({"status": "error", "error": "仅支持 .txt、.md 或 .csv 简历文件"}, status=400)
    if getattr(upload, "size", 0) > MAX_RESUME_BYTES:
        return JsonResponse({"status": "error", "error": "简历文件不能超过 1 MB"}, status=400)
    try:
        raw = upload.read(MAX_RESUME_BYTES + 1)
    except Exception as exc:
        return JsonResponse({"status": "error", "error": f"读取简历失败：{exc}"}, status=400)
    if len(raw) > MAX_RESUME_BYTES:
        return JsonResponse({"status": "error", "error": "简历文件不能超过 1 MB"}, status=400)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return JsonResponse({"status": "error", "error": "简历必须使用 UTF-8 编码"}, status=400)
    try:
        result = analyze_resume_text(
            text,
            root=BASE_DIR,
            current_job=request.POST.get("current_job") or None,
            target_job=request.POST.get("target_job") or None,
        )
    except ValueError as exc:
        return JsonResponse({"status": "error", "error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"status": "error", "error": "分析所需的数据文件暂不可用"}, status=503)
    result["filename"] = upload.name
    return JsonResponse(result)


def index(request):
    return render(request, "topic17_app/index.html", {"summary": _summary()})
