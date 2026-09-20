"""Django 视图与业务服务之间的接口适配层。

页面请求在这里被转换为 SQLite、DataLoader、推荐服务或简历分析函数的输入，
随后把结果封装成 HTML、JSON 或 SVG 响应。视图本身不重复实现推荐算法。
"""

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
from src.transition_graph import render_transition_graph_svg
from src.recommendation_service import (
    career_path_for_user,
    forecast_for_user,
    recommend_for_user,
    skill_gap_for_user,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MAX_RESUME_BYTES = 1024 * 1024


def _db_path() -> Path:
    """返回 Web 查询使用的 SQLite 路径，并允许测试通过环境变量覆盖。"""

    return Path(os.environ.get("TOPIC17_DB_PATH", str(BASE_DIR / "artifacts" / "topic17.sqlite3")))


def _summary() -> dict[str, Any]:
    """聚合数据库、DataLoader、模型状态和示例用户，形成首页与摘要 API 数据。"""

    db_path = _db_path()
    # 数据库异常被折叠为状态字段，使首页仍能展示其他独立模块的信息。
    try:
        counts = get_counts(db_path)
        database_status: dict[str, Any] = {"ok": True}
    except Exception as exc:
        counts = {}
        database_status = {"ok": False, "error": str(exc)}

    # DataLoader 是可选 PyTorch 链路；不可用时返回可诊断状态而不是中断页面。
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

    # 数据库可用时读取一个示例画像；首页摘要不携带体积较大的完整事件列表。
    profile: dict[str, Any] | None = None
    if database_status["ok"]:
        try:
            profile = query_user_profile(db_path, "u001")
            # 技能事件适合专门接口查询，但对首页摘要而言体积过大。
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
    """直接读取模型评估 JSON，无需导入可选的 PyTorch 运行环境。"""

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
    """返回首页使用的综合系统摘要；该接口不修改任何数据。"""

    return JsonResponse(_summary())


def model_info_api(request):
    """以 GET 方式返回训练轮数、切分、baseline 与 TemporalGAT 评估元数据。"""

    if request.method != "GET":
        return JsonResponse({"status": "error", "error": "仅支持 GET 请求"}, status=405)
    return JsonResponse(_model_info())


def occupations_api(request):
    """读取职位 CSV，补充中文名称后返回前端三个职位下拉框的数据源。"""

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
        return JsonResponse({"status": "error", "error": "职位字典暂不可用"}, status=500)
    return JsonResponse({"status": "ok", "occupations": rows})


def _required_query(request, *names: str) -> dict[str, str] | JsonResponse:
    """统一读取必填 GET 参数；缺失时直接生成 HTTP 400 JSON 响应。"""

    values = {name: str(request.GET.get(name, "")).strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        return JsonResponse({"status": "error", "error": f"missing query parameter(s): {', '.join(missing)}"}, status=400)
    return values


def _service_response(result: dict) -> JsonResponse:
    """把业务服务状态统一映射成成功 200 或参数/业务错误 400。"""

    return JsonResponse(result, status=200 if result.get("status") in {"ok", "baseline"} else 400)


def recommendation_api(request):
    """读取 user_id/top_k，调用离线推荐服务并返回排序后的候选职位。"""

    # 输入校验：user_id 必填，top_k 必须能转换为正整数才能由服务接受。
    values = _required_query(request, "user_id")
    if isinstance(values, JsonResponse):
        return values
    try:
        top_k = int(request.GET.get("top_k", "5"))
    except ValueError:
        top_k = 0
    # 业务计算与输出：服务读取离线排序文件，视图只负责转换为 HTTP 响应。
    return _service_response(recommend_for_user(values["user_id"], top_k=top_k, root=BASE_DIR))


def skill_gap_api(request):
    """读取用户和目标职位，调用服务比较最新技能水平与职位需求。"""

    # 输入校验 → 服务读取 clean CSV 并计算技能缺口 → JSON 状态映射。
    values = _required_query(request, "user_id", "occupation_id")
    if isinstance(values, JsonResponse):
        return values
    return _service_response(skill_gap_for_user(values["user_id"], values["occupation_id"], root=BASE_DIR))


def career_path_api(request):
    """读取用户和目标职位，返回最多三跳的职业转移路径及概率乘积。"""

    # 输入校验 → 服务在转移图上搜索路径 → JSON 状态映射。
    values = _required_query(request, "user_id", "occupation_id")
    if isinstance(values, JsonResponse):
        return values
    return _service_response(career_path_for_user(values["user_id"], values["occupation_id"], root=BASE_DIR))


def forecast_api(request):
    """读取 user_id/months，基于离线保存的末值与斜率外推线性趋势预测。"""

    # 输入校验：user_id 必填，months 转换失败时交给服务按无效值处理。
    values = _required_query(request, "user_id")
    if isinstance(values, JsonResponse):
        return values
    try:
        months = int(request.GET.get("months", "6"))
    except ValueError:
        months = 0
    # 服务读取 processed/skill_forecast_6m.csv 的 last_level 与 monthly_slope，
    # 按请求月份重新外推 baseline，再经统一状态规则转换为 HTTP 响应。
    return _service_response(forecast_for_user(values["user_id"], months=months, root=BASE_DIR))


def bipartite_graph(request):
    """读取离线生成的职位—技能二部图 SVG；文件缺失时返回 404。"""

    graph_path = BASE_DIR / "data" / "processed" / "bipartite" / "bipartite_graph.svg"
    if not graph_path.exists():
        return HttpResponse("bipartite graph is not generated", status=404, content_type="text/plain; charset=utf-8")
    return HttpResponse(graph_path.read_text(encoding="utf-8"), content_type="image/svg+xml")


def transition_graph(request):
    """根据当前转移 JSON 和职位字典动态生成只读 SVG；数据缺失返回 404。"""

    try:
        svg = render_transition_graph_svg(BASE_DIR)
    except FileNotFoundError:
        return HttpResponse("transition graph data is not generated", status=404, content_type="text/plain; charset=utf-8")
    return HttpResponse(svg, content_type="image/svg+xml")


def resume_upload_api(request):
    """校验并在内存中分析简历，返回推荐、技能缺口和职业路径。

    仅接受 POST 以及 UTF-8 编码的 ``.txt``、``.md``、``.csv`` 文件，大小上限
    为 1 MB。上传内容不会写入 SQLite、CSV 或训练数据，业务结果保持
    ``persisted=false``；参数错误返回 400，方法错误返回 405，依赖数据缺失返回 503。
    """

    # 1. 请求方法：上传接口只接受 multipart POST。
    if request.method != "POST":
        return JsonResponse({"status": "error", "error": "仅支持 POST 请求"}, status=405)
    # 2. 文件存在性：表单字段名必须为 resume。
    upload = request.FILES.get("resume")
    if upload is None:
        return JsonResponse({"status": "error", "error": "请上传简历文件"}, status=400)
    # 3. 扩展名：仅允许纯文本类文件，避免解析复杂二进制格式。
    suffix = Path(upload.name or "").suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        return JsonResponse({"status": "error", "error": "仅支持 .txt、.md 或 .csv 简历文件"}, status=400)
    # 4. 声明大小：先根据上传对象元数据拒绝超过 1 MB 的文件。
    if getattr(upload, "size", 0) > MAX_RESUME_BYTES:
        return JsonResponse({"status": "error", "error": "简历文件不能超过 1 MB"}, status=400)
    # 5. 实际读取：最多多读一个字节，用于防止不可信 size 元数据绕过限制。
    try:
        raw = upload.read(MAX_RESUME_BYTES + 1)
    except Exception as exc:
        return JsonResponse({"status": "error", "error": f"读取简历失败：{exc}"}, status=400)
    if len(raw) > MAX_RESUME_BYTES:
        return JsonResponse({"status": "error", "error": "简历文件不能超过 1 MB"}, status=400)
    # 6. 文本解码：utf-8-sig 同时兼容普通 UTF-8 与带 BOM 的文件。
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return JsonResponse({"status": "error", "error": "简历必须使用 UTF-8 编码"}, status=400)
    # 7. 业务分析：文本与可选职位传给纯内存分析服务，不发生持久化写入。
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
    # 8. JSON 输出：附带原始文件名，保留服务返回的 persisted=false 与分析结果。
    result["filename"] = upload.name
    return JsonResponse(result)


def index(request):
    """渲染首页，并把综合摘要作为首屏模板上下文注入页面。"""

    return render(request, "topic17_app/index.html", {"summary": _summary()})
