"""把职位转移概率渲染为独立 SVG 的只读模块。

数据来自 ``data/processed/transition_graph.json`` 和
``data/clean/occupations.csv``；本模块只读取文件并生成字符串，不写入
SQLite 或修改任何数据。
"""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from src.generate_chinese_resumes import OCCUPATION_NAMES_ZH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_graph(root: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """读取职位中英文名称和转移边，统一成渲染所需的节点/边结构。"""
    occupations_path = root / "data" / "clean" / "occupations.csv"
    transitions_path = root / "data" / "processed" / "transition_graph.json"
    with occupations_path.open("r", encoding="utf-8-sig", newline="") as handle:
        occupations = {
            str(row["occupation_id"]): {
                "name": str(row["occupation_name"]),
                "name_zh": OCCUPATION_NAMES_ZH.get(str(row["occupation_name"]), str(row["occupation_name"])),
            }
            for row in csv.DictReader(handle)
        }
    raw_graph = json.loads(transitions_path.read_text(encoding="utf-8"))
    edges = [
        {
            "source": str(source),
            "target": str(edge["to_job"]),
            "probability": float(edge["probability"]),
        }
        for source, targets in raw_graph.items()
        for edge in targets
    ]
    return occupations, edges


def _topological_levels(node_ids: list[str], edges: list[dict[str, Any]]) -> dict[str, int]:
    """用拓扑层级把起始、中间、目标职位从左到右排列。

    正常教学图是 DAG；若未来数据出现环，未访问节点安全回退到第 0 层，
    保证页面仍能渲染而不会因拓扑排序失败中断。
    """
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].append(target)
        indegree[target] = indegree.get(target, 0) + 1
        indegree.setdefault(source, 0)

    levels = {node_id: 0 for node_id in node_ids}
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    visited: set[str] = set()
    while queue:
        source = queue.popleft()
        visited.add(source)
        for target in adjacency.get(source, []):
            levels[target] = max(levels.get(target, 0), levels[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    # 当前教学图无环；保留安全回退，让未来含环的数据仍能显示而不是让
    # 仪表盘因拓扑队列耗尽或层级缺失而失败。
    for node_id in node_ids:
        if node_id not in visited:
            levels[node_id] = 0
    return levels


def _positions(levels: dict[str, int]) -> dict[str, tuple[float, float]]:
    """按层级和节点数量计算画布坐标，避免同层节点重叠。"""
    canvas_width = 1160.0
    node_width = 230.0
    node_height = 72.0
    left = 28.0
    top = 105.0
    available_height = 570.0
    max_level = max(levels.values(), default=0)
    horizontal_step = (canvas_width - 2 * left - node_width) / max(max_level, 1)
    grouped: dict[int, list[str]] = defaultdict(list)
    for node_id, level in levels.items():
        grouped[level].append(node_id)

    positions: dict[str, tuple[float, float]] = {}
    for level, node_ids in grouped.items():
        node_ids.sort()
        if len(node_ids) == 1:
            ys = [top + (available_height - node_height) / 2]
        else:
            gap = max(12.0, (available_height - len(node_ids) * node_height) / (len(node_ids) - 1))
            ys = [top + index * (node_height + gap) for index in range(len(node_ids))]
        x = left + level * horizontal_step
        for node_id, y in zip(node_ids, ys):
            positions[node_id] = (x, y)
    return positions


def _bezier_point(
    start: tuple[float, float],
    control_a: tuple[float, float],
    control_b: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """计算三次贝塞尔曲线在参数 ``t`` 处的点，用于放置概率标签。"""
    u = 1.0 - t
    return (
        u**3 * start[0] + 3 * u * u * t * control_a[0] + 3 * u * t * t * control_b[0] + t**3 * end[0],
        u**3 * start[1] + 3 * u * u * t * control_a[1] + 3 * u * t * t * control_b[1] + t**3 * end[1],
    )


def render_transition_graph_svg(root: Path | None = None) -> str:
    """返回当前职位转移数据的自包含 SVG 字符串。

    每条边用带箭头的贝塞尔曲线表示方向，线宽与概率相关，曲线上标注
    百分比；节点颜色按无入边的起始、同时有入/出边的中间、无出边的目标
    分类。函数只读 JSON/CSV，调用方负责把字符串作为 Web 资源返回。
    """

    root_path = Path(root) if root is not None else PROJECT_ROOT
    occupations, edges = _load_graph(root_path)
    levels = _topological_levels(sorted(occupations), edges)
    positions = _positions(levels)
    outgoing = {edge["source"] for edge in edges}
    incoming = {edge["target"] for edge in edges}
    node_width = 230.0
    node_height = 72.0

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="760" viewBox="0 0 1160 760" role="img" aria-labelledby="graph-title graph-desc">',
        '<title id="graph-title">Occupation transition probability graph</title>',
        '<desc id="graph-desc">Directed edges show transition probabilities between occupations in the teaching dataset.</desc>',
        '<defs>',
        '<marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,4 L0,8 Z" fill="#4f46e5"/></marker>',
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.12"/></filter>',
        '<style>.node-title{font:700 16px "Microsoft YaHei","Segoe UI",sans-serif;fill:#172033}.node-code{font:12px "Segoe UI",sans-serif;fill:#64748b}.edge-label{font:700 12px "Segoe UI",sans-serif;fill:#4338ca}.heading{font:700 23px "Microsoft YaHei","Segoe UI",sans-serif;fill:#172033}.subtitle{font:13px "Microsoft YaHei","Segoe UI",sans-serif;fill:#64748b}.legend{font:12px "Microsoft YaHei","Segoe UI",sans-serif;fill:#475569}</style>',
        '</defs>',
        '<rect width="1160" height="760" rx="22" fill="#f8fafc"/>',
        '<text x="28" y="38" class="heading">职位转移概率图</text>',
        '<text x="28" y="62" class="subtitle">箭头表示转移方向，百分比表示教学数据中的转移概率</text>',
        '<g transform="translate(750 34)"><circle cx="0" cy="0" r="6" fill="#dbeafe" stroke="#2563eb"/><text x="12" y="4" class="legend">起始职位</text><circle cx="100" cy="0" r="6" fill="#ede9fe" stroke="#7c3aed"/><text x="112" y="4" class="legend">中间职位</text><circle cx="212" cy="0" r="6" fill="#dcfce7" stroke="#16a34a"/><text x="224" y="4" class="legend">目标职位</text></g>',
    ]

    # 用曲线控制点错开平行边，标签沿曲线轮换位置，减少文字重叠。
    for index, edge in enumerate(edges):
        source_x, source_y = positions[edge["source"]]
        target_x, target_y = positions[edge["target"]]
        x1 = source_x + node_width
        y1 = source_y + node_height / 2
        x2 = target_x - 8
        y2 = target_y + node_height / 2
        bend = max(44.0, (x2 - x1) * 0.42)
        control_a = (x1 + bend, y1)
        control_b = (x2 - bend, y2)
        probability = float(edge["probability"])
        stroke_width = 1.7 + probability * 3.0
        label_t = (0.34, 0.5, 0.66)[index % 3]
        label_x, label_y = _bezier_point((x1, y1), control_a, control_b, (x2, y2), label_t)
        label_y += (-15.0, 0.0, 15.0)[index % 3]
        label = f"{probability * 100:.1f}%"
        parts.extend(
            [
                f'<path d="M{x1:.1f},{y1:.1f} C{control_a[0]:.1f},{control_a[1]:.1f} {control_b[0]:.1f},{control_b[1]:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="#4f46e5" stroke-width="{stroke_width:.2f}" stroke-opacity="0.58" marker-end="url(#arrowhead)"/>',
                f'<rect x="{label_x - 25:.1f}" y="{label_y - 13:.1f}" width="50" height="22" rx="11" fill="#ffffff" stroke="#c7d2fe"/>',
                f'<text x="{label_x:.1f}" y="{label_y + 3:.1f}" text-anchor="middle" class="edge-label">{label}</text>',
            ]
        )

    for node_id in sorted(occupations):
        x, y = positions[node_id]
        name = occupations[node_id]["name"]
        name_zh = occupations[node_id]["name_zh"]
        if node_id not in outgoing and node_id not in incoming:
            fill, stroke = "#f1f5f9", "#94a3b8"
        elif node_id not in incoming:
            fill, stroke = "#dbeafe", "#2563eb"
        elif node_id not in outgoing:
            fill, stroke = "#dcfce7", "#16a34a"
        else:
            fill, stroke = "#ede9fe", "#7c3aed"
        safe_name = html.escape(name)
        safe_name_zh = html.escape(name_zh)
        safe_id = html.escape(node_id)
        parts.extend(
            [
                f'<g filter="url(#shadow)"><title>{safe_name}</title>',
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_width:.1f}" height="{node_height:.1f}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
                f'<text x="{x + 14:.1f}" y="{y + 30:.1f}" class="node-title">{safe_name_zh}</text>',
                f'<text x="{x + 14:.1f}" y="{y + 53:.1f}" class="node-code">{safe_id}</text>',
                '</g>',
            ]
        )

    parts.append('</svg>')
    return "".join(parts)


__all__ = ["render_transition_graph_svg"]
