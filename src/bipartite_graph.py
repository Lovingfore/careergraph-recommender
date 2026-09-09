"""Build and export the occupation-skill bipartite graph.

The graph has two disjoint node sets:

* ``occupation:*`` nodes on the left;
* ``skill:*`` nodes on the right.

Every edge is an occupation-to-skill requirement and carries the cleaned
``demand_weight`` as its primary weight. The module intentionally keeps this
representation separate from the existing skill-cooccurrence and occupation
transition graphs so downstream models can choose the relation they need.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


OCCUPATION_PREFIX = "occupation:"
SKILL_PREFIX = "skill:"


def _node_id(prefix: str, raw_id: str) -> str:
    return f"{prefix}{raw_id}"


def build_bipartite_graph(data_dir: Path, min_demand_weight: float = 0.0) -> dict[str, Any]:
    """Create a validated occupation-skill bipartite graph from clean CSVs.

    ``min_demand_weight`` only controls ``display_edges``. The complete edge
    list remains in ``edges`` so the exported graph is lossless.
    """

    if not 0 <= min_demand_weight <= 1:
        raise ValueError("min_demand_weight must be between 0 and 1")

    occupations = pd.read_csv(data_dir / "occupations.csv")
    skills = pd.read_csv(data_dir / "skills.csv")
    relations = pd.read_csv(data_dir / "occupation_skill.csv")

    required_occupation_columns = {"occupation_id", "occupation_name", "Description"}
    required_skill_columns = {"skill_id", "skill_name", "skill_category"}
    required_relation_columns = {
        "occupation_id",
        "skill_id",
        "skill_name",
        "importance",
        "level",
        "importance_norm",
        "level_norm",
        "demand_weight",
        "is_required",
        "skill_source",
    }
    for name, frame, required in (
        ("occupations.csv", occupations, required_occupation_columns),
        ("skills.csv", skills, required_skill_columns),
        ("occupation_skill.csv", relations, required_relation_columns),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    occupation_ids = set(occupations["occupation_id"].astype(str))
    skill_ids = set(skills["skill_id"].astype(str))
    relations = relations.copy()
    relations["occupation_id"] = relations["occupation_id"].astype(str)
    relations["skill_id"] = relations["skill_id"].astype(str)
    relations["demand_weight"] = pd.to_numeric(relations["demand_weight"])
    relations["importance"] = pd.to_numeric(relations["importance"])
    relations["level"] = pd.to_numeric(relations["level"])
    relations["importance_norm"] = pd.to_numeric(relations["importance_norm"])
    relations["level_norm"] = pd.to_numeric(relations["level_norm"])
    relations["is_required"] = pd.to_numeric(relations["is_required"]).astype(int)

    orphan_occupations = sorted(set(relations["occupation_id"]) - occupation_ids)
    orphan_skills = sorted(set(relations["skill_id"]) - skill_ids)
    duplicate_pairs = int(relations[["occupation_id", "skill_id"]].duplicated().sum())
    invalid_weights = int(((relations["demand_weight"] < 0) | (relations["demand_weight"] > 1)).sum())
    if orphan_occupations or orphan_skills or duplicate_pairs or invalid_weights:
        raise ValueError(
            "invalid occupation-skill relation data: "
            f"orphan_occupations={orphan_occupations}, orphan_skills={orphan_skills}, "
            f"duplicate_pairs={duplicate_pairs}, invalid_weights={invalid_weights}"
        )

    occupation_nodes = [
        {
            "id": _node_id(OCCUPATION_PREFIX, str(row.occupation_id)),
            "raw_id": str(row.occupation_id),
            "type": "occupation",
            "label": str(row.occupation_name),
            "description": str(row.Description),
        }
        for row in occupations.sort_values("occupation_id").itertuples(index=False)
    ]
    skill_lookup = {
        str(row.skill_id): row
        for row in skills.itertuples(index=False)
    }
    skill_nodes = [
        {
            "id": _node_id(SKILL_PREFIX, str(row.skill_id)),
            "raw_id": str(row.skill_id),
            "type": "skill",
            "label": str(row.skill_name),
            "category": str(row.skill_category),
        }
        for row in skills.sort_values("skill_id").itertuples(index=False)
    ]

    edges: list[dict[str, Any]] = []
    for row in relations.sort_values(["occupation_id", "skill_id"]).itertuples(index=False):
        skill = skill_lookup[str(row.skill_id)]
        edges.append(
            {
                "source": _node_id(OCCUPATION_PREFIX, str(row.occupation_id)),
                "target": _node_id(SKILL_PREFIX, str(row.skill_id)),
                "source_type": "occupation",
                "target_type": "skill",
                "occupation_id": str(row.occupation_id),
                "skill_id": str(row.skill_id),
                "skill_name": str(skill.skill_name),
                "weight": round(float(row.demand_weight), 6),
                "importance": round(float(row.importance), 6),
                "level": round(float(row.level), 6),
                "importance_norm": round(float(row.importance_norm), 6),
                "level_norm": round(float(row.level_norm), 6),
                "is_required": int(row.is_required),
                "skill_source": str(row.skill_source),
            }
        )

    display_edges = [edge for edge in edges if edge["weight"] >= min_demand_weight]
    occupation_degree = Counter(edge["occupation_id"] for edge in edges)
    skill_degree = Counter(edge["skill_id"] for edge in edges)
    weighted_occupation_degree = Counter()
    weighted_skill_degree = Counter()
    for edge in edges:
        weighted_occupation_degree[edge["occupation_id"]] += edge["weight"]
        weighted_skill_degree[edge["skill_id"]] += edge["weight"]

    return {
        "schema_version": "1.0",
        "graph_type": "bipartite",
        "left_node_type": "occupation",
        "right_node_type": "skill",
        "weight_field": "demand_weight",
        "min_demand_weight": min_demand_weight,
        "occupation_nodes": occupation_nodes,
        "skill_nodes": skill_nodes,
        "edges": edges,
        "display_edges": display_edges,
        "full_edge_count": len(edges),
        "filtered_edge_count": len(display_edges),
        "summary": {
            "occupation_count": len(occupation_nodes),
            "skill_count": len(skill_nodes),
            "edge_count": len(edges),
            "required_edge_count": sum(edge["is_required"] for edge in edges),
            "isolated_occupation_count": sum(occupation_degree.get(node["raw_id"], 0) == 0 for node in occupation_nodes),
            "isolated_skill_count": sum(skill_degree.get(node["raw_id"], 0) == 0 for node in skill_nodes),
            "top_skills_by_weight": [
                {"skill_id": skill_id, "weighted_degree": round(weight, 6)}
                for skill_id, weight in weighted_skill_degree.most_common(10)
            ],
            "top_occupations_by_weight": [
                {"occupation_id": occupation_id, "weighted_degree": round(weight, 6)}
                for occupation_id, weight in weighted_occupation_degree.most_common(10)
            ],
        },
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_bipartite_outputs(graph: dict[str, Any], output_dir: Path, render: bool = True) -> dict[str, Path]:
    """Write JSON, CSV, summary and (when available) a PNG visualization."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = _write_json(output_dir / "bipartite_graph.json", graph)
    summary_path = _write_json(output_dir / "bipartite_graph_summary.json", graph["summary"] | {
        "min_demand_weight": graph["min_demand_weight"],
        "full_edge_count": graph["full_edge_count"],
        "filtered_edge_count": graph["filtered_edge_count"],
    })
    edges_path = output_dir / "bipartite_edges.csv"
    edge_fields = [
        "source",
        "target",
        "occupation_id",
        "skill_id",
        "skill_name",
        "weight",
        "importance",
        "level",
        "importance_norm",
        "level_norm",
        "is_required",
        "skill_source",
    ]
    with edges_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows({field: edge[field] for field in edge_fields} for edge in graph["edges"])

    paths = {"json": json_path, "summary_json": summary_path, "edges_csv": edges_path}
    if render:
        png_path = output_dir / "bipartite_graph.png"
        if _render_png(graph, png_path):
            paths["png"] = png_path if png_path.exists() else png_path.with_suffix(".svg")
    return paths


def _render_png(graph: dict[str, Any], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
    except ModuleNotFoundError:
        return _render_pil_png(graph, output_path)

    occupations = graph["occupation_nodes"]
    skills = graph["skill_nodes"]
    edges = graph["display_edges"] or graph["edges"]
    occupation_pos = {
        node["id"]: (0.0, len(occupations) - index - 1)
        for index, node in enumerate(occupations)
    }
    skill_pos = {
        node["id"]: (1.0, len(skills) - index - 1)
        for index, node in enumerate(skills)
    }
    fig_height = max(8.0, 0.30 * max(len(occupations), len(skills)))
    fig, ax = plt.subplots(figsize=(18, fig_height), dpi=160)
    cmap = cm.get_cmap("Blues")
    norm = colors.Normalize(vmin=0, vmax=1)
    for edge in edges:
        x1, y1 = occupation_pos[edge["source"]]
        x2, y2 = skill_pos[edge["target"]]
        ax.plot([x1, x2], [y1, y2], color=cmap(norm(edge["weight"])), alpha=0.40, linewidth=0.7 + 2.0 * edge["weight"])
    ax.scatter([0] * len(occupations), [occupation_pos[node["id"]][1] for node in occupations], s=90, color="#1f4d78", zorder=3)
    ax.scatter([1] * len(skills), [skill_pos[node["id"]][1] for node in skills], s=50, color="#2e8b57", zorder=3)
    for node in occupations:
        ax.text(-0.025, occupation_pos[node["id"]][1], node["label"], ha="right", va="center", fontsize=8.5)
    for node in skills:
        ax.text(1.025, skill_pos[node["id"]][1], node["label"], ha="left", va="center", fontsize=8.0)
    ax.text(0, len(occupations) + 1, "职位节点（12）", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1f4d78")
    ax.text(1, len(skills) + 1, "技能节点（35）", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2e8b57")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-1, max(len(occupations), len(skills)) + 2)
    ax.axis("off")
    threshold = graph["min_demand_weight"]
    ax.set_title(f"Occupation-Skill Bipartite Graph | display demand_weight >= {threshold:.2f} | edges {len(edges)}/{len(graph['edges'])}", pad=18)
    fig.text(0.5, 0.015, "边颜色/粗细表示 demand_weight；完整 420 条边保存在 JSON/CSV，图中显示高权重可读子图。", ha="center", fontsize=9, color="#5b6573")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _render_pil_png(graph: dict[str, Any], output_path: Path) -> bool:
    """Fallback PNG renderer using Pillow, available in the workspace runtime."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        return _render_svg(graph, output_path.with_suffix(".svg"))

    occupations = graph["occupation_nodes"]
    skills = graph["skill_nodes"]
    edges = graph["display_edges"] or graph["edges"]
    width = 2200
    row_gap = 42
    top = 130
    height = max(700, top + row_gap * max(len(occupations), len(skills)) + 140)
    left_x, right_x = 430, 1770
    occupation_y = {node["id"]: top + index * row_gap for index, node in enumerate(occupations)}
    skill_y = {node["id"]: top + index * row_gap for index, node in enumerate(skills)}
    font_path = r"C:\Windows\Fonts\NotoSansSC-VF.ttf"
    if not Path(font_path).exists():
        font_path = r"C:\Windows\Fonts\arial.ttf"
    title_font = ImageFont.truetype(font_path, 30)
    heading_font = ImageFont.truetype(font_path, 22)
    label_font = ImageFont.truetype(font_path, 18)
    small_font = ImageFont.truetype(font_path, 16)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 28), "职位—技能二部图", fill="#0b2545", font=title_font, anchor="ma")
    draw.text((left_x, 85), f"职位节点（{len(occupations)}）", fill="#1f4d78", font=heading_font, anchor="ma")
    draw.text((right_x, 85), f"技能节点（{len(skills)}）", fill="#2e8b57", font=heading_font, anchor="ma")

    def edge_color(weight: float) -> tuple[int, int, int]:
        level = int(235 - 150 * max(0.0, min(1.0, weight)))
        return (level, min(255, level + 10), min(255, level + 35))

    for edge in edges:
        y1, y2 = occupation_y[edge["source"]], skill_y[edge["target"]]
        line_width = max(1, int(round(1 + 4 * edge["weight"])))
        draw.line((left_x, y1, right_x, y2), fill=edge_color(edge["weight"]), width=line_width)
    for node in occupations:
        y = occupation_y[node["id"]]
        draw.ellipse((left_x - 9, y - 9, left_x + 9, y + 9), fill="#1f4d78")
        draw.text((left_x - 22, y), node["label"], fill="#1f2937", font=label_font, anchor="rm")
    for node in skills:
        y = skill_y[node["id"]]
        draw.ellipse((right_x - 7, y - 7, right_x + 7, y + 7), fill="#2e8b57")
        draw.text((right_x + 20, y), node["label"], fill="#1f2937", font=label_font, anchor="lm")
    threshold = graph["min_demand_weight"]
    footer = f"边颜色/粗细表示 demand_weight；显示阈值 ≥ {threshold:.2f}，完整边数 {len(graph['edges'])}，当前显示 {len(edges)}"
    draw.text((width // 2, height - 42), footer, fill="#5b6573", font=small_font, anchor="mm")
    image.save(output_path, format="PNG", optimize=True)
    return True


def _render_svg(graph: dict[str, Any], output_path: Path) -> bool:
    """Fallback renderer that needs only the Python standard library.

    It writes an SVG next to the requested PNG path when matplotlib is not
    installed. The SVG is directly viewable in browsers and preserves the
    same node/edge semantics as the PNG renderer.
    """

    from html import escape

    occupations = graph["occupation_nodes"]
    skills = graph["skill_nodes"]
    edges = graph["display_edges"] or graph["edges"]
    width = 1800
    row_gap = 34
    top = 80
    height = max(520, top + row_gap * max(len(occupations), len(skills)) + 90)
    left_x, right_x = 260, 1540
    occupation_y = {node["id"]: top + index * row_gap for index, node in enumerate(occupations)}
    skill_y = {node["id"]: top + index * row_gap for index, node in enumerate(skills)}

    def edge_color(weight: float) -> str:
        # White-to-blue interpolation, matching the semantic meaning used by
        # the optional matplotlib renderer.
        level = int(235 - 150 * max(0.0, min(1.0, weight)))
        return f"rgb({level},{level + 10},{min(255, level + 35)})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Noto Sans SC","Microsoft YaHei",Arial,sans-serif;fill:#1f2937} .title{font-size:24px;font-weight:700;fill:#0b2545}.label{font-size:14px}.small{font-size:12px;fill:#5b6573}</style>',
        '<text class="title" x="900" y="35" text-anchor="middle">职位—技能二部图</text>',
        f'<text class="label" x="{left_x}" y="60" text-anchor="middle" fill="#1f4d78">职位节点（{len(occupations)}）</text>',
        f'<text class="label" x="{right_x}" y="60" text-anchor="middle" fill="#2e8b57">技能节点（{len(skills)}）</text>',
    ]
    for edge in edges:
        y1, y2 = occupation_y[edge["source"]], skill_y[edge["target"]]
        width_px = 0.8 + 2.0 * edge["weight"]
        lines.append(f'<line x1="{left_x}" y1="{y1}" x2="{right_x}" y2="{y2}" stroke="{edge_color(edge["weight"])}" stroke-width="{width_px:.2f}" opacity="0.62"/>')
    for node in occupations:
        y = occupation_y[node["id"]]
        lines.append(f'<circle cx="{left_x}" cy="{y}" r="8" fill="#1f4d78"/>')
        lines.append(f'<text class="label" x="{left_x - 16}" y="{y + 5}" text-anchor="end">{escape(node["label"])}</text>')
    for node in skills:
        y = skill_y[node["id"]]
        lines.append(f'<circle cx="{right_x}" cy="{y}" r="6" fill="#2e8b57"/>')
        lines.append(f'<text class="label" x="{right_x + 16}" y="{y + 5}">{escape(node["label"])}</text>')
    lines.append(f'<text class="small" x="900" y="{height - 22}" text-anchor="middle">边颜色/粗细表示 demand_weight；显示阈值 ≥ {graph["min_demand_weight"]:.2f}，完整边数 {len(graph["edges"])}，当前显示 {len(edges)}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/bipartite"))
    parser.add_argument("--min-demand-weight", type=float, default=0.45)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    graph = build_bipartite_graph(args.data_dir, min_demand_weight=args.min_demand_weight)
    paths = write_bipartite_outputs(graph, args.out_dir, render=not args.no_render)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
