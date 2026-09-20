"""准备 O*NET 职位/技能字典与确定性合成中文简历数据集。

O*NET 30.2 的职位和技能表属于公开参考数据；简历记录、月度技能事件和职位
转移记录均为可复现的合成教学数据，不包含任何真实个人身份或联系方式。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_integrity import validate_clean_dataset
from generate_chinese_resumes import generate_resumes


OCCUPATION_CODES = [
    "15-1211.00",  # Computer Systems Analysts
    "15-1212.00",  # Information Security Analysts
    "15-1242.00",  # Database Administrators
    "15-1243.00",  # Database Architects
    "15-1244.00",  # Network and Computer Systems Administrators
    "15-1251.00",  # Computer Programmers
    "15-1252.00",  # Software Developers
    "15-1253.00",  # Software Quality Assurance Analysts and Testers
    "15-1254.00",  # Web Developers
    "15-1255.00",  # Web and Digital Interface Designers
    "15-2031.00",  # Operations Research Analysts
    "15-2051.00",  # Data Scientists
]

USER_PROFILES = [
    {"user_id": "u001", "current_job": "15-1251.00", "target_job": "15-1252.00"},
    {"user_id": "u002", "current_job": "15-1254.00", "target_job": "15-1252.00"},
    {"user_id": "u003", "current_job": "15-1242.00", "target_job": "15-2051.00"},
    {"user_id": "u004", "current_job": "15-1244.00", "target_job": "15-1212.00"},
    {"user_id": "u005", "current_job": "15-1253.00", "target_job": "15-1252.00"},
    {"user_id": "u006", "current_job": "15-1211.00", "target_job": "15-2051.00"},
]

TRANSITION_COUNTS = [
    ("15-1251.00", "15-1252.00", 24),
    ("15-1254.00", "15-1252.00", 18),
    ("15-1253.00", "15-1252.00", 16),
    ("15-1242.00", "15-2051.00", 9),
    ("15-1211.00", "15-2051.00", 12),
    ("15-1244.00", "15-1212.00", 15),
    ("15-1252.00", "15-2051.00", 10),
    ("15-1252.00", "15-1212.00", 8),
    ("15-1251.00", "15-1254.00", 10),
    ("15-1254.00", "15-1255.00", 12),
    ("15-1251.00", "15-2031.00", 5),
    ("15-2031.00", "15-2051.00", 9),
]

# O*NET 30.2 的少数职位只在子 SOC 编码中提供详细技能行。项目对外继续使用
# 预定的父职位 ID，同时记录实际作为代理或聚合来源的原始子编码。
SKILL_SOURCE_CODES = {
    "15-1255.00": ["15-1255.01"],
    "15-2051.00": ["15-2051.01", "15-2051.02"],
}


def read_tsv(path: Path) -> pd.DataFrame:
    """读取 O*NET 制表符文件，并以 UTF-8 字符串友好的方式返回 DataFrame。"""

    return pd.read_csv(path, sep="\t", encoding="utf-8-sig", low_memory=False)


def find_onet_dir(raw_dir: Path) -> Path:
    """在常见目录结构及递归子目录中定位 O*NET 原始文本目录。"""

    candidates = [raw_dir / "db_30_2_text", raw_dir / "onet_30_2" / "db_30_2_text"]
    for candidate in candidates:
        if (candidate / "Occupation Data.txt").exists():
            return candidate
    matches = list(raw_dir.rglob("Occupation Data.txt"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError("Cannot find O*NET 'Occupation Data.txt' under " + str(raw_dir))


def make_occupations(onet_dir: Path, out_dir: Path) -> pd.DataFrame:
    """筛选项目指定的 12 个职业，规范字段后写入 ``occupations.csv``。"""

    # 职业筛选严格使用锁定的 O*NET-SOC 编码；任何缺失都会中止流水线，
    # 防止下游在不完整职位集合上静默生成特征和推荐。
    source = read_tsv(onet_dir / "Occupation Data.txt")
    code_col = "O*NET-SOC Code"
    selected = source[source[code_col].isin(OCCUPATION_CODES)].copy()
    missing = sorted(set(OCCUPATION_CODES) - set(selected[code_col]))
    if missing:
        raise ValueError(f"Selected occupation codes are missing from O*NET: {missing}")
    selected = selected.rename(columns={code_col: "occupation_id", "Title": "occupation_name"})
    selected = selected[["occupation_id", "occupation_name", "Description"]].sort_values("occupation_id")
    selected.to_csv(out_dir / "occupations.csv", index=False, encoding="utf-8-sig")
    return selected


def make_occupation_skills(onet_dir: Path, occupations: pd.DataFrame, out_dir: Path):
    """清洗职位技能关系，归一化需求权重并输出关系表与技能字典。"""

    raw = read_tsv(onet_dir / "Skills.txt")
    selected_ids = set(occupations["occupation_id"])
    source_to_target = {
        source: target
        for target in selected_ids
        for source in SKILL_SOURCE_CODES.get(target, [target])
    }
    raw = raw[raw["O*NET-SOC Code"].isin(source_to_target)].copy()
    raw["target_occupation_id"] = raw["O*NET-SOC Code"].map(source_to_target)
    raw = raw[raw["Scale ID"].isin(["IM", "LV"])]
    raw = raw[raw["Recommend Suppress"] != "Y"].copy()
    raw["Data Value"] = pd.to_numeric(raw["Data Value"], errors="coerce")
    raw = raw.dropna(subset=["Data Value"])
    pivot = raw.pivot_table(
        index=["target_occupation_id", "Element ID", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.rename(
        columns={"target_occupation_id": "occupation_id", "Element ID": "skill_id", "Element Name": "skill_name", "IM": "importance", "LV": "level"}
    )
    pivot["importance"] = pivot["importance"].fillna(0)
    pivot["level"] = pivot["level"].fillna(0)
    # O*NET 的重要性量表上限为 5、等级量表上限为 7；分别归一化后相乘，
    # 得到 [0, 1] 的 demand_weight，供推荐、缺口分析和图谱边权共同使用。
    pivot["importance_norm"] = (pivot["importance"] / 5.0).clip(0, 1)
    pivot["level_norm"] = (pivot["level"] / 7.0).clip(0, 1)
    pivot["demand_weight"] = (pivot["importance_norm"] * pivot["level_norm"]).round(6)
    pivot["is_required"] = (pivot["demand_weight"] >= 0.28).astype(int)
    pivot["skill_source"] = pivot["occupation_id"].map(
        lambda occupation_id: ",".join(SKILL_SOURCE_CODES.get(occupation_id, [occupation_id]))
    )
    pivot = pivot.sort_values(["occupation_id", "skill_id"])
    pivot.to_csv(out_dir / "occupation_skill.csv", index=False, encoding="utf-8-sig")

    skills = pivot[["skill_id", "skill_name"]].drop_duplicates().sort_values("skill_id")
    skills["skill_category"] = skills["skill_name"].map(
        lambda value: "cognitive" if value in {"Reading Comprehension", "Critical Thinking", "Complex Problem Solving", "Systems Analysis", "Systems Evaluation"}
        else "technical" if value in {"Programming", "Technology Design", "Operations Analysis", "Troubleshooting", "Quality Control Analysis"}
        else "communication" if value in {"Writing", "Speaking", "Active Listening", "Social Perceptiveness"}
        else "general"
    )
    skills.to_csv(out_dir / "skills.csv", index=False, encoding="utf-8-sig")
    return pivot, skills


def make_users_and_events(occupation_skill: pd.DataFrame, out_dir: Path, resumes: pd.DataFrame | None = None):
    """从简历画像或内置规则生成用户表，并展开连续六个月技能事件。"""

    if resumes is None:
        profiles = pd.DataFrame(USER_PROFILES)
    else:
        profiles = resumes[["user_id", "current_job", "target_job"]].copy()
    profiles.to_csv(out_dir / "user_profiles.csv", index=False, encoding="utf-8-sig")
    base = occupation_skill.pivot_table(index="occupation_id", columns="skill_id", values="demand_weight", fill_value=0)
    rng = np.random.default_rng(20260908)
    rows = []
    # 每个用户以当前职位需求或简历技能水平为初值，再按月份加入确定性趋势和
    # 固定随机种子的微小扰动，形成 DataLoader 使用的用户—月份—技能长表。
    for profile in profiles.to_dict("records"):
        vector = base.loc[profile["current_job"]]
        resume_levels = {}
        if resumes is not None:
            resume_row = resumes.loc[resumes["user_id"] == profile["user_id"]].iloc[0]
            resume_levels = {str(key): float(value) for key, value in json.loads(resume_row["skill_levels"]).items()}
        for month in range(6):
            for skill_id, demand in vector.items():
                initial = resume_levels.get(str(skill_id), float(np.clip(0.18 + 0.68 * demand, 0, 1)))
                trend = 0.018 * month if demand > 0.25 else 0.008 * month
                noise = rng.normal(0, 0.012)
                level = float(np.clip(initial + trend + noise, 0, 1))
                rows.append({
                    "user_id": profile["user_id"],
                    "month": month,
                    "skill_id": skill_id,
                    "level": round(level, 6),
                    "source": "synthetic_chinese_resume",
                })
    events = pd.DataFrame(rows)
    events.to_csv(out_dir / "user_skill_events.csv", index=False, encoding="utf-8-sig")
    return profiles, events


def make_transitions(occupations: pd.DataFrame, out_dir: Path):
    """生成教学用职位转移记录，并按起始职位归一化转移概率。"""

    transitions = pd.DataFrame(TRANSITION_COUNTS, columns=["from_job", "to_job", "transition_count"])
    transitions = transitions[transitions["from_job"].isin(occupations["occupation_id"]) & transitions["to_job"].isin(occupations["occupation_id"])].copy()
    # 同一起始职位的计数除以该职位总计，使每组转移概率之和为 1。
    totals = transitions.groupby("from_job")["transition_count"].transform("sum")
    transitions["transition_probability"] = (transitions["transition_count"] / totals).round(6)
    transitions["source"] = "derived_teaching_data"
    transitions.to_csv(out_dir / "job_transitions.csv", index=False, encoding="utf-8-sig")
    return transitions


def main():
    """编排原始定位、清洗、合成数据生成、完整性校验和摘要落盘。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "clean")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # 离线构建顺序：定位 O*NET → 职位与技能清洗 → 简历/事件合成 → 转移数据。
    onet_dir = find_onet_dir(args.raw_dir)
    occupations = make_occupations(onet_dir, args.out_dir)
    occupation_skill, skills = make_occupation_skills(onet_dir, occupations, args.out_dir)
    resumes = generate_resumes(occupations, occupation_skill, skills, count=300, seed=20260911)
    resumes.to_csv(args.out_dir / "resumes_zh.csv", index=False, encoding="utf-8-sig")
    profiles, events = make_users_and_events(occupation_skill, args.out_dir, resumes=resumes)
    transitions = make_transitions(occupations, args.out_dir)
    clean_log = pd.DataFrame([
        {"stage": "occupation_filter", "input": "O*NET Occupation Data.txt", "output": "occupations.csv", "action": "select 12 software/data/network occupations", "rows": len(occupations)},
        {"stage": "skill_filter", "input": "O*NET Skills.txt", "output": "occupation_skill.csv", "action": "keep IM/LV, remove suppressed values, fill numeric values", "rows": len(occupation_skill)},
        {"stage": "skill_dictionary", "input": "occupation_skill.csv", "output": "skills.csv", "action": "deduplicate skill_id and assign simple categories", "rows": len(skills)},
        {"stage": "synthetic_resume_profile", "input": "resumes_zh.csv", "output": "user_profiles.csv and user_skill_events.csv", "action": "generate 300 synthetic Chinese technical resumes and six months of skill events", "rows": len(profiles)},
        {"stage": "teaching_transition", "input": "selected occupations", "output": "job_transitions.csv", "action": "deterministically derive transition counts and normalize probabilities", "rows": len(transitions)},
    ])
    clean_log.to_csv(args.out_dir / "clean_log.csv", index=False, encoding="utf-8-sig")
    # 所有 CSV 写完后立即检查外键、重复键、数值范围和概率和；失败则不产出
    # 看似成功的摘要，提醒调用方修复数据问题。
    integrity = validate_clean_dataset(args.out_dir)
    if not integrity["ok"]:
        raise ValueError("Clean dataset integrity check failed: " + json.dumps(integrity, ensure_ascii=False))
    # dataset_summary.json 记录公开数据来源、许可、产物规模与合成数据边界，
    # 供人工审计和最终报告读取；Web 摘要与阶段验收使用各自的实时统计链路。
    summary = {
        "source": "O*NET 30.2 Database, February 2026 release",
        "source_url": "https://www.onetcenter.org/database.html",
        "license": "CC BY 4.0",
        "selected_occupations": int(len(occupations)),
        "skills": int(len(skills)),
        "occupation_skill_rows": int(len(occupation_skill)),
        "resumes": int(len(resumes)),
        "users": int(len(profiles)),
        "user_skill_event_rows": int(len(events)),
        "transition_rows": int(len(transitions)),
        "note": "O*NET tables are public source data. resumes_zh.csv, user_skill_events.csv and job_transitions.csv are deterministic synthetic teaching data and contain no real personal information.",
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
