"""Prepare the O*NET dictionary and a small synthetic Chinese resume dataset.

The O*NET occupation and skill tables are public reference data. The resume
records, monthly skill events and job transitions are deterministic synthetic
teaching data; they contain no real person's identity or contact information.
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

# O*NET 30.2 exposes detailed skill rows for some occupations through child
# SOC codes. Keep the intended parent occupation IDs in the project while
# recording the exact raw codes used as skill proxies/aggregates.
SKILL_SOURCE_CODES = {
    "15-1255.00": ["15-1255.01"],
    "15-2051.00": ["15-2051.01", "15-2051.02"],
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig", low_memory=False)


def find_onet_dir(raw_dir: Path) -> Path:
    candidates = [raw_dir / "db_30_2_text", raw_dir / "onet_30_2" / "db_30_2_text"]
    for candidate in candidates:
        if (candidate / "Occupation Data.txt").exists():
            return candidate
    matches = list(raw_dir.rglob("Occupation Data.txt"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError("Cannot find O*NET 'Occupation Data.txt' under " + str(raw_dir))


def make_occupations(onet_dir: Path, out_dir: Path) -> pd.DataFrame:
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
    if resumes is None:
        profiles = pd.DataFrame(USER_PROFILES)
    else:
        profiles = resumes[["user_id", "current_job", "target_job"]].copy()
    profiles.to_csv(out_dir / "user_profiles.csv", index=False, encoding="utf-8-sig")
    base = occupation_skill.pivot_table(index="occupation_id", columns="skill_id", values="demand_weight", fill_value=0)
    rng = np.random.default_rng(20260908)
    rows = []
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
    transitions = pd.DataFrame(TRANSITION_COUNTS, columns=["from_job", "to_job", "transition_count"])
    transitions = transitions[transitions["from_job"].isin(occupations["occupation_id"]) & transitions["to_job"].isin(occupations["occupation_id"])].copy()
    totals = transitions.groupby("from_job")["transition_count"].transform("sum")
    transitions["transition_probability"] = (transitions["transition_count"] / totals).round(6)
    transitions["source"] = "derived_teaching_data"
    transitions.to_csv(out_dir / "job_transitions.csv", index=False, encoding="utf-8-sig")
    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "clean")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    integrity = validate_clean_dataset(args.out_dir)
    if not integrity["ok"]:
        raise ValueError("Clean dataset integrity check failed: " + json.dumps(integrity, ensure_ascii=False))
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
