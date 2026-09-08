"""Cross-table integrity checks for the Topic 17 data package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / name)


def validate_clean_dataset(data_dir: Path) -> dict:
    """Return machine-readable referential and range checks for clean CSVs."""
    occupations = _read(data_dir, "occupations.csv")
    skills = _read(data_dir, "skills.csv")
    occupation_skill = _read(data_dir, "occupation_skill.csv")
    profiles = _read(data_dir, "user_profiles.csv")
    events = _read(data_dir, "user_skill_events.csv")
    transitions = _read(data_dir, "job_transitions.csv")

    occupation_ids = set(occupations["occupation_id"])
    skill_ids = set(skills["skill_id"])
    user_ids = set(profiles["user_id"])
    probability_sums = transitions.groupby("from_job")["transition_probability"].sum()
    probability_sum_errors = sorted(
        str(job) for job, total in probability_sums.items() if abs(float(total) - 1.0) > 1e-5
    )
    result = {
        "orphan_occupation_ids": sorted(set(occupation_skill["occupation_id"]) - occupation_ids),
        "orphan_skill_ids": sorted(set(occupation_skill["skill_id"]) - skill_ids),
        "invalid_profile_current_jobs": sorted(set(profiles["current_job"]) - occupation_ids),
        "invalid_profile_target_jobs": sorted(set(profiles["target_job"]) - occupation_ids),
        "invalid_event_user_ids": sorted(set(events["user_id"]) - user_ids),
        "invalid_event_skill_ids": sorted(set(events["skill_id"]) - skill_ids),
        "invalid_transition_from_jobs": sorted(set(transitions["from_job"]) - occupation_ids),
        "invalid_transition_to_jobs": sorted(set(transitions["to_job"]) - occupation_ids),
        "duplicate_occupation_skill_pairs": int(occupation_skill[["occupation_id", "skill_id"]].duplicated().sum()),
        "duplicate_event_keys": int(events[["user_id", "month", "skill_id"]].duplicated().sum()),
        "probability_sum_errors": probability_sum_errors,
        "skill_level_out_of_range": int(((events["level"] < 0) | (events["level"] > 1)).sum()),
        "demand_weight_out_of_range": int(((occupation_skill["demand_weight"] < 0) | (occupation_skill["demand_weight"] > 1)).sum()),
    }
    result["ok"] = not any(value for key, value in result.items() if key != "ok")
    return result


def validate_feature_outputs(data_dir: Path, processed_dir: Path) -> dict:
    """Check that generated feature matrices cover all selected/profile jobs."""
    occupations = _read(data_dir, "occupations.csv")
    occupation_skill = _read(data_dir, "occupation_skill.csv")
    profiles = _read(data_dir, "user_profiles.csv")
    feature_jobs = _read(processed_dir, "job_feature_matrix.csv")
    feature_occupation_ids = sorted(set(feature_jobs["occupation_id"]))
    skill_occupation_ids = sorted(set(occupation_skill["occupation_id"]))
    selected_occupation_ids = sorted(set(occupations["occupation_id"]))
    target_jobs = set(profiles["target_job"])
    result = {
        "selected_occupation_ids": selected_occupation_ids,
        "skill_occupation_ids": skill_occupation_ids,
        "feature_occupation_ids": feature_occupation_ids,
        "missing_skill_occupations": sorted(set(selected_occupation_ids) - set(skill_occupation_ids)),
        "missing_feature_occupations": sorted(set(selected_occupation_ids) - set(feature_occupation_ids)),
        "missing_target_jobs": sorted(target_jobs - set(feature_occupation_ids)),
    }
    result["ok"] = not any(
        result[key] for key in ["missing_skill_occupations", "missing_feature_occupations", "missing_target_jobs"]
    ) and result["feature_occupation_ids"] == result["skill_occupation_ids"]
    return result
