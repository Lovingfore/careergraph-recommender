"""对 Topic 17 清洗数据与特征产物执行跨表完整性检查。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    """从指定目录读取 CSV，统一为后续校验提供 DataFrame。"""

    return pd.read_csv(data_dir / name)


def validate_clean_dataset(data_dir: Path) -> dict:
    """返回 clean CSV 的外键、重复键、概率和数值范围检查结果。"""

    occupations = _read(data_dir, "occupations.csv")
    skills = _read(data_dir, "skills.csv")
    occupation_skill = _read(data_dir, "occupation_skill.csv")
    profiles = _read(data_dir, "user_profiles.csv")
    events = _read(data_dir, "user_skill_events.csv")
    transitions = _read(data_dir, "job_transitions.csv")

    # 1. CSV 外键与孤儿 ID：构造主表 ID 集合，供关系表、画像、事件和转移表核对。
    occupation_ids = set(occupations["occupation_id"])
    skill_ids = set(skills["skill_id"])
    user_ids = set(profiles["user_id"])
    # 3. 概率范围：每个起始职位的出边概率必须在浮点容差内合计为 1。
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
        # 2. 主键/复合键：关系表和事件表不得出现重复业务键。
        "duplicate_occupation_skill_pairs": int(occupation_skill[["occupation_id", "skill_id"]].duplicated().sum()),
        "duplicate_event_keys": int(events[["user_id", "month", "skill_id"]].duplicated().sum()),
        "probability_sum_errors": probability_sum_errors,
        # 3. 数值范围：技能水平和职位需求权重必须处于归一化区间 [0, 1]。
        "skill_level_out_of_range": int(((events["level"] < 0) | (events["level"] > 1)).sum()),
        "demand_weight_out_of_range": int(((occupation_skill["demand_weight"] < 0) | (occupation_skill["demand_weight"] > 1)).sum()),
    }
    # 5. 汇总：任一错误列表非空或错误计数非零，整体 ok 即为 false。
    result["ok"] = not any(value for key, value in result.items() if key != "ok")
    return result


def validate_feature_outputs(data_dir: Path, processed_dir: Path) -> dict:
    """检查职位特征矩阵是否覆盖全部选择职位及用户目标职位。"""

    occupations = _read(data_dir, "occupations.csv")
    occupation_skill = _read(data_dir, "occupation_skill.csv")
    profiles = _read(data_dir, "user_profiles.csv")
    feature_jobs = _read(processed_dir, "job_feature_matrix.csv")
    feature_occupation_ids = sorted(set(feature_jobs["occupation_id"]))
    skill_occupation_ids = sorted(set(occupation_skill["occupation_id"]))
    selected_occupation_ids = sorted(set(occupations["occupation_id"]))
    target_jobs = set(profiles["target_job"])
    # 4. 特征覆盖：O*NET 选择职位、技能关系职位和矩阵职位应完全一致；用户画像
    # 中的所有目标职位也必须能在职位特征矩阵中找到。
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
