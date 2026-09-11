"""Generate a small, deterministic Chinese technical-resume dataset.

The records are synthetic and contain no real person's identity or contact
information.  O*NET remains the occupation/skill dictionary; this module only
creates the user-facing resume layer used by the course demo.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


EDUCATION = ("本科", "本科", "硕士", "大专")
MAJORS = ("计算机科学与技术", "软件工程", "数据科学与大数据技术", "网络工程", "信息安全")
CITIES = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安")
OCCUPATION_NAMES_ZH = {
    "Computer Systems Analysts": "计算机系统分析师",
    "Information Security Analysts": "信息安全分析师",
    "Database Administrators": "数据库管理员",
    "Database Architects": "数据库架构师",
    "Network and Computer Systems Administrators": "网络与计算机系统管理员",
    "Computer Programmers": "计算机程序员",
    "Software Developers": "软件开发工程师",
    "Software Quality Assurance Analysts and Testers": "软件质量保证分析师与测试工程师",
    "Web Developers": "网页开发工程师",
    "Web and Digital Interface Designers": "网页与数字界面设计师",
    "Operations Research Analysts": "运筹分析师",
    "Data Scientists": "数据科学家",
}
SKILL_NAMES_ZH = {
    "Reading Comprehension": "阅读理解", "Active Listening": "积极倾听", "Writing": "书面表达", "Speaking": "口头表达",
    "Mathematics": "数学", "Science": "科学", "Critical Thinking": "批判性思维", "Active Learning": "主动学习",
    "Learning Strategies": "学习策略", "Monitoring": "监控", "Social Perceptiveness": "社会洞察", "Coordination": "协调",
    "Persuasion": "说服", "Negotiation": "谈判", "Instructing": "指导", "Service Orientation": "服务导向",
    "Complex Problem Solving": "复杂问题解决", "Operations Analysis": "运营分析", "Technology Design": "技术设计",
    "Equipment Selection": "设备选型", "Installation": "安装", "Programming": "编程", "Operations Monitoring": "运营监控",
    "Operation and Control": "操作与控制", "Equipment Maintenance": "设备维护", "Troubleshooting": "故障排查",
    "Repairing": "修复", "Quality Control Analysis": "质量控制分析", "Judgment and Decision Making": "判断与决策",
    "Systems Analysis": "系统分析", "Systems Evaluation": "系统评估", "Time Management": "时间管理",
    "Management of Financial Resources": "财务资源管理", "Management of Material Resources": "物资资源管理",
    "Management of Personnel Resources": "人员资源管理",
}
TARGET_BY_CURRENT = {
    "15-1251.00": "15-1252.00",
    "15-1253.00": "15-1252.00",
    "15-1254.00": "15-1252.00",
    "15-1242.00": "15-2051.00",
    "15-1211.00": "15-2051.00",
    "15-1244.00": "15-1212.00",
    "15-2031.00": "15-2051.00",
}
LEGACY_PROFILES = (
    ("15-1251.00", "15-1252.00"),
    ("15-1254.00", "15-1252.00"),
    ("15-1242.00", "15-2051.00"),
    ("15-1244.00", "15-1212.00"),
    ("15-1253.00", "15-1252.00"),
    ("15-1211.00", "15-2051.00"),
)


def _skill_map(occupation_skill: pd.DataFrame) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for occupation_id, group in occupation_skill.groupby("occupation_id", sort=True):
        result[str(occupation_id)] = [
            {"skill_id": str(row.skill_id), "skill_name": str(row.skill_name), "demand_weight": float(row.demand_weight)}
            for row in group.itertuples(index=False)
        ]
    return result


def generate_resumes(
    occupations: pd.DataFrame,
    occupation_skill: pd.DataFrame,
    skills: pd.DataFrame,
    count: int = 300,
    seed: int = 20260911,
) -> pd.DataFrame:
    """Return ``count`` reproducible Chinese technical resumes."""

    if count < 1:
        raise ValueError("count must be >= 1")
    occupation_rows = occupations.sort_values("occupation_id").to_dict("records")
    occupation_names = {str(row["occupation_id"]): str(row["occupation_name"]) for row in occupation_rows}
    skill_names_en = skills.set_index("skill_id")["skill_name"].astype(str).to_dict()
    skill_names = {skill_id: SKILL_NAMES_ZH.get(name, name) for skill_id, name in skill_names_en.items()}
    skill_map = _skill_map(occupation_skill)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for index in range(count):
        if index < len(LEGACY_PROFILES) and all(job in occupation_names for job in LEGACY_PROFILES[index]):
            current, target = LEGACY_PROFILES[index]
        else:
            current = str(occupation_rows[index % len(occupation_rows)]["occupation_id"])
            target = TARGET_BY_CURRENT.get(current)
        if target is None or target == current:
            target = str(occupation_rows[(index + 1) % len(occupation_rows)]["occupation_id"])
        profile = skill_map[current]
        skill_levels: dict[str, float] = {}
        for skill in profile:
            # Keep the level close to the current occupation demand while
            # adding small individual differences for recommendation ranking.
            level = np.clip(0.20 + 0.85 * skill["demand_weight"] + rng.normal(0, 0.045), 0.05, 0.98)
            skill_levels[skill["skill_id"]] = round(float(level), 6)
        selected_skill_ids = [skill_id for skill_id, level in skill_levels.items() if level >= 0.28]
        selected_names = [skill_names.get(skill_id, skill_id) for skill_id in selected_skill_ids]
        education = EDUCATION[index % len(EDUCATION)]
        major = MAJORS[index % len(MAJORS)]
        city = CITIES[index % len(CITIES)]
        years = int(np.clip((index * 7) % 11, 0, 10))
        current_name = OCCUPATION_NAMES_ZH.get(occupation_names[current], occupation_names[current])
        target_name = OCCUPATION_NAMES_ZH.get(occupation_names[target], occupation_names[target])
        project = f"负责{current_name}相关项目，完成需求分析、代码实现、测试验证和上线复盘，使用{major}知识解决实际业务问题。"
        work = f"在{city}参与{current_name}工作{years}年，持续提升{ '、'.join(selected_names[:4]) or '基础技术能力'}等技能。"
        intro = f"我具备{education}学历，专业为{major}，目标方向是{target_name}。重视工程质量、团队协作和持续学习。"
        resume_text = "\n".join([
            f"求职方向：{target_name}",
            f"教育背景：{education}｜{major}",
            f"工作经历：{work}",
            f"项目经历：{project}",
            f"技能：{'、'.join(selected_names)}",
            f"自我评价：{intro}",
        ])
        rows.append({
            "resume_id": f"CR{index + 1:04d}",
            "user_id": f"u{index + 1:03d}",
            "education": education,
            "major": major,
            "years_experience": years,
            "city": city,
            "current_job": current,
            "current_job_name": current_name,
            "target_job": target,
            "target_job_name": target_name,
            "skills": "、".join(selected_names),
            "skill_levels": json.dumps(skill_levels, ensure_ascii=False, sort_keys=True),
            "work_experience": work,
            "project_experience": project,
            "self_introduction": intro,
            "resume_text": resume_text,
            "source": "synthetic_chinese_resume",
        })
    return pd.DataFrame(rows)


__all__ = ["generate_resumes"]
