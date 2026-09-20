"""Topic 17 的 SQLite 数据库、事务式 CSV 导入器和查询辅助函数。

``data/clean`` 下的 CSV 是便于审计、交换和重建的事实层；SQLite 是供
验收脚本与 Django Web 层查询的运行时查询层。本模块只维护这一份关系
模式，不再引入另一套 ORM 表结构。
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any


TABLE_NAMES = (
    "occupations",
    "skills",
    "occupation_skill",
    "user_profiles",
    "user_skill_events",
    "job_transitions",
)


# 表的创建顺序体现外键依赖：先建职位/技能字典，再建关系、用户和转移表；
# 索引最后创建，保证导入和查询都使用当前六张核心表。
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS occupations (
    occupation_id TEXT PRIMARY KEY,
    occupation_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    skill_category TEXT NOT NULL DEFAULT 'general'
);

CREATE TABLE IF NOT EXISTS occupation_skill (
    occupation_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    importance REAL NOT NULL,
    level REAL NOT NULL,
    importance_norm REAL NOT NULL,
    level_norm REAL NOT NULL,
    demand_weight REAL NOT NULL,
    is_required INTEGER NOT NULL CHECK (is_required IN (0, 1)),
    skill_source TEXT NOT NULL,
    PRIMARY KEY (occupation_id, skill_id),
    FOREIGN KEY (occupation_id) REFERENCES occupations (occupation_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    current_job TEXT NOT NULL,
    target_job TEXT NOT NULL,
    FOREIGN KEY (current_job) REFERENCES occupations (occupation_id),
    FOREIGN KEY (target_job) REFERENCES occupations (occupation_id)
);

CREATE TABLE IF NOT EXISTS user_skill_events (
    user_id TEXT NOT NULL,
    month INTEGER NOT NULL CHECK (month >= 0),
    skill_id TEXT NOT NULL,
    level REAL NOT NULL CHECK (level >= 0 AND level <= 1),
    source TEXT NOT NULL,
    PRIMARY KEY (user_id, month, skill_id),
    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS job_transitions (
    from_job TEXT NOT NULL,
    to_job TEXT NOT NULL,
    transition_count INTEGER NOT NULL CHECK (transition_count >= 0),
    transition_probability REAL NOT NULL CHECK (transition_probability >= 0 AND transition_probability <= 1),
    source TEXT NOT NULL,
    PRIMARY KEY (from_job, to_job),
    FOREIGN KEY (from_job) REFERENCES occupations (occupation_id) ON DELETE CASCADE,
    FOREIGN KEY (to_job) REFERENCES occupations (occupation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_occupation_skill_skill ON occupation_skill (skill_id);
CREATE INDEX IF NOT EXISTS idx_events_user_month ON user_skill_events (user_id, month);
CREATE INDEX IF NOT EXISTS idx_events_skill ON user_skill_events (skill_id);
CREATE INDEX IF NOT EXISTS idx_transitions_from ON job_transitions (from_job);
CREATE INDEX IF NOT EXISTS idx_transitions_to ON job_transitions (to_job);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    """打开数据库连接，启用外键约束，并返回按列名访问的行对象。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path) -> None:
    """创建六张核心表及索引；已存在的表保持不变，供后续导入复用。"""

    conn = _connect(Path(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """以 UTF-8（兼容 BOM）读取一个 CSV，先转成字典行供契约校验和插入。"""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _required_columns(rows: list[dict[str, str]], path: Path, columns: set[str]) -> None:
    """检查 CSV 非空且包含调用方声明的字段契约，避免半途才暴露列缺失。"""
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"CSV {path} is missing columns: {sorted(missing)}")


def load_csv_data(db_path: Path, data_dir: Path) -> dict[str, int]:
    """按外键依赖顺序将 clean CSV 全量替换进数据库，并返回各表行数。

    六个文件先全部读取和校验，随后在同一事务内删除旧数据、批量插入新
    数据并执行一致性检查。任何解析、约束或校验异常都会回滚，因此失败
    时保留导入前的数据库内容，避免出现半套数据。
    """

    data_dir = Path(data_dir)
    files = {name: data_dir / f"{name}.csv" for name in TABLE_NAMES}
    rows = {name: _read_csv(path) for name, path in files.items()}
    _required_columns(rows["occupations"], files["occupations"], {"occupation_id", "occupation_name", "Description"})
    _required_columns(rows["skills"], files["skills"], {"skill_id", "skill_name", "skill_category"})
    _required_columns(rows["occupation_skill"], files["occupation_skill"], {"occupation_id", "skill_id", "skill_name", "importance", "level", "importance_norm", "level_norm", "demand_weight", "is_required", "skill_source"})
    _required_columns(rows["user_profiles"], files["user_profiles"], {"user_id", "current_job", "target_job"})
    _required_columns(rows["user_skill_events"], files["user_skill_events"], {"user_id", "month", "skill_id", "level", "source"})
    _required_columns(rows["job_transitions"], files["job_transitions"], {"from_job", "to_job", "transition_count", "transition_probability", "source"})

    conn = _connect(Path(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        # 删除与插入都按外键依赖处理：先删子表再删父表，避免外键约束阻止
        # 重建；后续插入则反向先写字典，再写关系和事件。
        for table in ("user_skill_events", "job_transitions", "user_profiles", "occupation_skill", "skills", "occupations"):
            conn.execute(f"DELETE FROM {table}")

        # executemany 将 CSV 行批量写入，减少逐行往返，同时仍由 SQLite
        # 外键和 CHECK 约束拦截非法值。
        conn.executemany(
            "INSERT INTO occupations (occupation_id, occupation_name, description) VALUES (?, ?, ?)",
            [(r["occupation_id"], r["occupation_name"], r.get("Description", "")) for r in rows["occupations"]],
        )
        conn.executemany(
            "INSERT INTO skills (skill_id, skill_name, skill_category) VALUES (?, ?, ?)",
            [(r["skill_id"], r["skill_name"], r["skill_category"]) for r in rows["skills"]],
        )
        conn.executemany(
            """INSERT INTO occupation_skill
            (occupation_id, skill_id, skill_name, importance, level, importance_norm,
             level_norm, demand_weight, is_required, skill_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r["occupation_id"], r["skill_id"], r["skill_name"], float(r["importance"]),
                    float(r["level"]), float(r["importance_norm"]), float(r["level_norm"]),
                    float(r["demand_weight"]), int(r["is_required"]), r["skill_source"],
                )
                for r in rows["occupation_skill"]
            ],
        )
        conn.executemany(
            "INSERT INTO user_profiles (user_id, current_job, target_job) VALUES (?, ?, ?)",
            [(r["user_id"], r["current_job"], r["target_job"]) for r in rows["user_profiles"]],
        )
        conn.executemany(
            "INSERT INTO user_skill_events (user_id, month, skill_id, level, source) VALUES (?, ?, ?, ?, ?)",
            [(r["user_id"], int(r["month"]), r["skill_id"], float(r["level"]), r["source"]) for r in rows["user_skill_events"]],
        )
        conn.executemany(
            "INSERT INTO job_transitions (from_job, to_job, transition_count, transition_probability, source) VALUES (?, ?, ?, ?, ?)",
            [(r["from_job"], r["to_job"], int(r["transition_count"]), float(r["transition_probability"]), r["source"]) for r in rows["job_transitions"]],
        )
        # 提交前再次做孤儿行和转移概率校验；只有全部通过才让新快照生效。
        _validate_connection(conn)
        conn.commit()
        return get_counts_from_connection(conn)
    except Exception:
        # 回滚覆盖删除、插入和校验期间的全部写入，调用方可据异常定位问题。
        conn.rollback()
        raise
    finally:
        conn.close()


def _validate_connection(conn: sqlite3.Connection) -> None:
    """检查孤儿外键引用和各起始职位的转移概率和。

    CSV 是否为空由导入前的 ``_required_columns`` 负责，本函数只验证已经写入
    当前连接的数据之间是否保持引用和概率一致性。
    """
    orphan_checks = (
        ("occupation_skill", "occupation_id", "occupations", "occupation_id"),
        ("occupation_skill", "skill_id", "skills", "skill_id"),
        ("user_profiles", "current_job", "occupations", "occupation_id"),
        ("user_profiles", "target_job", "occupations", "occupation_id"),
        ("user_skill_events", "user_id", "user_profiles", "user_id"),
        ("user_skill_events", "skill_id", "skills", "skill_id"),
        ("job_transitions", "from_job", "occupations", "occupation_id"),
        ("job_transitions", "to_job", "occupations", "occupation_id"),
    )
    for child, child_col, parent, parent_col in orphan_checks:
        query = f"SELECT COUNT(*) FROM {child} c LEFT JOIN {parent} p ON c.{child_col}=p.{parent_col} WHERE p.{parent_col} IS NULL"
        if conn.execute(query).fetchone()[0]:
            raise ValueError(f"orphan rows in {child}.{child_col}")
    bad_probability = conn.execute(
        "SELECT from_job FROM job_transitions GROUP BY from_job HAVING ABS(SUM(transition_probability) - 1.0) > 0.00001"
    ).fetchall()
    if bad_probability:
        raise ValueError(f"transition probabilities do not sum to 1: {[row[0] for row in bad_probability]}")


def get_counts_from_connection(conn: sqlite3.Connection) -> dict[str, int]:
    """在现有连接上返回六张核心表的行数摘要，不额外开启事务。"""
    return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in TABLE_NAMES}


def get_counts(db_path: Path) -> dict[str, int]:
    """打开并关闭独立连接，返回数据库各核心表的行数摘要。"""
    conn = _connect(Path(db_path))
    try:
        return get_counts_from_connection(conn)
    finally:
        conn.close()


def query_user_profile(db_path: Path, user_id: str) -> dict[str, Any]:
    """查询用户画像、当前/目标职位名称及按月份排序的技能事件。

    返回字典同时保留原始职位 ID、可读名称、事件列表和事件数量；未知
    ``user_id`` 以 ``KeyError`` 明确告知调用方，而不是返回空画像。
    """

    conn = _connect(Path(db_path))
    try:
        profile = conn.execute(
            """SELECT p.user_id, p.current_job, p.target_job,
                      cj.occupation_name AS current_job_name,
                      tj.occupation_name AS target_job_name
               FROM user_profiles p
               JOIN occupations cj ON cj.occupation_id = p.current_job
               JOIN occupations tj ON tj.occupation_id = p.target_job
              WHERE p.user_id = ?""",
            (user_id,),
        ).fetchone()
        if profile is None:
            raise KeyError(f"Unknown user_id: {user_id}")
        events = [dict(row) for row in conn.execute(
            "SELECT user_id, month, skill_id, level, source FROM user_skill_events WHERE user_id = ? ORDER BY month, skill_id",
            (user_id,),
        )]
        result = dict(profile)
        result["events"] = events
        result["event_count"] = len(events)
        return result
    finally:
        conn.close()


__all__ = [
    "SCHEMA_SQL",
    "TABLE_NAMES",
    "initialize_database",
    "load_csv_data",
    "get_counts",
    "query_user_profile",
]
