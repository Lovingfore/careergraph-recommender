"""SQLite schema, transactional CSV importer, and query helpers for Topic 17.

The CSV files under ``data/clean`` remain the auditable exchange format.  This
module provides one shared database service used by the command line verifier
and the Django web layer; it deliberately does not introduce a second ORM
schema.
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
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path) -> None:
    """Create the six core tables and indexes if they do not already exist."""

    conn = _connect(Path(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _required_columns(rows: list[dict[str, str]], path: Path, columns: set[str]) -> None:
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"CSV {path} is missing columns: {sorted(missing)}")


def load_csv_data(db_path: Path, data_dir: Path) -> dict[str, int]:
    """Replace database contents with clean CSV data in one transaction.

    Any parsing, constraint, or post-import integrity error raises an
    exception and rolls the transaction back, leaving the previous database
    contents untouched.
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
        # Delete children first so this also works with foreign_keys enabled.
        for table in ("user_skill_events", "job_transitions", "user_profiles", "occupation_skill", "skills", "occupations"):
            conn.execute(f"DELETE FROM {table}")

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
        _validate_connection(conn)
        conn.commit()
        return get_counts_from_connection(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _validate_connection(conn: sqlite3.Connection) -> None:
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
    return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in TABLE_NAMES}


def get_counts(db_path: Path) -> dict[str, int]:
    conn = _connect(Path(db_path))
    try:
        return get_counts_from_connection(conn)
    finally:
        conn.close()


def query_user_profile(db_path: Path, user_id: str) -> dict[str, Any]:
    """Return a user profile, job names, and chronological skill events."""

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
