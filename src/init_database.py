"""Initialize and populate the Topic 17 SQLite database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both ``python src/init_database.py`` and module imports.
    from database import initialize_database, load_csv_data
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as src.init_database
    from src.database import initialize_database, load_csv_data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the Topic 17 SQLite database from clean CSV files")
    parser.add_argument("--db-path", type=Path, default=root / "artifacts" / "topic17.sqlite3")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "clean")
    args = parser.parse_args()
    initialize_database(args.db_path)
    counts = load_csv_data(args.db_path, args.data_dir)
    print(json.dumps({"db_path": str(args.db_path), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
