"""A transparent six-month skill forecast baseline using per-skill linear trend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "clean")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.data_dir / "user_skill_events.csv")
    rows = []
    for (user_id, skill_id), group in events.groupby(["user_id", "skill_id"]):
        group = group.sort_values("month")
        x = group["month"].to_numpy(dtype=float)
        y = group["level"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(group) >= 2 else 0.0
        last_level = float(group.iloc[-1]["level"])
        forecast = float(np.clip(last_level + slope * 6, 0, 1))
        rows.append({
            "user_id": user_id,
            "skill_id": skill_id,
            "last_level": round(last_level, 6),
            "monthly_slope": round(slope, 6),
            "forecast_6m": round(forecast, 6),
        })
    output = pd.DataFrame(rows)
    output.to_csv(args.out_dir / "skill_forecast_6m.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(output)} six-month skill forecasts")


if __name__ == "__main__":
    main()
