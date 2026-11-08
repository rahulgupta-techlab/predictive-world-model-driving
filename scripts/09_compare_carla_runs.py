"""Summarize completed CARLA mode runs; never mixes seeds or maps silently."""
import argparse
import csv
import json
from pathlib import Path


def load_runs(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(data)
    if not rows:
        raise ValueError("Provide at least one run summary JSON.")
    return rows


def summarize(rows):
    keys = {(row.get("seed"), row.get("map"), row.get("carla_version")) for row in rows}
    if len(keys) != 1:
        raise ValueError("Runs must share the same seed, map and CARLA version for a direct comparison.")
    modes = [row.get("mode") for row in rows]
    if len(set(modes)) != len(modes):
        raise ValueError("Only one summary per mode is accepted.")
    return [{
        "mode": row.get("mode"),
        "seed": row.get("seed"),
        "collisions": row.get("collision_count"),
        "distance_m": row.get("distance_travelled_m"),
        "mean_speed_mps": row.get("mean_speed_mps"),
        "minimum_predicted_clearance_m": row.get("minimum_predicted_clearance_m"),
        "fallback_interventions": row.get("fallback_interventions"),
        "simulated_frames": row.get("simulated_frames")
    } for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", default="experiments/results/comparison.csv")
    args = parser.parse_args()
    rows = summarize(load_runs(args.summaries))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
