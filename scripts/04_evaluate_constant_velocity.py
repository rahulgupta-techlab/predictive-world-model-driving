"""Evaluate a transparent constant-velocity trajectory forecasting baseline."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(pred, target, miss_threshold=2.0):
    # pred/target: [samples, future_steps, 2]
    errors = np.linalg.norm(pred - target, axis=-1)
    ade = float(errors.mean())
    fde = float(errors[:, -1].mean())
    miss_rate = float((errors[:, -1] > miss_threshold).mean())
    return {"ADE_m": ade, "FDE_m": fde, "miss_rate": miss_rate,
            "miss_threshold_m": miss_threshold, "samples": int(len(errors))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/processed/trajectories.npz")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--dt", type=float, default=0.25, help="Seconds between trajectory samples")
    p.add_argument("--miss-threshold", type=float, default=2.0)
    p.add_argument("--output", default="experiments/results/constant_velocity.json")
    args = p.parse_args()

    data_path = Path(args.dataset)
    if not data_path.exists():
        p.error(f"Dataset not found: {data_path}. Run scripts/03_prepare_trajectory_dataset.py first.")
    data = np.load(data_path, allow_pickle=False)
    mask = data["split"] == args.split
    x, y = data["x"][mask], data["y"][mask]
    if len(x) == 0:
        raise RuntimeError(f"No '{args.split}' samples. Collect more distinct episodes; the split is episode-level.")

    # Estimate velocity using the final two observed points, then extrapolate.
    velocity = (x[:, -1, :] - x[:, -2, :]) / args.dt
    horizon = np.arange(1, y.shape[1] + 1, dtype=np.float32) * args.dt
    pred = x[:, -1:, :] + velocity[:, None, :] * horizon[None, :, None]
    result = {
        "model": "constant_velocity",
        "split": args.split,
        "dataset": str(data_path),
        "sample_interval_seconds": args.dt,
        **metrics(pred, y, args.miss_threshold),
        "warning": "Metrics are meaningful only on a held-out split from multiple episodes."
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
