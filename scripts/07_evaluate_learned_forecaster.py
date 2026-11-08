"""Evaluate a trained LSTM/Transformer checkpoint on a held-out split."""
import argparse
import json
from pathlib import Path
import numpy as np
from pwmd.prediction.forecasters import load_forecaster, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/trajectories.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--output", default="experiments/results/learned_forecasting.json")
    args = parser.parse_args()
    path = Path(args.dataset)
    if not path.exists():
        parser.error(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=False)
    mask = data["split"] == args.split
    x, y = data["x"][mask], data["y"][mask]
    if not len(x):
        raise SystemExit(f"No '{args.split}' samples; collect enough distinct episodes first.")
    model, device, meta = load_forecaster(args.checkpoint)
    if x.shape[1] != meta["obs_steps"] or y.shape[1] != meta["pred_steps"]:
        raise SystemExit("Dataset window sizes do not match checkpoint metadata.")
    pred = predict(model, device, x)
    errors = np.linalg.norm(pred - y, axis=-1)
    result = {**meta, "split": args.split, "samples": int(len(x)),
              "ADE_m": float(errors.mean()), "FDE_m": float(errors[:, -1].mean()),
              "miss_rate": float((errors[:, -1] > args.miss_threshold).mean()),
              "miss_threshold_m": args.miss_threshold, "dataset": str(path),
              "checkpoint": args.checkpoint, "device": str(device),
              "warning": "Check episode-level split integrity and dataset quality before interpreting metrics."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
