"""Build actor-centric trajectory windows from CARLA collector CSV files.

The collector records world-coordinate actor states at the simulation rate. This
script subsamples tracks, converts each window to the actor's local frame at the
last observed step, and writes train/validation/test arrays to an NPZ file.
Episodes, not windows, are assigned to splits to reduce temporal leakage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rotate_to_local(points: np.ndarray, origin: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Translate points by origin and rotate world XY into a heading-aligned frame."""
    delta = points[:, :2] - origin[:2]
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]))


def episode_split(episode_ids, seed=42):
    ids = sorted(set(episode_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if len(ids) < 3:
        # Avoid pretending a tiny dataset supports independent evaluation.
        return {eid: "train" for eid in ids}
    n = len(ids)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    return {eid: ("test" if i < n_test else "validation" if i < n_test + n_val else "train")
            for i, eid in enumerate(ids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output", default="data/processed/trajectories.npz")
    p.add_argument("--config", default="configs/collection.yaml")
    p.add_argument("--obs-steps", type=int, default=8, help="Observed samples (2 s at 4 Hz)")
    p.add_argument("--pred-steps", type=int, default=12, help="Future samples (3 s at 4 Hz)")
    p.add_argument("--sample-stride", type=int, default=5, help="Subsample 20 Hz logs to 4 Hz")
    p.add_argument("--min-speed", type=float, default=0.0, help="Minimum actor speed in m/s; keep zero to include stopped actors")
    args = p.parse_args()

    root = Path(args.data_dir)
    output = Path(args.output)
    if args.obs_steps < 2 or args.pred_steps < 1 or args.sample_stride < 1:
        p.error("obs-steps >= 2, pred-steps >= 1, and sample-stride >= 1 are required")
    if not root.exists():
        p.error(f"Data directory does not exist: {root}")

    samples, episode_labels, actor_labels = [], [], []
    episode_dirs = sorted(d for d in root.glob("*/*") if d.is_dir())
    for episode in episode_dirs:
        csv_path = episode / "actors.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        df = pd.read_csv(csv_path)
        required = {"frame", "actor_id", "x", "y", "yaw_deg", "speed_mps"}
        if not required.issubset(df.columns):
            print(f"Skipping {csv_path}: missing columns {sorted(required - set(df.columns))}")
            continue
        # Keep a regular temporal grid; CARLA logs contain one row per actor per tick.
        df = df.sort_values(["actor_id", "frame"])
        for actor_id, track in df.groupby("actor_id"):
            track = track.drop_duplicates("frame").sort_values("frame").iloc[::args.sample_stride]
            frames = track["frame"].to_numpy(dtype=np.int64)
            xy = track[["x", "y"]].to_numpy(dtype=np.float32)
            yaw = np.deg2rad(track["yaw_deg"].to_numpy(dtype=np.float32))
            speed = track["speed_mps"].to_numpy(dtype=np.float32)
            total = args.obs_steps + args.pred_steps
            for start in range(0, len(track) - total + 1):
                # Reject gaps so every window has a fixed temporal interval.
                if not np.all(np.diff(frames[start:start + total]) == args.sample_stride):
                    continue
                end_obs = start + args.obs_steps
                if float(np.nanmean(speed[start:end_obs])) < args.min_speed:
                    continue
                window_xy = xy[start:start + total]
                anchor = window_xy[args.obs_steps - 1]
                anchor_yaw = float(yaw[end_obs - 1])
                local = rotate_to_local(window_xy, anchor, anchor_yaw)
                if not np.isfinite(local).all():
                    continue
                samples.append((local[:args.obs_steps], local[args.obs_steps:]))
                episode_labels.append(episode.name)
                actor_labels.append(str(actor_id))

    if not samples:
        raise RuntimeError("No trajectory windows found. Collect episodes first or reduce min-speed/window lengths.")
    split_map = episode_split(episode_labels)
    x = np.stack([s[0] for s in samples]).astype(np.float32)
    y = np.stack([s[1] for s in samples]).astype(np.float32)
    splits = np.array([split_map[e] for e in episode_labels])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, split=splits,
                        episode_id=np.asarray(episode_labels),
                        actor_id=np.asarray(actor_labels))
    summary = {name: int((splits == name).sum()) for name in ("train", "validation", "test")}
    metadata = {
        "output": str(output), "samples": len(samples), "obs_steps": args.obs_steps,
        "pred_steps": args.pred_steps, "sample_stride": args.sample_stride,
        "nominal_sample_rate_hz": 20.0 / args.sample_stride,
        "episodes": len(set(episode_labels)), "split_samples": summary,
        "split_policy": "episode-level; if fewer than three episodes, all samples remain in train"
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
