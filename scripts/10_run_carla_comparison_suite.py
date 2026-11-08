"""Run reproducible autopilot/CV/learned CARLA comparisons over multiple seeds."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_suite(seeds, seconds, vehicles, checkpoint, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        summaries = []
        for mode in ("autopilot", "constant_velocity", "learned"):
            summary = output_dir / f"carla_{mode}_seed{seed}.json"
            command = [
                sys.executable, str(ROOT / "scripts/08_run_carla_experiment.py"),
                "--mode", mode, "--seed", str(seed), "--seconds", str(seconds),
                "--vehicles", str(vehicles), "--output", str(summary)
            ]
            if mode == "learned":
                command.extend(["--checkpoint", str(checkpoint)])
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            summaries.append(summary)
        comparison = output_dir / f"comparison_seed{seed}.csv"
        command = [
            sys.executable, str(ROOT / "scripts/09_compare_carla_runs.py"),
            *[str(path) for path in summaries], "--output", str(comparison)
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Completed {len(seeds)} seed(s). Results: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--output-dir", default="experiments/results/carla_suite")
    args = parser.parse_args()
    if args.seconds <= 0 or args.vehicles < 0:
        parser.error("--seconds must be positive and --vehicles non-negative")
    if not args.seeds:
        parser.error("At least one seed is required")
    run_suite(args.seeds, args.seconds, args.vehicles, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
