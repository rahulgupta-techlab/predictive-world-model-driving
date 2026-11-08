from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_collector_sampling_matches_planner_prediction_grid():
    collection = yaml.safe_load((ROOT / "configs/collection.yaml").read_text())
    planner = yaml.safe_load((ROOT / "configs/planner.yaml").read_text())
    sample_dt = collection["fixed_delta_seconds"] * collection["camera_stride"]
    assert abs(sample_dt - planner["time_step_seconds"]) < 1e-9
    assert abs(planner["horizon_seconds"] / planner["time_step_seconds"] - 12) < 1e-9
