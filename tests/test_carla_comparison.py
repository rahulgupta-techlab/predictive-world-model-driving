import json
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts/09_compare_carla_runs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _run(mode, seed=42, map_name="Town03"):
    return {"mode": mode, "seed": seed, "map": map_name, "carla_version": "0.9.15",
            "collision_count": 0, "distance_travelled_m": 100.0,
            "mean_speed_mps": 5.0, "minimum_predicted_clearance_m": 1.0,
            "fallback_interventions": 0, "simulated_frames": 1200}


def test_comparison_keeps_matching_modes_and_metrics():
    rows = module.summarize([_run("autopilot"), _run("constant_velocity"), _run("learned")])
    assert [row["mode"] for row in rows] == ["autopilot", "constant_velocity", "learned"]
    assert rows[0]["collisions"] == 0


def test_comparison_rejects_mixed_seeds():
    with pytest.raises(ValueError, match="same seed"):
        module.summarize([_run("autopilot"), _run("learned", seed=43)])


def test_comparison_rejects_duplicate_modes():
    with pytest.raises(ValueError, match="one summary per mode"):
        module.summarize([_run("autopilot"), _run("autopilot")])
