import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("planning_risk", ROOT / "src/pwmd/planning/risk.py")
planning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planning)


def test_learned_actor_future_is_scored_in_same_frame():
    times, paths = planning.generate_candidate_paths(
        5.0, horizon_s=1.0, dt=0.1, steering_angles_deg=(0,))
    # Predicted actor crosses the straight candidate at t=0.5 s.
    actor_future = np.column_stack((np.full(len(times), 2.5),
                                   np.linspace(2.0, -2.0, len(times))))
    risk = planning.score_candidate_path_from_predictions(
        paths[0.0], times, [actor_future], actor_radii_m=[0.7],
        ego_radius_m=0.7, safety_margin_m=0.2)
    assert risk.collision_predicted
    assert risk.time_to_collision_s is not None


def test_predicted_future_can_change_candidate_ranking():
    times, paths = planning.generate_candidate_paths(
        4.0, horizon_s=1.5, dt=0.1, steering_angles_deg=(-12, 0, 12))
    # Occupy the centreline while leaving a lateral candidate comparatively clear.
    actor_future = np.column_stack((4.0 * times, np.zeros_like(times)))
    choice = planning.choose_lowest_risk_from_predictions(
        paths, times, [actor_future], actor_radii_m=[0.7])
    assert choice["steering_deg"] in paths
    assert choice["risk"].minimum_clearance_m >= min(
        risk.minimum_clearance_m for risk in choice["all_candidates"].values())


def test_prediction_shape_mismatch_is_rejected():
    times, paths = planning.generate_candidate_paths(3.0, horizon_s=1.0)
    with np.testing.assert_raises(ValueError):
        planning.score_candidate_path_from_predictions(
            paths[0.0], times, [np.zeros((2, 2))])
