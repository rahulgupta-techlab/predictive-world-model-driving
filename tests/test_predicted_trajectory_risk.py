import numpy as np
from pwmd.planning import risk as planning


def test_learned_actor_future_is_scored_in_same_frame():
    times, paths = planning.generate_candidate_paths(5.0, horizon_s=1.0, dt=0.1, steering_angles_deg=(0,))
    future = np.column_stack((np.full(len(times), 2.5), np.linspace(2.0, -2.0, len(times))))
    risk = planning.score_candidate_path_from_predictions(paths[0.0], times, [future],
        actor_radii_m=[0.7], ego_radius_m=0.7, safety_margin_m=0.2)
    assert risk.collision_predicted and risk.time_to_collision_s is not None


def test_predicted_future_can_be_ranked_across_candidates():
    times, paths = planning.generate_candidate_paths(4.0, horizon_s=1.5, dt=0.1, steering_angles_deg=(-12, 0, 12))
    future = np.column_stack((4.0 * times, np.zeros_like(times)))
    choice = planning.choose_lowest_risk_from_predictions(paths, times, [future], actor_radii_m=[0.7])
    assert choice["steering_deg"] in paths
    assert choice["risk"].minimum_clearance_m >= min(r.minimum_clearance_m for r in choice["all_candidates"].values())


def test_prediction_shape_mismatch_is_rejected():
    times, paths = planning.generate_candidate_paths(3.0, horizon_s=1.0)
    with np.testing.assert_raises(ValueError):
        planning.score_candidate_path_from_predictions(paths[0.0], times, [np.zeros((2, 2))])
