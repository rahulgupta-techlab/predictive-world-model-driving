import numpy as np
from pwmd.planning import risk as planning


def test_candidate_paths_are_finite_and_include_straight_path():
    times, paths = planning.generate_candidate_paths(8.0, horizon_s=2.0, dt=0.1)
    assert 0.0 in paths and paths[0.0].shape == (len(times), 2)
    assert np.isfinite(paths[0.0]).all() and paths[0.0][-1, 0] > 0


def test_collision_risk_detects_stationary_actor_in_ego_path():
    times, paths = planning.generate_candidate_paths(5.0, horizon_s=2.0, dt=0.1, steering_angles_deg=(0,))
    actor = planning.ActorState("blocking", (2.5, 0.0), (0.0, 0.0), radius_m=0.8)
    result = planning.score_candidate_path(paths[0.0], times, [actor], ego_radius_m=0.8, safety_margin_m=0.2)
    assert result.collision_predicted and result.time_to_collision_s is not None
    assert result.minimum_clearance_m < 0


def test_empty_actor_list_has_no_finite_risk():
    times, paths = planning.generate_candidate_paths(3.0, horizon_s=1.0)
    result = planning.score_candidate_path(paths[0.0], times, [])
    assert np.isinf(result.minimum_clearance_m) and not result.collision_predicted


def test_risk_selector_returns_one_of_supplied_candidates():
    times, paths = planning.generate_candidate_paths(4.0, horizon_s=1.5)
    actor = planning.ActorState("nearby", (3.0, 0.0), (0.0, 0.0), radius_m=0.6)
    assert planning.choose_lowest_risk(paths, times, [actor])["steering_deg"] in paths


def test_equal_risk_candidates_follow_lane_reference_preference():
    times, paths = planning.generate_candidate_paths(
        4.0, horizon_s=1.0, dt=0.1, steering_angles_deg=(-5, 0, 5, 10))
    result = planning.choose_lowest_risk(
        paths, times, [], preferred_steering_deg=5.0)
    assert result["steering_deg"] == 5.0
