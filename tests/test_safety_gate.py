import numpy as np
from pwmd.planning.risk import ActorState, generate_candidate_paths
from pwmd.planning.safety import decide_action, decide_action_from_predictions


def test_safety_gate_uses_stop_fallback_for_blocked_path():
    times, paths = generate_candidate_paths(5.0, horizon_s=2.0, dt=0.1, steering_angles_deg=(0,))
    actor = ActorState("blocker", (2.5, 0.0), (0.0, 0.0), radius_m=0.8)
    decision = decide_action(paths, times, [actor], current_speed_mps=5.0)
    assert decision.mode == "fallback_stop" and decision.emergency_brake
    assert decision.target_speed_mps == 0.0


def test_safety_gate_keeps_clear_candidate():
    times, paths = generate_candidate_paths(4.0, horizon_s=2.0)
    decision = decide_action(paths, times, [], current_speed_mps=4.0)
    assert decision.mode == "predictive_candidate" and not decision.emergency_brake


def test_learned_prediction_uses_same_stop_fallback_gate():
    times, paths = generate_candidate_paths(5.0, horizon_s=1.0, dt=0.1, steering_angles_deg=(0,))
    future = np.column_stack((np.full(len(times), 2.5), np.zeros(len(times))))
    decision = decide_action_from_predictions(paths, times, [future], current_speed_mps=5.0,
                                               actor_radii_m=[0.8])
    assert decision.mode == "fallback_stop" and decision.emergency_brake
