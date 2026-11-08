"""Rule-based introspection gate for the kinematic candidate planner.

This is an experimental guardrail, not a certified functional-safety mechanism.
Actor predictions are assumed to be valid, synchronized and in a common frame.
"""
from dataclasses import dataclass

from .risk import choose_lowest_risk, choose_lowest_risk_from_predictions


@dataclass(frozen=True)
class PlannerDecision:
    mode: str
    steering_deg: float
    target_speed_mps: float
    emergency_brake: bool
    reason: str
    minimum_clearance_m: float
    time_to_collision_s: float | None


def _decision(selected, current_speed_mps, minimum_clearance_m,
              minimum_ttc_s, fallback_speed_mps):
    risk = selected["risk"]
    unsafe_clearance = risk.minimum_clearance_m < minimum_clearance_m
    unsafe_ttc = risk.time_to_collision_s is not None and risk.time_to_collision_s < minimum_ttc_s
    if risk.collision_predicted or unsafe_clearance or unsafe_ttc:
        return PlannerDecision(
            "fallback_stop", 0.0, fallback_speed_mps, True,
            "No candidate satisfies configured clearance and time-to-collision thresholds",
            risk.minimum_clearance_m, risk.time_to_collision_s)
    return PlannerDecision(
        "predictive_candidate", selected["steering_deg"], current_speed_mps, False,
        "Selected candidate meets configured kinematic risk thresholds",
        risk.minimum_clearance_m, risk.time_to_collision_s)


def decide_action(paths, times, actors, current_speed_mps,
                  ego_radius_m=1.0, safety_margin_m=0.5,
                  minimum_clearance_m=0.5, minimum_ttc_s=1.0,
                  fallback_speed_mps=0.0, preferred_steering_deg=0.0):
    """Select using constant-velocity actor extrapolations."""
    if current_speed_mps < 0 or fallback_speed_mps < 0:
        raise ValueError("Speeds must be non-negative")
    selected = choose_lowest_risk(paths, times, actors, ego_radius_m, safety_margin_m,
                                  preferred_steering_deg)
    return _decision(selected, current_speed_mps, minimum_clearance_m,
                     minimum_ttc_s, fallback_speed_mps)


def decide_action_from_predictions(paths, times, predicted_actor_paths,
                                   current_speed_mps, actor_radii_m=None,
                                   ego_radius_m=1.0, safety_margin_m=0.5,
                                   minimum_clearance_m=0.5, minimum_ttc_s=1.0,
                                   fallback_speed_mps=0.0, preferred_steering_deg=0.0):
    """Select or stop using learned actor futures supplied by an inference model."""
    if current_speed_mps < 0 or fallback_speed_mps < 0:
        raise ValueError("Speeds must be non-negative")
    selected = choose_lowest_risk_from_predictions(
        paths, times, predicted_actor_paths, actor_radii_m,
        ego_radius_m, safety_margin_m, preferred_steering_deg)
    return _decision(selected, current_speed_mps, minimum_clearance_m,
                     minimum_ttc_s, fallback_speed_mps)
