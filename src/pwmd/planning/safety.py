"""Rule-based introspection gate and conservative fallback for the kinematic planner.

This is an experimental guardrail around the candidate scorer, not a certified
functional-safety mechanism. It assumes actor states and ego-relative geometry
are valid and current.
"""
from dataclasses import dataclass

from .risk import choose_lowest_risk


@dataclass(frozen=True)
class PlannerDecision:
    mode: str
    steering_deg: float
    target_speed_mps: float
    emergency_brake: bool
    reason: str
    minimum_clearance_m: float
    time_to_collision_s: float | None


def decide_action(paths, times, actors, current_speed_mps,
                  ego_radius_m=1.0, safety_margin_m=0.5,
                  minimum_clearance_m=0.5, minimum_ttc_s=1.0,
                  fallback_speed_mps=0.0):
    """Select a low-risk candidate or request a conservative stop fallback."""
    if current_speed_mps < 0 or fallback_speed_mps < 0:
        raise ValueError("Speeds must be non-negative")
    selected = choose_lowest_risk(paths, times, actors, ego_radius_m, safety_margin_m)
    risk = selected["risk"]
    unsafe_clearance = risk.minimum_clearance_m < minimum_clearance_m
    unsafe_ttc = risk.time_to_collision_s is not None and risk.time_to_collision_s < minimum_ttc_s
    if risk.collision_predicted or unsafe_clearance or unsafe_ttc:
        return PlannerDecision(
            mode="fallback_stop",
            steering_deg=0.0,
            target_speed_mps=fallback_speed_mps,
            emergency_brake=True,
            reason="No candidate satisfies the configured clearance and time-to-collision thresholds",
            minimum_clearance_m=risk.minimum_clearance_m,
            time_to_collision_s=risk.time_to_collision_s
        )
    return PlannerDecision(
        mode="predictive_candidate",
        steering_deg=selected["steering_deg"],
        target_speed_mps=current_speed_mps,
        emergency_brake=False,
        reason="Selected candidate meets the configured kinematic risk thresholds",
        minimum_clearance_m=risk.minimum_clearance_m,
        time_to_collision_s=risk.time_to_collision_s
    )
