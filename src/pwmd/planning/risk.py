"""Simple kinematic candidate-path generation and collision-risk scoring.

All positions are in a common 2-D Cartesian frame and time is in seconds.
This deliberately transparent baseline does not model vehicle dynamics,
road boundaries, traffic rules, or uncertainty.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActorState:
    actor_id: str
    position: tuple
    velocity: tuple
    radius_m: float = 1.0


@dataclass(frozen=True)
class RiskResult:
    minimum_separation_m: float
    minimum_clearance_m: float
    time_of_minimum_separation_s: float
    collision_predicted: bool
    time_to_collision_s: float | None


def generate_candidate_paths(speed_mps, horizon_s=3.0, dt=0.1,
                             steering_angles_deg=(-12, -6, 0, 6, 12),
                             wheelbase_m=2.8):
    """Generate constant-steering bicycle-model paths in ego-local coordinates."""
    if speed_mps < 0 or horizon_s <= 0 or dt <= 0 or wheelbase_m <= 0:
        raise ValueError("speed must be non-negative; horizon, dt and wheelbase must be positive")
    steps = int(np.floor(horizon_s / dt))
    times = np.arange(1, steps + 1, dtype=np.float64) * dt
    paths = {}
    for angle in steering_angles_deg:
        delta = np.deg2rad(float(angle))
        yaw_rate = speed_mps * np.tan(delta) / wheelbase_m
        if abs(yaw_rate) < 1e-9:
            x = speed_mps * times
            y = np.zeros_like(times)
        else:
            yaw = yaw_rate * times
            x = speed_mps / yaw_rate * np.sin(yaw)
            y = speed_mps / yaw_rate * (1.0 - np.cos(yaw))
        paths[float(angle)] = np.column_stack((x, y))
    return times, paths


def constant_velocity_actor(actor, times):
    p = np.asarray(actor.position, dtype=np.float64)
    v = np.asarray(actor.velocity, dtype=np.float64)
    return p[None, :] + times[:, None] * v[None, :]


def score_candidate_path(ego_path, times, actors, ego_radius_m=1.0,
                         safety_margin_m=0.5, collision_threshold_m=0.0):
    """Score a candidate path against constant-velocity actor extrapolations."""
    ego_path = np.asarray(ego_path, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if ego_path.shape != (len(times), 2):
        raise ValueError("ego_path must have shape [len(times), 2]")
    if ego_radius_m < 0 or safety_margin_m < 0:
        raise ValueError("radii and safety margin must be non-negative")
    min_sep, min_clearance, min_time = float("inf"), float("inf"), 0.0
    ttc = None
    for actor in actors:
        predicted = constant_velocity_actor(actor, times)
        separation = np.linalg.norm(ego_path - predicted, axis=1)
        clearance = separation - (ego_radius_m + actor.radius_m + safety_margin_m)
        idx = int(np.argmin(separation))
        if separation[idx] < min_sep:
            min_sep, min_time = float(separation[idx]), float(times[idx])
        min_clearance = min(min_clearance, float(clearance.min()))
        collision_indices = np.flatnonzero(clearance <= collision_threshold_m)
        if collision_indices.size:
            candidate_ttc = float(times[int(collision_indices[0])])
            ttc = candidate_ttc if ttc is None else min(ttc, candidate_ttc)
    if not actors:
        min_sep, min_clearance = float("inf"), float("inf")
    return RiskResult(
        minimum_separation_m=min_sep,
        minimum_clearance_m=min_clearance,
        time_of_minimum_separation_s=min_time,
        collision_predicted=bool(min_clearance <= collision_threshold_m),
        time_to_collision_s=ttc
    )


def choose_lowest_risk(paths, times, actors, ego_radius_m=1.0,
                       safety_margin_m=0.5):
    """Return the lowest-risk candidate; ties prefer smaller steering magnitude."""
    scored = []
    for steering, path in paths.items():
        risk = score_candidate_path(path, times, actors, ego_radius_m, safety_margin_m)
        # Unsafe candidates rank behind any candidate with positive clearance.
        scored.append((risk.collision_predicted, -risk.minimum_clearance_m,
                       abs(float(steering)), float(steering), risk))
    if not scored:
        raise ValueError("At least one candidate path is required")
    scored.sort(key=lambda item: item[:3])
    best = scored[0]
    return {"steering_deg": best[3], "risk": best[4],
            "all_candidates": {item[3]: item[4] for item in scored}}
