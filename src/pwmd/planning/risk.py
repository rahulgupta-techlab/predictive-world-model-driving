"""Kinematic candidate paths and risk scoring against actor predictions.

Coordinates share one 2-D frame and time is in seconds. This transparent
baseline omits road boundaries, traffic rules, vehicle dynamics and uncertainty.
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
    times = np.arange(1, int(np.floor(horizon_s / dt)) + 1, dtype=np.float64) * dt
    paths = {}
    for angle in steering_angles_deg:
        yaw_rate = speed_mps * np.tan(np.deg2rad(float(angle))) / wheelbase_m
        if abs(yaw_rate) < 1e-9:
            x, y = speed_mps * times, np.zeros_like(times)
        else:
            yaw = yaw_rate * times
            x = speed_mps / yaw_rate * np.sin(yaw)
            y = speed_mps / yaw_rate * (1.0 - np.cos(yaw))
        paths[float(angle)] = np.column_stack((x, y))
    return times, paths


def constant_velocity_actor(actor, times):
    p = np.asarray(actor.position, dtype=np.float64)
    v = np.asarray(actor.velocity, dtype=np.float64)
    return p[None, :] + np.asarray(times)[:, None] * v[None, :]


def _score_against_predictions(ego_path, times, predictions, radii,
                               ego_radius_m, safety_margin_m, collision_threshold_m):
    ego_path, times = np.asarray(ego_path, dtype=np.float64), np.asarray(times, dtype=np.float64)
    if ego_path.shape != (len(times), 2):
        raise ValueError("ego_path must have shape [len(times), 2]")
    if ego_radius_m < 0 or safety_margin_m < 0 or collision_threshold_m < 0:
        raise ValueError("radii, margins and collision threshold must be non-negative")
    if len(predictions) != len(radii):
        raise ValueError("One radius is required for each actor prediction")
    min_sep, min_clearance, min_time, ttc = float("inf"), float("inf"), 0.0, None
    for predicted, radius in zip(predictions, radii):
        predicted = np.asarray(predicted, dtype=np.float64)
        if predicted.shape != ego_path.shape or not np.isfinite(predicted).all():
            raise ValueError("Each actor prediction must be finite and match ego_path shape")
        if radius < 0:
            raise ValueError("Actor radii must be non-negative")
        separation = np.linalg.norm(ego_path - predicted, axis=1)
        clearance = separation - (ego_radius_m + float(radius) + safety_margin_m)
        idx = int(np.argmin(separation))
        if separation[idx] < min_sep:
            min_sep, min_time = float(separation[idx]), float(times[idx])
        min_clearance = min(min_clearance, float(clearance.min()))
        hits = np.flatnonzero(clearance <= collision_threshold_m)
        if hits.size:
            hit_time = float(times[int(hits[0])])
            ttc = hit_time if ttc is None else min(ttc, hit_time)
    if not predictions:
        min_sep = min_clearance = float("inf")
    return RiskResult(min_sep, min_clearance, min_time,
                      bool(min_clearance <= collision_threshold_m), ttc)


def score_candidate_path(ego_path, times, actors, ego_radius_m=1.0,
                         safety_margin_m=0.5, collision_threshold_m=0.0):
    """Score against constant-velocity actor extrapolations."""
    predictions = [constant_velocity_actor(actor, times) for actor in actors]
    radii = [actor.radius_m for actor in actors]
    return _score_against_predictions(ego_path, times, predictions, radii,
                                      ego_radius_m, safety_margin_m, collision_threshold_m)


def score_candidate_path_from_predictions(ego_path, times, predicted_actor_paths,
                                          actor_radii_m=None, ego_radius_m=1.0,
                                          safety_margin_m=0.5,
                                          collision_threshold_m=0.0):
    """Score against supplied [future_steps, 2] actor futures in the same frame."""
    predictions = list(predicted_actor_paths)
    radii = list(actor_radii_m) if actor_radii_m is not None else [1.0] * len(predictions)
    return _score_against_predictions(ego_path, times, predictions, radii,
                                      ego_radius_m, safety_margin_m, collision_threshold_m)


def _choose(paths, scorer):
    scored = []
    for steering, path in paths.items():
        risk = scorer(path)
        scored.append((risk.collision_predicted, -risk.minimum_clearance_m,
                       abs(float(steering)), float(steering), risk))
    if not scored:
        raise ValueError("At least one candidate path is required")
    scored.sort(key=lambda item: item[:3])
    best = scored[0]
    return {"steering_deg": best[3], "risk": best[4],
            "all_candidates": {item[3]: item[4] for item in scored}}


def choose_lowest_risk(paths, times, actors, ego_radius_m=1.0, safety_margin_m=0.5):
    return _choose(paths, lambda path: score_candidate_path(
        path, times, actors, ego_radius_m, safety_margin_m))


def choose_lowest_risk_from_predictions(paths, times, predicted_actor_paths,
                                        actor_radii_m=None, ego_radius_m=1.0,
                                        safety_margin_m=0.5):
    return _choose(paths, lambda path: score_candidate_path_from_predictions(
        path, times, predicted_actor_paths, actor_radii_m, ego_radius_m, safety_margin_m))
