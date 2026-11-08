"""Coordinate transforms for actor-centric predictions and ego-relative planning."""
import numpy as np


def actor_local_to_ego_frame(predicted_xy, actor_position_ego, actor_yaw_relative_rad):
    """Transform actor-heading-aligned future XY points into the ego 2-D frame.

    actor_position_ego is the actor's current XY position relative to the ego.
    actor_yaw_relative_rad is actor heading minus ego heading, in radians.
    """
    points = np.asarray(predicted_xy, dtype=np.float64)
    origin = np.asarray(actor_position_ego, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or origin.shape != (2,):
        raise ValueError("Expected predicted_xy [steps, 2] and actor_position_ego [2]")
    if not np.isfinite(points).all() or not np.isfinite(origin).all():
        raise ValueError("Coordinates must be finite")
    c, s = np.cos(actor_yaw_relative_rad), np.sin(actor_yaw_relative_rad)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ rotation.T + origin[None, :]
