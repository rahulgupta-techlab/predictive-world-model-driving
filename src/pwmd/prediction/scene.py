"""Build synchronized actor-centric model inputs and ego-frame future metadata.

The forecaster is trained on each actor's past XY positions, translated to its
last observation and rotated into that actor's heading frame. This module
reproduces that convention while retaining the transform needed by the planner.
"""
from collections import defaultdict, deque
import numpy as np


def _world_to_heading_local(points, origin, yaw_rad):
    delta = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]))


def _world_vector_to_ego(vector, ego_yaw_rad):
    c, s = np.cos(ego_yaw_rad), np.sin(ego_yaw_rad)
    return np.array([c * vector[0] + s * vector[1],
                     -s * vector[0] + c * vector[1]], dtype=np.float64)


class SceneAssembler:
    """Maintain fixed-rate actor histories and produce one synchronized scene.

    Call update() once per CARLA tick with all current actor states. The
    assembler samples every sample_stride ticks and returns None until each
    actor has a complete observation window.
    """
    def __init__(self, obs_steps=8, sample_stride=5, max_missing_ticks=10,
                 minimum_speed_mps=0.0):
        if obs_steps < 2 or sample_stride < 1 or max_missing_ticks < 1:
            raise ValueError("obs_steps >= 2 and positive stride/missing limit required")
        self.obs_steps = int(obs_steps)
        self.sample_stride = int(sample_stride)
        self.max_missing_ticks = int(max_missing_ticks)
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.histories = defaultdict(lambda: deque(maxlen=self.obs_steps))
        self.last_seen = {}

    def update(self, frame, ego_state, actor_states):
        """Return model inputs and per-actor ego-frame metadata, or None.

        ego_state: {x, y, yaw_deg}; each actor: {id, x, y, yaw_deg,
        speed_mps, radius_m, type_id}. Actor states must be from the same frame.
        """
        frame = int(frame)
        if frame % self.sample_stride:
            self._expire(frame)
            return None
        ego_yaw = np.deg2rad(float(ego_state["yaw_deg"]))
        ego_origin = np.array([ego_state["x"], ego_state["y"]], dtype=np.float64)
        seen = set()
        for actor in actor_states:
            actor_id = str(actor["id"])
            seen.add(actor_id)
            history = self.histories[actor_id]
            previous = self.last_seen.get(actor_id)
            if previous is not None and frame - previous != self.sample_stride:
                history.clear()
            history.append({
                "frame": frame, "x": float(actor["x"]), "y": float(actor["y"]),
                "yaw_deg": float(actor["yaw_deg"]),
                "speed_mps": float(actor.get("speed_mps", 0.0)),
                "radius_m": float(actor.get("radius_m", 1.0)),
                "type_id": str(actor.get("type_id", "unknown"))
            })
            self.last_seen[actor_id] = frame
        self._expire(frame)
        ids, windows, positions, headings, radii = [], [], [], [], []
        for actor_id in sorted(seen):
            history = self.histories[actor_id]
            if len(history) != self.obs_steps:
                continue
            rows = list(history)
            if any(rows[i]["frame"] - rows[i-1]["frame"] != self.sample_stride
                   for i in range(1, len(rows))):
                continue
            if rows[-1]["speed_mps"] < self.minimum_speed_mps:
                continue
            world_xy = np.array([[r["x"], r["y"]] for r in rows], dtype=np.float64)
            anchor = world_xy[-1]
            actor_yaw = np.deg2rad(rows[-1]["yaw_deg"])
            local = _world_to_heading_local(world_xy, anchor, actor_yaw)
            actor_delta_world = anchor - ego_origin
            actor_position_ego = _world_vector_to_ego(actor_delta_world, ego_yaw)
            ids.append(actor_id)
            windows.append(local.astype(np.float32))
            positions.append(actor_position_ego)
            headings.append(float(actor_yaw - ego_yaw))
            radii.append(float(rows[-1]["radius_m"]))
        if not windows:
            return {"frame": frame, "actor_ids": [], "observed_xy": np.empty((0, self.obs_steps, 2), np.float32),
                    "actor_positions_ego": np.empty((0, 2), np.float64),
                    "actor_yaw_relative_rad": np.empty((0,), np.float64),
                    "actor_radii_m": [], "visible_actor_count": len(seen),
                    "untracked_actor_count": len(seen)}
        return {
            "frame": frame, "actor_ids": ids,
            "observed_xy": np.stack(windows),
            "actor_positions_ego": np.stack(positions),
            "actor_yaw_relative_rad": np.asarray(headings, dtype=np.float64),
            "actor_radii_m": radii,
            "visible_actor_count": len(seen),
            "untracked_actor_count": len(seen) - len(ids)
        }

    def _expire(self, frame):
        stale = [actor_id for actor_id, last in self.last_seen.items()
                 if frame - last > self.max_missing_ticks]
        for actor_id in stale:
            self.histories.pop(actor_id, None)
            self.last_seen.pop(actor_id, None)
