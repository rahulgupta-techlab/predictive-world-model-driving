"""CARLA closed-loop experiment harness for autopilot, CV and learned prediction.

Run one mode per invocation with the same seed/map/scenario settings for fair
comparisons. This is an experimental research controller, not a safety system.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pwmd.prediction.scene import SceneAssembler
from pwmd.prediction.forecasters import load_forecaster, predict
from pwmd.prediction.coordinates import actor_local_to_ego_frame
from pwmd.planning.risk import ActorState, generate_candidate_paths
from pwmd.planning.safety import decide_action, decide_action_from_predictions


def _speed(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def _world_vector_to_ego(vx, vy, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([c * vx + s * vy, -s * vx + c * vy], dtype=np.float64)


def _control_for_decision(carla, decision, speed_mps, previous_steer,
                          cruise_speed_mps, max_steer_deg=30.0):
    if decision.emergency_brake:
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0), 0.0
    target = min(cruise_speed_mps, decision.target_speed_mps)
    error = target - speed_mps
    throttle = float(np.clip(0.35 * error, 0.0, 0.55))
    brake = float(np.clip(-0.45 * error, 0.0, 1.0))
    steer = float(np.clip(decision.steering_deg / max_steer_deg, -1.0, 1.0))
    steer = 0.65 * previous_steer + 0.35 * steer
    return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer), steer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["autopilot", "constant_velocity", "learned"], required=True)
    parser.add_argument("--checkpoint", help="Required for --mode learned")
    parser.add_argument("--config", default="configs/collection.yaml")
    parser.add_argument("--planner-config", default="configs/planner.yaml")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--cruise-speed", type=float, default=6.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.mode == "learned" and not args.checkpoint:
        parser.error("--checkpoint is required for learned mode")
    if args.seconds <= 0 or args.vehicles < 0:
        parser.error("--seconds must be positive and --vehicles non-negative")

    try:
        import carla
    except ImportError as exc:
        raise SystemExit("CARLA Python API is required. Install the wheel matching your CARLA server.") from exc

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    planner_cfg = yaml.safe_load(Path(args.planner_config).read_text(encoding="utf-8"))
    fixed_dt = float(cfg["fixed_delta_seconds"])
    sample_stride = int(cfg["camera_stride"])
    sample_dt = fixed_dt * sample_stride
    horizon = float(planner_cfg["horizon_seconds"])
    planner_dt = float(planner_cfg["time_step_seconds"])
    times = np.arange(1, int(round(horizon / planner_dt)) + 1) * planner_dt
    if args.mode == "learned":
        model, device, metadata = load_forecaster(args.checkpoint)
        if metadata["obs_steps"] != 8:
            raise SystemExit("Current scene assembler expects 8 observed samples; retrain/configure matching checkpoint.")
        if metadata["pred_steps"] != len(times):
            raise SystemExit(f"Checkpoint predicts {metadata['pred_steps']} steps but planner requires {len(times)}.")
        if not math.isclose(sample_dt, planner_dt, rel_tol=0.0, abs_tol=1e-6):
            raise SystemExit("Learned model sampling interval must equal planner time_step_seconds.")
    else:
        model = device = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    client = carla.Client(cfg["host"], int(cfg["port"]))
    client.set_timeout(float(cfg["timeout_seconds"]))
    server_version, client_version = client.get_server_version(), client.get_client_version()
    if server_version != client_version:
        raise SystemExit(f"CARLA version mismatch: server={server_version}, client={client_version}")
    world = client.get_world()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager(int(cfg["traffic_manager_port"]))
    spawned, collision_events = [], []
    control_rows = []
    ego = None
    assembler = SceneAssembler(obs_steps=8, sample_stride=sample_stride,
                               max_missing_ticks=sample_stride * 2)
    previous_steer = 0.0
    last_control = None
    interventions = 0
    min_clearance = float("inf")
    distance_m = 0.0
    previous_location = None
    speed_sum = 0.0
    frames = 0
    output = Path(args.output) if args.output else Path("experiments/results") / f"carla_{args.mode}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_dt
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(args.seed)
        world.tick()
        spawn_points = world.get_map().get_spawn_points()
        rng = random.Random(args.seed)
        rng.shuffle(spawn_points)
        if not spawn_points:
            raise RuntimeError("No map spawn points available.")
        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(ego_bp, spawn_points[0])
        if ego is None:
            raise RuntimeError("Could not spawn ego vehicle.")
        spawned.append(ego)
        if args.mode == "autopilot":
            ego.set_autopilot(True, int(cfg["traffic_manager_port"]))
        else:
            ego.set_autopilot(False)

        vehicle_bps = bp_lib.filter("vehicle.*")
        npc_count = 0
        for transform in spawn_points[1:]:
            if npc_count >= args.vehicles:
                break
            bp = rng.choice(vehicle_bps)
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle:
                vehicle.set_autopilot(True, int(cfg["traffic_manager_port"]))
                spawned.append(vehicle)
                npc_count += 1

        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(lambda event: collision_events.append({
            "frame": int(event.frame), "other_actor": event.other_actor.type_id
        }))
        spawned.append(collision_sensor)

        total_frames = int(args.seconds / fixed_dt)
        print(f"Starting mode={args.mode}, seed={args.seed}, frames={total_frames}, NPCs={npc_count}")
        for step in range(total_frames):
            frame = world.tick()
            ego_tf = ego.get_transform()
            ego_v = ego.get_velocity()
            ego_speed = _speed(ego_v)
            ego_yaw = math.radians(ego_tf.rotation.yaw)
            ego_loc = ego_tf.location
            if previous_location is not None:
                distance_m += ego_loc.distance(previous_location)
            previous_location = ego_loc
            speed_sum += ego_speed
            frames += 1

            if args.mode != "autopilot":
                actor_states = []
                for actor in world.get_actors():
                    if actor.id == ego.id or not (
                        actor.type_id.startswith("vehicle.") or
                        actor.type_id.startswith("walker.pedestrian.")
                    ):
                        continue
                    tf = actor.get_transform()
                    if tf.location.distance(ego_loc) > float(cfg["actor_radius_m"]):
                        continue
                    velocity = actor.get_velocity()
                    radius = 0.45 if actor.type_id.startswith("walker.") else 1.0
                    actor_states.append({
                        "id": actor.id, "x": tf.location.x, "y": tf.location.y,
                        "yaw_deg": tf.rotation.yaw, "speed_mps": _speed(velocity),
                        "radius_m": radius, "type_id": actor.type_id
                    })
                ego_state = {"x": ego_loc.x, "y": ego_loc.y, "yaw_deg": ego_tf.rotation.yaw}
                scene = assembler.update(frame, ego_state, actor_states)
                if scene is not None:
                    candidate_times, paths = generate_candidate_paths(
                        ego_speed, horizon_s=horizon, dt=planner_dt,
                        steering_angles_deg=planner_cfg["steering_candidates_deg"],
                        wheelbase_m=float(planner_cfg["wheelbase_m"]))
                    if args.mode == "constant_velocity":
                        actor_models = []
                        for actor in actor_states:
                            pos = _world_vector_to_ego(actor["x"] - ego_loc.x,
                                                       actor["y"] - ego_loc.y, ego_yaw)
                            vel = _world_vector_to_ego(
                                world.get_actor(int(actor["id"])).get_velocity().x - ego_v.x,
                                world.get_actor(int(actor["id"])).get_velocity().y - ego_v.y,
                                ego_yaw)
                            actor_models.append(ActorState(str(actor["id"]), tuple(pos),
                                                            tuple(vel), actor["radius_m"]))
                        decision = decide_action(
                            paths, candidate_times, actor_models, ego_speed,
                            ego_radius_m=float(planner_cfg["ego_radius_m"]),
                            safety_margin_m=float(planner_cfg["safety_margin_m"]))
                    elif scene["actor_ids"]:
                        local_future = predict(model, device, scene["observed_xy"])
                        ego_future = [
                            actor_local_to_ego_frame(
                                local_future[i], scene["actor_positions_ego"][i],
                                scene["actor_yaw_relative_rad"][i])
                            for i in range(len(scene["actor_ids"]))
                        ]
                        decision = decide_action_from_predictions(
                            paths, candidate_times, ego_future, ego_speed,
                            actor_radii_m=scene["actor_radii_m"],
                            ego_radius_m=float(planner_cfg["ego_radius_m"]),
                            safety_margin_m=float(planner_cfg["safety_margin_m"]))
                    else:
                        # No track has a full observation window yet; hold speed
                        # with a straight candidate, but do not infer free space.
                        from pwmd.planning.safety import PlannerDecision
                        decision = PlannerDecision("warmup", 0.0, min(ego_speed, args.cruise_speed),
                                                    False, "Waiting for synchronized actor histories",
                                                    float("inf"), None)
                    min_clearance = min(min_clearance, decision.minimum_clearance_m)
                    if decision.emergency_brake:
                        interventions += 1
                    last_control, previous_steer = _control_for_decision(
                        carla, decision, ego_speed, previous_steer, args.cruise_speed)
                    ego.apply_control(last_control)
                    control_rows.append({
                        "frame": frame, "mode": decision.mode,
                        "steering_deg": decision.steering_deg,
                        "target_speed_mps": decision.target_speed_mps,
                        "speed_mps": ego_speed, "minimum_clearance_m": decision.minimum_clearance_m,
                        "ttc_s": decision.time_to_collision_s,
                        "emergency_brake": decision.emergency_brake
                    })
                elif last_control is not None:
                    ego.apply_control(last_control)

            if step % 200 == 0:
                print(f"{args.mode}: {step}/{total_frames} frames; speed={ego_speed:.1f} m/s; collisions={len(collision_events)}")

        summary = {
            "mode": args.mode, "seed": args.seed, "map": world.get_map().name,
            "carla_version": server_version, "requested_seconds": args.seconds,
            "simulated_frames": frames, "npc_vehicles_spawned": npc_count,
            "collision_count": len(collision_events), "collisions": collision_events,
            "distance_travelled_m": distance_m,
            "mean_speed_mps": speed_sum / max(frames, 1),
            "minimum_predicted_clearance_m": None if not math.isfinite(min_clearance) else min_clearance,
            "fallback_interventions": interventions,
            "controller_frames": len(control_rows),
            "checkpoint": args.checkpoint, "device": str(device) if device is not None else None,
            "note": "Experimental simulator result only; compare repeated seeds and route progress. Not a safety validation."
        }
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        csv_path = output.with_suffix(".controls.csv")
        if control_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(control_rows[0]))
                writer.writeheader()
                writer.writerows(control_rows)
        print(json.dumps(summary, indent=2))
        print(f"Saved summary: {output.resolve()}")
    finally:
        for actor in reversed(spawned):
            try:
                if actor.is_alive:
                    if actor.type_id.startswith("sensor."):
                        actor.stop()
                    actor.destroy()
            except Exception:
                pass
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
