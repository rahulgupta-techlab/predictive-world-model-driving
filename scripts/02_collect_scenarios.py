import argparse
import csv
import json
import math
import queue
import random
from datetime import datetime
from pathlib import Path

import carla
import numpy as np
import yaml


def speed(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)


def save_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="mixed_traffic")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--config", default="configs/collection.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    client = carla.Client(cfg["host"], int(cfg["port"]))
    client.set_timeout(float(cfg["timeout_seconds"]))
    sv, cv = client.get_server_version(), client.get_client_version()
    if sv != cv:
        raise RuntimeError(f"CARLA version mismatch: server={sv}, client={cv}")

    world = client.get_world()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager(int(cfg["traffic_manager_port"]))
    base_seed = int(cfg["seed"])

    try:
        for ep in range(args.episodes):
            seed = base_seed + ep
            random.seed(seed)
            np.random.seed(seed)
            world = client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = float(cfg["fixed_delta_seconds"])
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(seed)

            spawned = []
            sensor = None
            image_q = queue.Queue()
            ego_rows, actor_rows = [], []
            episode_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{seed}"
            out = Path(cfg["output_dir"]) / args.scenario / episode_id
            (out / "rgb").mkdir(parents=True, exist_ok=True)

            try:
                bp_lib = world.get_blueprint_library()
                spawn_points = world.get_map().get_spawn_points()
                random.shuffle(spawn_points)
                if not spawn_points:
                    raise RuntimeError("No spawn points found on this map.")

                ego_bp = bp_lib.find("vehicle.tesla.model3")
                ego_bp.set_attribute("role_name", "hero")
                ego = world.try_spawn_actor(ego_bp, spawn_points[0])
                if ego is None:
                    raise RuntimeError("Could not spawn ego vehicle.")
                spawned.append(ego)
                ego.set_autopilot(True, int(cfg["traffic_manager_port"]))

                vehicle_bps = bp_lib.filter("vehicle.*")
                bg_count = 0
                for transform in spawn_points[1:]:
                    if bg_count >= int(cfg["vehicle_count"]):
                        break
                    bp = random.choice(vehicle_bps)
                    if bp.has_attribute("color"):
                        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
                    vehicle = world.try_spawn_actor(bp, transform)
                    if vehicle:
                        vehicle.set_autopilot(True, int(cfg["traffic_manager_port"]))
                        spawned.append(vehicle)
                        bg_count += 1

                walker_bps = bp_lib.filter("walker.pedestrian.*")
                controller_bp = bp_lib.find("controller.ai.walker")
                for _ in range(int(cfg["walker_count"])):
                    loc = world.get_random_location_from_navigation()
                    if loc is None:
                        continue
                    walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
                    if walker:
                        spawned.append(walker)
                        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                        spawned.append(controller)
                        controller.start()
                        dest = world.get_random_location_from_navigation()
                        if dest:
                            controller.go_to_location(dest)
                        controller.set_max_speed(random.uniform(0.8, 1.8))

                cam_bp = bp_lib.find("sensor.camera.rgb")
                cam = cfg["camera"]
                cam_bp.set_attribute("image_size_x", str(cam["width"]))
                cam_bp.set_attribute("image_size_y", str(cam["height"]))
                cam_bp.set_attribute("fov", str(cam["fov"]))
                cam_tf = carla.Transform(
                    carla.Location(x=cam["x"], y=cam["y"], z=cam["z"]),
                    carla.Rotation(pitch=cam["pitch"])
                )
                sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
                sensor.listen(image_q.put)
                spawned.append(sensor)

                world.tick()
                total_frames = int(args.seconds / float(cfg["fixed_delta_seconds"]))
                radius = float(cfg["actor_radius_m"])
                stride = int(cfg["camera_stride"])

                for step in range(total_frames):
                    frame = world.tick()
                    snap = world.get_snapshot()
                    ts = snap.timestamp.elapsed_seconds
                    ego_tf, ego_v, ego_a = ego.get_transform(), ego.get_velocity(), ego.get_acceleration()
                    control = ego.get_control()
                    ego_rows.append({
                        "frame": frame, "timestamp_s": ts,
                        "x": ego_tf.location.x, "y": ego_tf.location.y, "z": ego_tf.location.z,
                        "yaw_deg": ego_tf.rotation.yaw,
                        "vx": ego_v.x, "vy": ego_v.y, "vz": ego_v.z,
                        "speed_mps": speed(ego_v),
                        "ax": ego_a.x, "ay": ego_a.y, "az": ego_a.z,
                        "throttle": control.throttle, "steer": control.steer,
                        "brake": control.brake
                    })

                    for actor in world.get_actors():
                        if actor.id == ego.id:
                            continue
                        if not (actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker.pedestrian.")):
                            continue
                        tf = actor.get_transform()
                        dist = tf.location.distance(ego_tf.location)
                        if dist > radius:
                            continue
                        v = actor.get_velocity()
                        actor_rows.append({
                            "frame": frame, "timestamp_s": ts, "actor_id": actor.id,
                            "type_id": actor.type_id, "x": tf.location.x, "y": tf.location.y,
                            "z": tf.location.z, "yaw_deg": tf.rotation.yaw,
                            "vx": v.x, "vy": v.y, "vz": v.z,
                            "speed_mps": speed(v), "distance_to_ego_m": dist
                        })

                    if step % stride == 0:
                        try:
                            image = image_q.get(timeout=2.0)
                            image.save_to_disk(str(out / "rgb" / f"{image.frame:08d}.png"))
                        except queue.Empty:
                            print(f"Warning: no camera image received near frame {frame}")

                    if step % 100 == 0:
                        print(f"Episode {ep+1}/{args.episodes}: {step}/{total_frames} frames")

                save_csv(out / "ego.csv", ego_rows)
                save_csv(out / "actors.csv", actor_rows)
                metadata = {
                    "episode_id": episode_id, "scenario": args.scenario, "seed": seed,
                    "carla_server_version": sv, "map": world.get_map().name,
                    "fixed_delta_seconds": settings.fixed_delta_seconds,
                    "requested_duration_seconds": args.seconds,
                    "ego_vehicle": ego.type_id,
                    "num_ego_frames": len(ego_rows),
                    "num_actor_observations": len(actor_rows),
                    "num_rgb_frames": len(list((out / "rgb").glob("*.png"))),
                    "note": "Ego uses CARLA Traffic Manager autopilot; this is data collection, not a validated driving policy."
                }
                (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print("Saved:", out.resolve())

            finally:
                if sensor:
                    try:
                        sensor.stop()
                    except Exception:
                        pass
                for actor in reversed(spawned):
                    try:
                        if actor.is_alive:
                            actor.destroy()
                    except Exception:
                        pass
                tm.set_synchronous_mode(False)
                world.apply_settings(original_settings)
    finally:
        tm.set_synchronous_mode(False)
        client.get_world().apply_settings(original_settings)

    print("Collection finished.")


if __name__ == "__main__":
    main()
