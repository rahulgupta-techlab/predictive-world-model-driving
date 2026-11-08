import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scene_assembler", ROOT / "src/pwmd/prediction/scene.py")
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)


def test_scene_assembler_emits_actor_centric_windows_and_ego_metadata():
    assembler = scene.SceneAssembler(obs_steps=3, sample_stride=2)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    result = None
    for frame in range(0, 6, 2):
        actor = {"id": 7, "x": float(frame), "y": 2.0, "yaw_deg": 0.0,
                 "speed_mps": 1.0, "radius_m": 0.8}
        result = assembler.update(frame, ego, [actor])
    assert result["actor_ids"] == ["7"]
    assert result["observed_xy"].shape == (1, 3, 2)
    np.testing.assert_allclose(result["actor_positions_ego"], [[4.0, 2.0]])
    np.testing.assert_allclose(result["observed_xy"][0, -1], [0.0, 0.0])


def test_scene_assembler_drops_track_after_sampling_gap():
    assembler = scene.SceneAssembler(obs_steps=3, sample_stride=2)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    actor = {"id": "a", "x": 0.0, "y": 0.0, "yaw_deg": 0.0, "speed_mps": 1.0}
    for frame in (0, 2, 4):
        result = assembler.update(frame, ego, [dict(actor, x=frame)])
    assert result["observed_xy"].shape[0] == 1
    assembler.update(6, ego, [])
    result = assembler.update(8, ego, [dict(actor, x=8)])
    assert result["observed_xy"].shape[0] == 0


def test_scene_assembler_uses_ego_heading_for_relative_position():
    assembler = scene.SceneAssembler(obs_steps=2, sample_stride=1)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 90.0}
    actor = {"id": 1, "x": 0.0, "y": 5.0, "yaw_deg": 90.0, "speed_mps": 1.0}
    assembler.update(0, ego, [actor])
    result = assembler.update(1, ego, [dict(actor, x=0.0, y=6.0)])
    np.testing.assert_allclose(result["actor_positions_ego"], [[6.0, 0.0]], atol=1e-6)
