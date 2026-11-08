import numpy as np
from pwmd.prediction.scene import SceneAssembler


def test_scene_assembler_emits_actor_centric_windows_and_ego_metadata():
    assembler = SceneAssembler(obs_steps=3, sample_stride=2)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    for frame in range(0, 6, 2):
        actor = {"id": 7, "x": float(frame), "y": 2.0, "yaw_deg": 0.0,
                 "speed_mps": 1.0, "radius_m": 0.8}
        result = assembler.update(frame, ego, [actor])
    assert result["actor_ids"] == ["7"] and result["observed_xy"].shape == (1, 3, 2)
    assert result["visible_actor_count"] == 1 and result["untracked_actor_count"] == 0
    np.testing.assert_allclose(result["actor_positions_ego"], [[4.0, 2.0]])
    np.testing.assert_allclose(result["observed_xy"][0, -1], [0.0, 0.0])


def test_scene_assembler_drops_track_after_sampling_gap():
    assembler = SceneAssembler(obs_steps=3, sample_stride=2)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    actor = {"id": "a", "x": 0.0, "y": 0.0, "yaw_deg": 0.0, "speed_mps": 1.0}
    for frame in (0, 2, 4):
        result = assembler.update(frame, ego, [dict(actor, x=frame)])
    assert result["observed_xy"].shape[0] == 1
    assembler.update(6, ego, [])
    result = assembler.update(8, ego, [dict(actor, x=8)])
    assert result["observed_xy"].shape[0] == 0


def test_scene_assembler_uses_ego_heading_for_relative_position():
    assembler = SceneAssembler(obs_steps=2, sample_stride=1)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 90.0}
    actor = {"id": 1, "x": 0.0, "y": 5.0, "yaw_deg": 90.0, "speed_mps": 1.0}
    assembler.update(0, ego, [actor])
    result = assembler.update(1, ego, [dict(actor, x=0.0, y=6.0)])
    np.testing.assert_allclose(result["actor_positions_ego"], [[6.0, 0.0]], atol=1e-6)


def test_scene_marks_newly_visible_actor_as_untracked_until_history_is_ready():
    assembler = SceneAssembler(obs_steps=3, sample_stride=1)
    ego = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    actor = {"id": "new", "x": 5.0, "y": 0.0, "yaw_deg": 0.0, "speed_mps": 1.0}
    result = assembler.update(0, ego, [actor])
    assert result["visible_actor_count"] == 1
    assert result["untracked_actor_count"] == 1
    for frame in (1, 2):
        result = assembler.update(frame, ego, [dict(actor, x=5.0 + frame)])
    assert result["untracked_actor_count"] == 0
