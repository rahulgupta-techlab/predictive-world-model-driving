import numpy as np

from scripts.03_prepare_trajectory_dataset import rotate_to_local
from scripts.04_evaluate_constant_velocity import metrics


def test_rotate_to_local_keeps_heading_aligned_x_axis():
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    local = rotate_to_local(points, np.array([0.0, 0.0]), 0.0)
    np.testing.assert_allclose(local, points, atol=1e-6)


def test_rotate_to_local_rotates_world_x_into_local_negative_y_for_north_heading():
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    local = rotate_to_local(points, np.array([0.0, 0.0]), np.pi / 2)
    np.testing.assert_allclose(local[1], [0.0, -1.0], atol=1e-6)


def test_metrics_perfect_prediction():
    target = np.zeros((2, 3, 2), dtype=np.float32)
    result = metrics(target.copy(), target)
    assert result["ADE_m"] == 0.0
    assert result["FDE_m"] == 0.0
    assert result["miss_rate"] == 0.0
    assert result["samples"] == 2


def test_metrics_detects_final_displacement_miss():
    target = np.zeros((1, 2, 2), dtype=np.float32)
    pred = np.array([[[0.0, 0.0], [3.0, 0.0]]], dtype=np.float32)
    result = metrics(pred, target, miss_threshold=2.0)
    assert result["FDE_m"] == 3.0
    assert result["miss_rate"] == 1.0
