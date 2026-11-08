import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prediction_coordinates", ROOT / "src/pwmd/prediction/coordinates.py")
coordinates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinates)


def test_actor_local_future_is_rotated_and_translated_to_ego_frame():
    local = np.array([[1.0, 0.0], [2.0, 0.0]])
    result = coordinates.actor_local_to_ego_frame(
        local, actor_position_ego=(10.0, 3.0), actor_yaw_relative_rad=np.pi / 2)
    np.testing.assert_allclose(result, [[10.0, 4.0], [10.0, 5.0]], atol=1e-7)


def test_coordinate_transform_rejects_wrong_shapes():
    with np.testing.assert_raises(ValueError):
        coordinates.actor_local_to_ego_frame(np.zeros((3, 3)), (0.0, 0.0), 0.0)
