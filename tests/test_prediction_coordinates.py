import numpy as np
from pwmd.prediction.coordinates import actor_local_to_ego_frame


def test_actor_local_future_is_rotated_and_translated_to_ego_frame():
    local = np.array([[1.0, 0.0], [2.0, 0.0]])
    result = actor_local_to_ego_frame(local, (10.0, 3.0), np.pi / 2)
    np.testing.assert_allclose(result, [[10.0, 4.0], [10.0, 5.0]], atol=1e-7)


def test_coordinate_transform_rejects_wrong_shapes():
    with np.testing.assert_raises(ValueError):
        actor_local_to_ego_frame(np.zeros((3, 3)), (0.0, 0.0), 0.0)
