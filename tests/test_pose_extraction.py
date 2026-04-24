"""Unit tests for pose extraction utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.data.pose_extractor import _landmarks_to_array, normalize_pose_coordinates


def _make_landmark(x: float, y: float, z: float, visibility: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def test_landmarks_to_array_returns_expected_shape() -> None:
    landmarks = SimpleNamespace(
        landmark=[
            _make_landmark(0.1, 0.2, 0.3),
            _make_landmark(0.4, 0.5, 0.6),
            _make_landmark(0.7, 0.8, 0.9),
        ]
    )

    array = _landmarks_to_array(landmarks, indices=[0, 2])

    assert array.shape == (2, 4)
    assert np.allclose(array[0], [0.1, 0.2, 0.3, 1.0])
    assert np.allclose(array[1], [0.7, 0.8, 0.9, 1.0])


def test_normalize_pose_coordinates_centers_shoulders() -> None:
    pose = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_pose_coordinates(pose, left_shoulder_idx=1, right_shoulder_idx=2)

    assert normalized.shape == (3, 4)
    assert np.allclose(normalized[1, :3], [-0.5, 0.0, 0.0])
    assert np.allclose(normalized[2, :3], [0.5, 0.0, 0.0])
