"""MediaPipe Holistic based pose extraction utilities for ASL videos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

POSE_LANDMARK_INDICES: Tuple[int, ...] = (
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
)


class PoseExtractionError(RuntimeError):
    """Raised when pose extraction fails for a sample."""


def _landmarks_to_array(landmarks: Any, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Convert MediaPipe landmarks into a dense `(N, 4)` array."""

    if landmarks is None:
        count = len(indices) if indices is not None else 0
        return np.zeros((count, 4), dtype=np.float32)

    selected = landmarks.landmark if indices is None else [landmarks.landmark[idx] for idx in indices]
    array = np.asarray(
        [[point.x, point.y, point.z, getattr(point, "visibility", 1.0)] for point in selected],
        dtype=np.float32,
    )
    return array


def normalize_pose_coordinates(
    pose_array: np.ndarray,
    left_shoulder_idx: int = 1,
    right_shoulder_idx: int = 2,
) -> np.ndarray:
    """Center and scale pose coordinates using the shoulder midpoint and distance."""

    if pose_array.ndim != 2 or pose_array.shape[1] < 3:
        raise ValueError("Expected pose_array with shape (num_joints, >=3)")

    coords = pose_array[:, :3].copy()
    confidence = pose_array[:, 3:].copy() if pose_array.shape[1] > 3 else None

    reference_center = (coords[left_shoulder_idx] + coords[right_shoulder_idx]) / 2.0
    shoulder_distance = np.linalg.norm(coords[left_shoulder_idx] - coords[right_shoulder_idx])
    scale = shoulder_distance if shoulder_distance > 1e-6 else 1.0

    normalized = (coords - reference_center) / scale
    if confidence is not None:
        normalized = np.concatenate([normalized, confidence], axis=1)
    return normalized.astype(np.float32)


class PoseExtractor:
    """Extract body, hand, and optional face landmarks from image frames."""

    def __init__(
        self,
        include_face: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError(
                "mediapipe is required for PoseExtractor. Install dependencies from requirements.txt."
            ) from exc

        self.include_face = include_face
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""

        self._holistic.close()

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract normalized landmarks from one BGR frame."""

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(frame_rgb)

        body = _landmarks_to_array(results.pose_landmarks, POSE_LANDMARK_INDICES)
        left_hand = _landmarks_to_array(results.left_hand_landmarks, range(21))
        right_hand = _landmarks_to_array(results.right_hand_landmarks, range(21))
        face = _landmarks_to_array(results.face_landmarks, range(468)) if self.include_face else np.zeros((0, 4), dtype=np.float32)

        normalized_body = normalize_pose_coordinates(body) if body.size else body
        normalized_left = normalize_pose_coordinates(left_hand, 0, 9) if left_hand.size else left_hand
        normalized_right = normalize_pose_coordinates(right_hand, 0, 9) if right_hand.size else right_hand

        return {
            "body": normalized_body,
            "left_hand": normalized_left,
            "right_hand": normalized_right,
            "face": face.astype(np.float32),
        }

    def extract_sequence(self, frames_bgr: Iterable[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract pose arrays for an entire frame sequence."""

        body_frames: List[np.ndarray] = []
        left_frames: List[np.ndarray] = []
        right_frames: List[np.ndarray] = []
        face_frames: List[np.ndarray] = []

        frame_count = 0
        for frame in frames_bgr:
            landmarks = self.extract_from_frame(frame)
            body_frames.append(landmarks["body"])
            left_frames.append(landmarks["left_hand"])
            right_frames.append(landmarks["right_hand"])
            if self.include_face:
                face_frames.append(landmarks["face"])
            frame_count += 1

        if frame_count == 0:
            raise PoseExtractionError("Cannot extract poses from an empty frame sequence.")

        sequence = {
            "body": np.stack(body_frames, axis=0),
            "left_hand": np.stack(left_frames, axis=0),
            "right_hand": np.stack(right_frames, axis=0),
        }
        if self.include_face:
            sequence["face"] = np.stack(face_frames, axis=0)
        return sequence

    def extract_from_video(self, video_path: Path, max_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Load a video from disk and extract a pose sequence."""

        from src.data.wlasl_loader import load_video_frames

        frames = load_video_frames(video_path, max_frames=max_frames)
        LOGGER.info("Extracting poses from %s (%d frames)", video_path, len(frames))
        return self.extract_sequence(frames)


def save_pose_sequence(
    sequence: Dict[str, np.ndarray],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist extracted pose arrays to a compressed NumPy archive."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = dict(sequence)
    if metadata is not None:
        arrays["metadata_json"] = np.asarray(str(metadata), dtype=object)

    np.savez_compressed(output_path, **arrays)
    LOGGER.info("Saved pose sequence to %s", output_path)
    return output_path
