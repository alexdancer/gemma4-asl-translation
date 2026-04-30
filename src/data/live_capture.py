"""Live camera capture to hands+pose feature stream utilities."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

BODY_JOINT_COUNT = 17
HAND_JOINT_COUNT = 21
LANDMARK_DIMENSIONS = 4
STREAM_COMPONENTS: Tuple[str, ...] = ("body", "left_hand", "right_hand")


class FrameSource(Protocol):
    """Minimal frame source interface used by the live feature stream."""

    def start(self) -> None:
        """Open the frame source."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one BGR frame from the source."""

    def stop(self) -> None:
        """Release the frame source."""


@dataclass(frozen=True)
class FeatureFrame:
    """One normalized live feature frame."""

    components: Mapping[str, np.ndarray]
    features: np.ndarray
    timestamp_s: float
    is_missing: bool


@dataclass(frozen=True)
class FeatureWindow:
    """A temporal hands+pose feature window consumable by inference."""

    components: Mapping[str, np.ndarray]
    features: np.ndarray
    missing_frame_count: int

    @property
    def sequence_length(self) -> int:
        return int(self.features.shape[0])


class OpenCVCameraSource:
    """OpenCV-backed live camera source."""

    def __init__(self, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: Optional[Any] = None

    def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError("opencv-python is required for OpenCVCameraSource.") from exc

        capture = cv2.VideoCapture(self.camera_index)
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")
        self._capture = capture

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._capture is None:
            raise RuntimeError("Camera source has not been started.")
        success, frame = self._capture.read()
        return bool(success), frame if success else None

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class PrerecordedVideoSource:
    """Prerecorded media source for deterministic demo fallback."""

    def __init__(self, media_path: Path | str) -> None:
        self.media_path = Path(media_path)
        self._capture: Optional[Any] = None
        self._array_frames: Optional[np.ndarray] = None
        self._array_index = 0

    def start(self) -> None:
        if not self.media_path.exists():
            raise FileNotFoundError(f"Prerecorded media path does not exist: {self.media_path}")
        if self.media_path.suffix == ".npy":
            frames = np.load(self.media_path)
            if frames.ndim != 4:
                raise ValueError("NumPy prerecorded media must have shape (frames, height, width, channels).")
            self._array_frames = frames.astype(np.uint8, copy=False)
            self._array_index = 0
            return

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError("opencv-python is required for video media paths; use .npy clips for lightweight runs.") from exc

        capture = cv2.VideoCapture(str(self.media_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open prerecorded media path: {self.media_path}")
        self._capture = capture

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._array_frames is not None:
            if self._array_index >= len(self._array_frames):
                return False, None
            frame = self._array_frames[self._array_index]
            self._array_index += 1
            return True, frame

        if self._capture is None:
            raise RuntimeError("Prerecorded media source has not been started.")
        success, frame = self._capture.read()
        return bool(success), frame if success else None

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._array_frames = None
        self._array_index = 0


class LiveFeatureStream:
    """Convert live BGR frames into normalized hands+pose temporal features."""

    def __init__(self, source: FrameSource, extractor: Any, normalize_features: bool = False) -> None:
        self.source = source
        self.extractor = extractor
        self.normalize_features = normalize_features
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.source.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            self.source.stop()
            return
        self.source.stop()
        self._started = False

    def capture_next(self) -> FeatureFrame:
        success, frame = self.source.read()
        if not success or frame is None:
            return self._missing_feature_frame()

        try:
            components = self.extractor.extract_from_frame(frame)
        except Exception as exc:  # pragma: no cover - defensive live demo path
            LOGGER.warning("Pose extraction failed for live frame; substituting zeros: %s", exc)
            return self._missing_feature_frame()

        stream_components = self._coerce_stream_components(components)
        features = flatten_stream_components(stream_components, normalize=self.normalize_features)
        return FeatureFrame(
            components=stream_components,
            features=features[0],
            timestamp_s=time.monotonic(),
            is_missing=False,
        )

    def capture_window(self, frame_count: int) -> FeatureWindow:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")

        frames = [self.capture_next() for _ in range(frame_count)]
        components: Dict[str, np.ndarray] = {}
        for component_name in STREAM_COMPONENTS:
            components[component_name] = np.stack(
                [feature_frame.components[component_name] for feature_frame in frames],
                axis=0,
            )
        features = np.stack([feature_frame.features for feature_frame in frames], axis=0).astype(np.float32)
        return FeatureWindow(
            components=components,
            features=features,
            missing_frame_count=sum(1 for feature_frame in frames if feature_frame.is_missing),
        )

    def _missing_feature_frame(self) -> FeatureFrame:
        components = _zero_components()
        features = flatten_stream_components(components, normalize=self.normalize_features)
        return FeatureFrame(
            components=components,
            features=features[0],
            timestamp_s=time.monotonic(),
            is_missing=True,
        )

    def _coerce_stream_components(self, components: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        coerced = _zero_components()
        for component_name in STREAM_COMPONENTS:
            component = components.get(component_name)
            if component is not None and component.shape == coerced[component_name].shape:
                coerced[component_name] = component.astype(np.float32)
        return coerced


def _zero_components() -> Dict[str, np.ndarray]:
    return {
        "body": np.zeros((BODY_JOINT_COUNT, LANDMARK_DIMENSIONS), dtype=np.float32),
        "left_hand": np.zeros((HAND_JOINT_COUNT, LANDMARK_DIMENSIONS), dtype=np.float32),
        "right_hand": np.zeros((HAND_JOINT_COUNT, LANDMARK_DIMENSIONS), dtype=np.float32),
    }


def flatten_stream_components(
    components: Mapping[str, np.ndarray],
    normalize: bool = False,
) -> np.ndarray:
    """Flatten live body and hand components into `(1, features)`."""

    arrays: List[np.ndarray] = []
    for component_name in STREAM_COMPONENTS:
        component = components[component_name]
        if component.ndim != 2:
            raise ValueError(f"Live component '{component_name}' must have shape (joints, channels).")
        arrays.append(component.reshape(1, -1))

    features = np.concatenate(arrays, axis=1).astype(np.float32)
    if not normalize:
        return features

    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)
