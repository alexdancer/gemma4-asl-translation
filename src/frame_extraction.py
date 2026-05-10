from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.video_ingest import VideoProbeResult

TARGET_FPS = 30.0
MAX_FRAMES = 300
CADENCE = "fixed_fps"
OVERFLOW_POLICY = "fail_fast"


@dataclass(frozen=True)
class ExtractedFrame:
    index: int
    timestamp_ms: int
    image_bgr: Any


@dataclass(frozen=True)
class FrameExtractionResult:
    frames: list[ExtractedFrame]
    frame_count: int
    first_ts_ms: int
    last_ts_ms: int
    effective_fps: float
    cadence: str


class FrameExtractionError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, status: str = "422 Unprocessable Entity", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.details = details or {}


def _expected_frame_count(duration_seconds: float, fps: float) -> int:
    if duration_seconds <= 0:
        return 0
    return int(math.ceil((duration_seconds * fps) - 1e-9))


def extract_frames_from_canonical_video(
    video_bytes: bytes,
    *,
    probe: VideoProbeResult,
    target_fps: float = TARGET_FPS,
    max_frames: int = MAX_FRAMES,
) -> FrameExtractionResult:
    expected_frames = _expected_frame_count(probe.duration_seconds, target_fps)
    if expected_frames > max_frames:
        raise FrameExtractionError(
            "FRAME_COUNT_EXCEEDED",
            "Frame extraction would exceed the configured maximum frame count",
            details={
                "fps": target_fps,
                "max_frames": max_frames,
                "video_duration_s": probe.duration_seconds,
                "expected_frame_count": expected_frames,
            },
        )

    with tempfile.TemporaryDirectory(prefix="asl_frame_extract_") as tmp_dir:
        video_path = Path(tmp_dir) / "canonical.mp4"
        video_path.write_bytes(video_bytes)

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise FrameExtractionError(
                "FRAME_EXTRACTION_UNAVAILABLE",
                "OpenCV (cv2) is required for backend frame extraction",
                retryable=False,
                status="503 Service Unavailable",
            ) from exc

        capture = cv2.VideoCapture(str(video_path))
        source_frames: list[Any] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                source_frames.append(frame)
        finally:
            capture.release()

    if not source_frames:
        raise FrameExtractionError("FRAME_EXTRACTION_FAILED", "No frames decoded from canonical video", retryable=False)

    sampled: list[ExtractedFrame] = []
    source_fps = probe.fps if probe.fps > 0 else target_fps
    for i in range(expected_frames):
        t_seconds = i / target_fps
        source_index = min(int(round(t_seconds * source_fps)), len(source_frames) - 1)
        sampled.append(ExtractedFrame(index=i, timestamp_ms=int(round(t_seconds * 1000.0)), image_bgr=source_frames[source_index]))

    if not sampled:
        raise FrameExtractionError("FRAME_EXTRACTION_FAILED", "No frames extracted after sampling", retryable=False)

    return FrameExtractionResult(
        frames=sampled,
        frame_count=len(sampled),
        first_ts_ms=sampled[0].timestamp_ms,
        last_ts_ms=sampled[-1].timestamp_ms,
        effective_fps=target_fps,
        cadence=CADENCE,
    )
