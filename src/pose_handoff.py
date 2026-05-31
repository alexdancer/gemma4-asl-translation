from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PoseInputFrame:
    index: int
    timestamp_ms: int
    image_bgr: Any


@dataclass(frozen=True)
class PoseExtractionRequest:
    request_id: str
    frames: list[PoseInputFrame]
    frame_count: int
    effective_fps: float
    cadence: str


@dataclass(frozen=True)
class PoseOutputFrame:
    index: int
    timestamp_ms: int
    landmarks: dict[str, Any]


@dataclass(frozen=True)
class PoseExtractionResult:
    frames: list[PoseOutputFrame]
    frame_count: int
    first_ts_ms: int
    last_ts_ms: int


class PosePipelineError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, status: str = "422 Unprocessable Entity", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.details = details or {}


def build_pose_extraction_request(*, request_id: str, frame_extraction: Any) -> PoseExtractionRequest:
    frames = [
        PoseInputFrame(index=int(frame.index), timestamp_ms=int(frame.timestamp_ms), image_bgr=frame.image_bgr)
        for frame in frame_extraction.frames
    ]
    return PoseExtractionRequest(
        request_id=request_id,
        frames=frames,
        frame_count=len(frames),
        effective_fps=float(frame_extraction.effective_fps),
        cadence=str(frame_extraction.cadence),
    )


def _default_extractor_factory():
    from src.data.pose_extractor import PoseExtractor

    return PoseExtractor()


def run_pose_extraction_pipeline(
    *,
    request_id: str,
    frame_extraction: Any,
    extractor_factory: Any = None,
) -> PoseExtractionResult:
    request = build_pose_extraction_request(request_id=request_id, frame_extraction=frame_extraction)
    if request.frame_count == 0:
        raise PosePipelineError("POSE_EXTRACTION_FAILED", "No extracted frames were available for pose extraction", retryable=False)

    factory = extractor_factory or _default_extractor_factory
    try:
        extractor = factory()
    except ImportError as exc:
        raise PosePipelineError(
            "POSE_EXTRACTION_UNAVAILABLE",
            "Pose extraction dependencies are unavailable",
            retryable=True,
            status="503 Service Unavailable",
        ) from exc

    outputs: list[PoseOutputFrame] = []
    try:
        for frame in request.frames:
            try:
                landmarks = extractor.extract_from_frame(frame.image_bgr)
            except Exception as exc:
                raise PosePipelineError(
                    "POSE_EXTRACTION_FAILED",
                    f"Pose extraction failed at frame index {frame.index}",
                    retryable=False,
                    status="422 Unprocessable Entity",
                    details={"index": frame.index, "timestamp_ms": frame.timestamp_ms, "error": str(exc)},
                ) from exc
            outputs.append(PoseOutputFrame(index=frame.index, timestamp_ms=frame.timestamp_ms, landmarks=landmarks))
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            close()

    return PoseExtractionResult(
        frames=outputs,
        frame_count=len(outputs),
        first_ts_ms=outputs[0].timestamp_ms,
        last_ts_ms=outputs[-1].timestamp_ms,
    )
