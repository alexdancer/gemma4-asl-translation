from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.pose_handoff import (
    PosePipelineError,
    build_pose_extraction_request,
    run_pose_extraction_pipeline,
)


class _FakeExtractor:
    def extract_from_frame(self, _frame):
        return {"body": [[0.0, 0.0, 0.0, 1.0]]}

    def close(self):
        return None


def _frame_extraction_stub():
    frames = [
        SimpleNamespace(index=0, timestamp_ms=0, image_bgr="frame0"),
        SimpleNamespace(index=1, timestamp_ms=33, image_bgr="frame1"),
    ]
    return SimpleNamespace(frames=frames, effective_fps=30.0, cadence="fixed_fps")


def test_build_pose_extraction_request_preserves_index_and_timestamp_alignment() -> None:
    request = build_pose_extraction_request(request_id="rid-1", frame_extraction=_frame_extraction_stub())

    assert request.request_id == "rid-1"
    assert request.frame_count == 2
    assert request.frames[0].index == 0
    assert request.frames[0].timestamp_ms == 0
    assert request.frames[1].index == 1
    assert request.frames[1].timestamp_ms == 33


def test_run_pose_extraction_pipeline_preserves_frame_timestamps() -> None:
    result = run_pose_extraction_pipeline(
        request_id="rid-1",
        frame_extraction=_frame_extraction_stub(),
        extractor_factory=lambda: _FakeExtractor(),
    )

    assert result.frame_count == 2
    assert result.frames[0].index == 0
    assert result.frames[0].timestamp_ms == 0
    assert result.frames[1].index == 1
    assert result.frames[1].timestamp_ms == 33
    assert result.first_ts_ms == 0
    assert result.last_ts_ms == 33


def test_run_pose_extraction_pipeline_returns_422_for_pose_failure() -> None:
    class _BadExtractor:
        def extract_from_frame(self, _frame):
            raise ValueError("bad frame")

        def close(self):
            return None

    with pytest.raises(PosePipelineError) as exc_info:
        run_pose_extraction_pipeline(
            request_id="rid-1",
            frame_extraction=_frame_extraction_stub(),
            extractor_factory=lambda: _BadExtractor(),
        )

    exc = exc_info.value
    assert exc.code == "POSE_EXTRACTION_FAILED"
    assert exc.status == "422 Unprocessable Entity"
    assert exc.retryable is False
    assert exc.details["index"] == 0
    assert exc.details["timestamp_ms"] == 0


def test_run_pose_extraction_pipeline_returns_503_for_dependency_unavailable() -> None:
    def _missing_factory():
        raise ImportError("mediapipe missing")

    with pytest.raises(PosePipelineError) as exc_info:
        run_pose_extraction_pipeline(
            request_id="rid-1",
            frame_extraction=_frame_extraction_stub(),
            extractor_factory=_missing_factory,
        )

    exc = exc_info.value
    assert exc.code == "POSE_EXTRACTION_UNAVAILABLE"
    assert exc.status == "503 Service Unavailable"
    assert exc.retryable is True
