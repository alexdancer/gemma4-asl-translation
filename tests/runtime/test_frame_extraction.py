from __future__ import annotations

import pytest

from src.frame_extraction import (
    MAX_FRAMES,
    TARGET_FPS,
    FrameExtractionError,
    _expected_frame_count,
    extract_frames_from_canonical_video,
)
from src.video_ingest import VideoProbeResult


def test_expected_frame_count_uses_fixed_30fps_grid() -> None:
    assert _expected_frame_count(1.0, TARGET_FPS) == 30
    assert _expected_frame_count(0.5, TARGET_FPS) == 15
    assert _expected_frame_count(0.0, TARGET_FPS) == 0


def test_extract_frames_fails_fast_when_expected_frame_count_exceeds_limit() -> None:
    probe = VideoProbeResult(duration_seconds=10.1, fps=30.0, width=1280, height=720)

    with pytest.raises(FrameExtractionError) as exc_info:
        extract_frames_from_canonical_video(b"not-used", probe=probe, target_fps=TARGET_FPS, max_frames=MAX_FRAMES)

    exc = exc_info.value
    assert exc.code == "FRAME_COUNT_EXCEEDED"
    assert exc.retryable is False
    assert exc.status == "422 Unprocessable Entity"
    assert exc.details["fps"] == 30.0
    assert exc.details["max_frames"] == 300
    assert exc.details["video_duration_s"] == 10.1
