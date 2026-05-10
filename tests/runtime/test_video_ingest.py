from __future__ import annotations

import json
from pathlib import Path

from src.video_ingest import (
    CanonicalVideoProfile,
    VideoProbeResult,
    _normalization_required,
    _parse_fps,
    process_uploaded_video,
)


def test_parse_fps_fraction() -> None:
    assert _parse_fps("30000/1001") == 30000 / 1001


def test_normalization_required_for_non_canonical_profile() -> None:
    probe = VideoProbeResult(duration_seconds=2.0, fps=24.0, width=1920, height=1080)
    assert _normalization_required(probe, CanonicalVideoProfile()) is True


def test_normalization_required_when_codec_or_pix_fmt_or_audio_mismatch() -> None:
    profile = CanonicalVideoProfile()

    wrong_codec = VideoProbeResult(
        duration_seconds=2.0,
        fps=30.0,
        width=1280,
        height=720,
        codec="mpeg4",
        pixel_format="yuv420p",
        has_audio=False,
    )
    assert _normalization_required(wrong_codec, profile) is True

    wrong_pix_fmt = VideoProbeResult(
        duration_seconds=2.0,
        fps=30.0,
        width=1280,
        height=720,
        codec="h264",
        pixel_format="yuv422p",
        has_audio=False,
    )
    assert _normalization_required(wrong_pix_fmt, profile) is True

    has_audio = VideoProbeResult(
        duration_seconds=2.0,
        fps=30.0,
        width=1280,
        height=720,
        codec="h264",
        pixel_format="yuv420p",
        has_audio=True,
    )
    assert _normalization_required(has_audio, profile) is True


def test_process_uploaded_video_rejects_long_duration_without_transcode(monkeypatch, tmp_path: Path) -> None:
    profile = CanonicalVideoProfile()

    monkeypatch.setattr(
        "src.video_ingest.probe_video",
        lambda _path: VideoProbeResult(duration_seconds=11.2, fps=30.0, width=1280, height=720),
    )

    called = {"normalize": 0}

    def fake_normalize(_in: Path, _out: Path, _profile: CanonicalVideoProfile) -> None:
        called["normalize"] += 1

    monkeypatch.setattr("src.video_ingest.normalize_video", fake_normalize)

    original, normalized, canonical_bytes, applied, returned_profile = process_uploaded_video(
        b"bytes",
        "clip.mov",
        profile=profile,
    )

    assert original.duration_seconds == 11.2
    assert normalized.duration_seconds == 11.2
    assert canonical_bytes == b"bytes"
    assert applied is False
    assert returned_profile == profile
    assert called["normalize"] == 0
