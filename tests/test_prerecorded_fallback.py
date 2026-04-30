"""Behavior tests for prerecorded fallback mode A."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.demo.fallback_a import DemoInferenceRunConfig, run_demo_inference_once
from src.data.live_capture import PrerecordedVideoSource
from scripts.run_prerecorded_fallback import main as run_prerecorded_main


def _write_video(path: Path, frame_values: list[int]) -> None:
    frames = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in frame_values], axis=0)
    with path.open("wb") as handle:
        np.save(handle, frames)


def test_prerecorded_video_source_reads_frames_from_media_path(tmp_path: Path) -> None:
    media_path = tmp_path / "demo_clip.npy"
    _write_video(media_path, [25, 125])
    source = PrerecordedVideoSource(media_path)

    source.start()
    first_ok, first_frame = source.read()
    second_ok, second_frame = source.read()
    end_ok, end_frame = source.read()
    source.stop()

    assert first_ok is True
    assert second_ok is True
    assert first_frame is not None
    assert second_frame is not None
    assert first_frame.shape == (4, 4, 3)
    assert end_ok is False
    assert end_frame is None


class FakeStream:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.window = type("FeatureWindow", (), {"features": np.ones((2, 3), dtype=np.float32)})()

    def start(self) -> None:
        self.started = True

    def capture_window(self, frame_count: int) -> object:
        assert frame_count == 2
        return self.window

    def stop(self) -> None:
        self.stopped = True


class FakePrediction:
    ok = True
    prediction = "thanks"
    confidence = 0.93
    latency_ms = 12.0
    latency_target_ms = 800.0
    error = None


def test_prerecorded_fallback_runs_same_inference_path_with_observable_mode() -> None:
    stream = FakeStream()
    calls: list[object] = []

    def predict(window: object) -> FakePrediction:
        calls.append(window)
        return FakePrediction()

    result = run_demo_inference_once(
        stream=stream,
        predict=predict,
        config=DemoInferenceRunConfig(mode="prerecorded", frame_count=2, media_path="demo.mp4"),
    )

    assert stream.started is True
    assert stream.stopped is True
    assert calls == [stream.window]
    assert result.mode == "prerecorded"
    assert result.media_path == "demo.mp4"
    assert result.output.display_text == "thanks"
    assert "mode=prerecorded" in result.observation


def test_prerecorded_fallback_demo_script_executes_end_to_end(tmp_path: Path, capsys) -> None:
    media_path = tmp_path / "thanks_clip.npy"
    _write_video(media_path, [200, 210, 220, 230])

    exit_code = run_prerecorded_main([
        "--media-path",
        str(media_path),
        "--frame-count",
        "4",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"mode": "prerecorded"' in captured.out
    assert '"display_text": "thanks"' in captured.out
