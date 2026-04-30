"""Behavior tests for live capture feature streaming."""

from __future__ import annotations

import numpy as np

from src.data.live_capture import LiveFeatureStream


class FakeFrameSource:
    def __init__(self, reads: list[tuple[bool, np.ndarray | None]]) -> None:
        self.reads = list(reads)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.reads:
            return False, None
        return self.reads.pop(0)

    def stop(self) -> None:
        self.stopped = True


class FakeExtractor:
    def extract_from_frame(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        value = float(frame_bgr[0, 0, 0])
        return {
            "body": np.full((17, 4), value, dtype=np.float32),
            "left_hand": np.full((21, 4), value + 1.0, dtype=np.float32),
            "right_hand": np.full((21, 4), value + 2.0, dtype=np.float32),
            "face": np.zeros((0, 4), dtype=np.float32),
        }


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_live_feature_stream_starts_and_stops_source() -> None:
    source = FakeFrameSource(reads=[])
    stream = LiveFeatureStream(source=source, extractor=FakeExtractor())

    stream.start()
    stream.stop()

    assert source.started is True
    assert source.stopped is True


def test_live_feature_stream_emits_flattened_hands_pose_window() -> None:
    source = FakeFrameSource(reads=[(True, _frame(3)), (True, _frame(4))])
    stream = LiveFeatureStream(source=source, extractor=FakeExtractor())

    window = stream.capture_window(frame_count=2)

    assert window.features.shape == (2, (17 + 21 + 21) * 4)
    assert window.missing_frame_count == 0
    assert window.components["body"].shape == (2, 17, 4)
    assert np.allclose(window.features[0, :4], [3.0, 3.0, 3.0, 3.0])


def test_live_feature_stream_substitutes_missing_frames_without_crashing() -> None:
    source = FakeFrameSource(reads=[(False, None), (True, _frame(5))])
    stream = LiveFeatureStream(source=source, extractor=FakeExtractor())

    window = stream.capture_window(frame_count=2)

    assert window.features.shape == (2, (17 + 21 + 21) * 4)
    assert window.missing_frame_count == 1
    assert np.allclose(window.features[0], 0.0)
    assert np.any(window.features[1] != 0.0)
