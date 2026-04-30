"""Prerecorded clip fallback orchestration for live inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

from src.demo.output_contract import DemoOutput, DemoOutputConfig, PredictionResult, format_demo_output

LOGGER = logging.getLogger(__name__)

DemoInferenceMode = Literal["live", "prerecorded"]


class FeatureStream(Protocol):
    """Capture stream interface shared by live camera and prerecorded clips."""

    def start(self) -> None:
        """Open the stream."""

    def capture_window(self, frame_count: int) -> object:
        """Capture one temporal feature window."""

    def stop(self) -> None:
        """Release the stream."""


@dataclass(frozen=True)
class DemoInferenceRunConfig:
    """One-shot demo inference settings."""

    mode: DemoInferenceMode
    frame_count: int
    media_path: Optional[str] = None
    output_config: DemoOutputConfig = DemoOutputConfig()

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        if self.mode == "prerecorded" and not self.media_path:
            raise ValueError("media_path is required for prerecorded mode.")


@dataclass(frozen=True)
class DemoInferenceRunResult:
    """Observable demo inference result for UI/logs."""

    mode: DemoInferenceMode
    output: DemoOutput
    observation: str
    media_path: Optional[str] = None


def run_demo_inference_once(
    stream: FeatureStream,
    predict: Callable[[object], PredictionResult],
    config: DemoInferenceRunConfig,
) -> DemoInferenceRunResult:
    """Capture one feature window and run it through the provided model path."""

    observation = _format_observation(config)
    LOGGER.info("Starting demo inference: %s", observation)
    stream.start()
    try:
        feature_window = stream.capture_window(config.frame_count)
        prediction = predict(feature_window)
        output = format_demo_output(prediction, config.output_config)
        LOGGER.info(
            "Completed demo inference: %s status=%s display=%s confidence=%.3f",
            observation,
            output.status,
            output.display_text,
            output.confidence,
        )
        return DemoInferenceRunResult(
            mode=config.mode,
            media_path=config.media_path,
            output=output,
            observation=observation,
        )
    finally:
        stream.stop()


def _format_observation(config: DemoInferenceRunConfig) -> str:
    parts = [f"mode={config.mode}", f"frame_count={config.frame_count}"]
    if config.media_path is not None:
        parts.append(f"media_path={config.media_path}")
    return " ".join(parts)
