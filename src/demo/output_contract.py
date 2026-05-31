"""Confidence-aware output contract for the ASL demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

DEFAULT_CONFIDENCE_THRESHOLD = 0.70

DemoOutputStatus = Literal["ok", "uncertain", "error"]


class PredictionResult(Protocol):
    """Inference result shape consumed by the demo output formatter."""

    ok: bool
    prediction: Optional[str]
    confidence: float
    latency_ms: float
    latency_target_ms: float
    error: Optional[str]


@dataclass(frozen=True)
class DemoOutputConfig:
    """Demo output settings that presenters can tune before showtime."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class DemoOutput:
    """Stable prediction payload for UI, logs, and fallback orchestration."""

    status: DemoOutputStatus
    display_text: str
    prediction: Optional[str]
    confidence: float
    confidence_threshold: float
    is_uncertain: bool
    latency_ms: float
    latency_target_ms: float
    error: Optional[str] = None


def format_demo_output(
    prediction: PredictionResult,
    config: DemoOutputConfig | None = None,
) -> DemoOutput:
    """Convert model inference into the confidence-aware demo contract."""

    config = config or DemoOutputConfig()
    confidence = _clamp_confidence(prediction.confidence)
    if not prediction.ok:
        return DemoOutput(
            status="error",
            display_text="uncertain",
            prediction=prediction.prediction,
            confidence=confidence,
            confidence_threshold=config.confidence_threshold,
            is_uncertain=True,
            latency_ms=prediction.latency_ms,
            latency_target_ms=prediction.latency_target_ms,
            error=prediction.error,
        )

    is_uncertain = confidence < config.confidence_threshold
    display_text = "uncertain" if is_uncertain else str(prediction.prediction)
    return DemoOutput(
        status="uncertain" if is_uncertain else "ok",
        display_text=display_text,
        prediction=prediction.prediction,
        confidence=confidence,
        confidence_threshold=config.confidence_threshold,
        is_uncertain=is_uncertain,
        latency_ms=prediction.latency_ms,
        latency_target_ms=prediction.latency_target_ms,
        error=prediction.error,
    )


def _clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, float(confidence)))
