"""Behavior tests for the confidence-aware demo output contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.demo.output_contract import DemoOutputConfig, format_demo_output


@dataclass(frozen=True)
class PredictionResult:
    ok: bool
    prediction: Optional[str]
    confidence: float
    latency_ms: float
    latency_target_ms: float = 800.0
    error: Optional[str] = None


def test_low_confidence_prediction_renders_as_uncertain() -> None:
    prediction = PredictionResult(
        ok=True,
        prediction="hello",
        confidence=0.42,
        latency_ms=17.5,
    )

    output = format_demo_output(prediction, DemoOutputConfig(confidence_threshold=0.70))

    assert output.prediction == "hello"
    assert output.confidence == 0.42
    assert output.confidence_threshold == 0.70
    assert output.is_uncertain is True
    assert output.display_text == "uncertain"
    assert output.status == "uncertain"


def test_confident_prediction_renders_predicted_gloss() -> None:
    prediction = PredictionResult(
        ok=True,
        prediction="thanks",
        confidence=0.91,
        latency_ms=12.0,
    )

    output = format_demo_output(prediction, DemoOutputConfig(confidence_threshold=0.70))

    assert output.display_text == "thanks"
    assert output.is_uncertain is False
    assert output.status == "ok"
