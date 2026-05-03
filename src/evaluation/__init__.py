"""Evaluation and demo readiness reporting."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ClassMetrics",
    "LatencyCheck",
    "LossCurve",
    "MetricSummary",
    "Phase2AConfig",
    "Phase2AReport",
    "Phase2ASplitSummary",
    "ReadinessGateConfig",
    "ReadinessReport",
    "SplitMetrics",
    "StabilityCheck",
    "build_demo_readiness_report",
    "build_phase2a_report",
    "write_phase2a_artifacts",
]


_READINESS_EXPORTS = {
    "LatencyCheck",
    "ReadinessGateConfig",
    "ReadinessReport",
    "SplitMetrics",
    "StabilityCheck",
    "build_demo_readiness_report",
}

_PHASE2A_EXPORTS = {
    "ClassMetrics",
    "LossCurve",
    "MetricSummary",
    "Phase2AConfig",
    "Phase2AReport",
    "build_phase2a_report",
    "write_phase2a_artifacts",
}


def __getattr__(name: str) -> Any:
    """Lazily load evaluation exports so optional deps do not affect all users."""

    if name in _READINESS_EXPORTS:
        module = import_module("src.evaluation.readiness")
        return getattr(module, name)
    if name == "Phase2ASplitSummary":
        module = import_module("src.evaluation.phase2a")
        return getattr(module, "SplitSummary")
    if name in _PHASE2A_EXPORTS:
        module = import_module("src.evaluation.phase2a")
        return getattr(module, name)
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
