"""Demo readiness gate for ASL evaluation metrics and runtime guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

Decision = Literal["go", "no_go"]
SplitRole = Literal["official", "iteration_only"]


@dataclass(frozen=True)
class ReadinessGateConfig:
    """Thresholds for the explicit demo go/no-go workflow."""

    min_macro_f1: float = 0.70
    max_latency_ms: float = 800.0
    min_stability: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_macro_f1 <= 1.0:
            raise ValueError("min_macro_f1 must be between 0.0 and 1.0.")
        if self.max_latency_ms < 0.0:
            raise ValueError("max_latency_ms must be non-negative.")
        if not 0.0 <= self.min_stability <= 1.0:
            raise ValueError("min_stability must be between 0.0 and 1.0.")


@dataclass(frozen=True)
class SplitMetrics:
    """Evaluation metrics for one split."""

    split_name: str
    split_role: SplitRole
    macro_f1: float
    sample_count: int


@dataclass(frozen=True)
class LatencyCheck:
    """Latency guardrail computed from measured inference runs."""

    p95_ms: float
    max_allowed_ms: float
    passed: bool


@dataclass(frozen=True)
class StabilityCheck:
    """Runtime stability guardrail based on successful repeated runs."""

    stable_runs: int
    total_runs: int
    stability: float
    min_required: float
    passed: bool


@dataclass(frozen=True)
class ReadinessReport:
    """One report containing official quality, latency, stability, and decision."""

    official_split: str
    iteration_split: str
    official_metrics: SplitMetrics
    iteration_metrics: SplitMetrics
    latency: LatencyCheck
    stability: StabilityCheck
    decision: Decision
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for scripts and UI."""

        return asdict(self)


def build_demo_readiness_report(
    *,
    signer_independent_truth: Sequence[str],
    signer_independent_predictions: Sequence[str],
    random_split_truth: Sequence[str],
    random_split_predictions: Sequence[str],
    latency_ms: Sequence[float],
    stable_runs: int,
    total_runs: int,
    config: ReadinessGateConfig | None = None,
) -> ReadinessReport:
    """Build the official demo readiness report.

    Signer-independent metrics are always official. Random split metrics are
    included only as iteration feedback so the report cannot overstate
    generalization.
    """

    config = config or ReadinessGateConfig()
    official_metrics = SplitMetrics(
        split_name="signer_independent",
        split_role="official",
        macro_f1=_macro_f1(signer_independent_truth, signer_independent_predictions),
        sample_count=len(signer_independent_truth),
    )
    iteration_metrics = SplitMetrics(
        split_name="random",
        split_role="iteration_only",
        macro_f1=_macro_f1(random_split_truth, random_split_predictions),
        sample_count=len(random_split_truth),
    )
    latency = _latency_check(latency_ms, config.max_latency_ms)
    stability = _stability_check(stable_runs, total_runs, config.min_stability)

    blockers: list[str] = []
    if official_metrics.macro_f1 < config.min_macro_f1:
        blockers.append(
            f"official signer-independent macro_f1 {official_metrics.macro_f1:.3f} "
            f"is below required {config.min_macro_f1:.3f}"
        )
    if not latency.passed:
        blockers.append(
            f"p95 latency {latency.p95_ms:.1f}ms exceeds allowed {latency.max_allowed_ms:.1f}ms"
        )
    if not stability.passed:
        blockers.append(
            f"stability {stability.stability:.3f} is below required {stability.min_required:.3f}"
        )

    return ReadinessReport(
        official_split="signer_independent",
        iteration_split="random",
        official_metrics=official_metrics,
        iteration_metrics=iteration_metrics,
        latency=latency,
        stability=stability,
        decision="no_go" if blockers else "go",
        blockers=tuple(blockers),
    )


def _macro_f1(truth: Sequence[str], predictions: Sequence[str]) -> float:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")
    if not truth:
        raise ValueError("truth and predictions must not be empty.")

    labels = sorted(set(truth) | set(predictions))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(1 for actual, predicted in zip(truth, predictions) if actual == label and predicted == label)
        false_positive = sum(1 for actual, predicted in zip(truth, predictions) if actual != label and predicted == label)
        false_negative = sum(1 for actual, predicted in zip(truth, predictions) if actual == label and predicted != label)
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)

    return round(sum(scores) / len(scores), 3)


def _latency_check(latency_ms: Sequence[float], max_allowed_ms: float) -> LatencyCheck:
    if not latency_ms:
        raise ValueError("latency_ms must contain at least one measurement.")
    measurements = sorted(float(value) for value in latency_ms)
    if any(value < 0.0 for value in measurements):
        raise ValueError("latency_ms measurements must be non-negative.")

    index = max(0, int(round(0.95 * (len(measurements) - 1))))
    p95_ms = round(measurements[index], 3)
    return LatencyCheck(
        p95_ms=p95_ms,
        max_allowed_ms=max_allowed_ms,
        passed=p95_ms <= max_allowed_ms,
    )


def _stability_check(stable_runs: int, total_runs: int, min_required: float) -> StabilityCheck:
    if total_runs <= 0:
        raise ValueError("total_runs must be positive.")
    if stable_runs < 0:
        raise ValueError("stable_runs must be non-negative.")
    if stable_runs > total_runs:
        raise ValueError("stable_runs cannot exceed total_runs.")

    stability = round(stable_runs / total_runs, 3)
    return StabilityCheck(
        stable_runs=stable_runs,
        total_runs=total_runs,
        stability=stability,
        min_required=min_required,
        passed=stability >= min_required,
    )

