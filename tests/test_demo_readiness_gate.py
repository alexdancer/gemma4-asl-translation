"""Behavior tests for demo readiness reporting."""

from __future__ import annotations

from src.evaluation.readiness import (
    ReadinessGateConfig,
    build_demo_readiness_report,
)


def test_readiness_report_uses_signer_independent_metrics_as_official() -> None:
    report = build_demo_readiness_report(
        signer_independent_truth=["hello", "thanks", "hello", "thanks"],
        signer_independent_predictions=["hello", "thanks", "hello", "hello"],
        random_split_truth=["hello", "thanks"],
        random_split_predictions=["hello", "thanks"],
        latency_ms=[210.0, 350.0, 410.0],
        stable_runs=9,
        total_runs=10,
        config=ReadinessGateConfig(
            min_macro_f1=0.70,
            max_latency_ms=800.0,
            min_stability=0.90,
        ),
    )

    assert report.official_split == "signer_independent"
    assert report.iteration_split == "random"
    assert report.official_metrics.split_role == "official"
    assert report.iteration_metrics.split_role == "iteration_only"
    assert report.official_metrics.macro_f1 == 0.733
    assert report.iteration_metrics.macro_f1 == 1.0
    assert report.latency.p95_ms == 410.0
    assert report.stability.stable_runs == 9
    assert report.stability.total_runs == 10
    assert report.decision == "go"
    assert report.as_dict()["decision"] == "go"


def test_readiness_report_returns_no_go_with_actionable_blockers() -> None:
    report = build_demo_readiness_report(
        signer_independent_truth=["hello", "thanks", "yes", "no"],
        signer_independent_predictions=["hello", "hello", "hello", "hello"],
        random_split_truth=["hello", "thanks"],
        random_split_predictions=["hello", "thanks"],
        latency_ms=[700.0, 950.0, 1100.0],
        stable_runs=7,
        total_runs=10,
        config=ReadinessGateConfig(
            min_macro_f1=0.80,
            max_latency_ms=800.0,
            min_stability=0.90,
        ),
    )

    assert report.decision == "no_go"
    assert len(report.blockers) == 3
    assert "macro_f1" in report.blockers[0]
    assert "latency" in report.blockers[1]
    assert "stability" in report.blockers[2]
