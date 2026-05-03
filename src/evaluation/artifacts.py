"""Artifact writing module for evaluation and diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths written by an evaluator run."""

    predictions_csv: Path
    metrics_json: Path


@dataclass(frozen=True)
class ConstrainedEvaluationArtifacts:
    """Paths written by a constrained diagnostic evaluator run."""

    constrained_predictions_csv: Path
    constrained_metrics_json: Path
    comparison_json: Path


@dataclass(frozen=True)
class PromptControlEvaluationArtifacts:
    """Paths written by a prompt-control evaluator run."""

    predictions_csv: Path
    metrics_json: Path
    comparison_json: Path
    report_md: Path


def write_evaluation_artifacts(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    out_dir: Path | str,
) -> EvaluationArtifacts:
    """Write predictions CSV and metrics JSON."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    fieldnames = [
        "index",
        "sample_id",
        "expected_gloss",
        "predicted_gloss",
        "raw_model_output",
        "valid_label",
        "correct",
        "mode",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return EvaluationArtifacts(predictions_csv=predictions_path, metrics_json=metrics_path)


def write_constrained_evaluation_artifacts(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    out_dir: Path | str,
) -> ConstrainedEvaluationArtifacts:
    """Write constrained predictions, constrained metrics, and comparison JSON."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "constrained_predictions.csv"
    metrics_path = output_dir / "constrained_metrics.json"
    comparison_path = output_dir / "comparison.json"

    fieldnames = [
        "index",
        "sample_id",
        "expected_gloss",
        "free_generation_prediction",
        "constrained_prediction",
        "constrained_correct",
        "top_scores",
        "mode",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return ConstrainedEvaluationArtifacts(
        constrained_predictions_csv=predictions_path,
        constrained_metrics_json=metrics_path,
        comparison_json=comparison_path,
    )


def write_prompt_control_evaluation_artifacts(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    out_dir: Path | str,
) -> PromptControlEvaluationArtifacts:
    """Write prompt-control predictions, metrics, comparison, and report."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    comparison_path = output_dir / "comparison.json"
    report_path = output_dir / "report.md"

    fieldnames = [
        "index",
        "sample_id",
        "expected_gloss",
        "predicted_gloss",
        "raw_model_output",
        "valid_label",
        "correct",
        "mode",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_prompt_control_report(comparison), encoding="utf-8")
    return PromptControlEvaluationArtifacts(
        predictions_csv=predictions_path,
        metrics_json=metrics_path,
        comparison_json=comparison_path,
        report_md=report_path,
    )


def _render_prompt_control_report(comparison: Mapping[str, Any]) -> str:
    baseline = comparison["baseline_free_generation"]
    prompt = comparison["prompt_control"]
    constrained = comparison.get("constrained_diagnostic")
    deltas = comparison["deltas"]
    recommendation = comparison["recommendation"]
    comparison_scope = comparison.get("comparison_scope", {})
    constrained_accuracy = (
        "n/a"
        if constrained is None
        else _format_percent(float(constrained["constrained_top1_accuracy"]))
    )
    constrained_invalid = "n/a"
    if constrained is not None:
        constrained_invalid = _format_percent(float(constrained["invalid_output_rate"]))
    constrained_samples = "n/a" if constrained is None else str(constrained.get("sample_count", ""))

    partial_note = (
        "**Note:** This is a partial/smoke comparison; run the full held-out split "
        "before making a go/no-go decision.\n\n"
        if comparison_scope.get("sample_count_matches_baseline") is False
        else ""
    )
    identity_note = (
        "**Note:** Baseline and prompt-control prediction identities do not align; "
        "this comparison is not a valid go/no-go gate.\n\n"
        if comparison_scope.get("sample_identity_matches_baseline") is False
        else ""
    )

    return (
        "# Issue #22 - Prompt-Control Output Experiment\n\n"
        "## Result\n\n"
        f"{recommendation['text']}\n\n"
        f"{partial_note}"
        f"{identity_note}"
        "## Metrics\n\n"
        "| Metric | Baseline free generation | Prompt control | Constrained diagnostic |\n"
        "|---|---:|---:|---:|\n"
        f"| Held-out samples | {baseline.get('sample_count')} | "
        f"{prompt.get('sample_count')} | {constrained_samples} |\n"
        "| Top-1 accuracy | "
        f"{_format_percent(float(baseline['strict_normalized_top1_accuracy']))} | "
        f"{_format_percent(float(prompt['strict_normalized_top1_accuracy']))} | "
        f"{constrained_accuracy} |\n"
        "| Invalid-output rate | "
        f"{_format_percent(float(baseline['invalid_output_rate']))} | "
        f"{_format_percent(float(prompt['invalid_output_rate']))} | "
        f"{constrained_invalid} |\n"
        "| Correct predictions | "
        f"{baseline.get('correct')} | {prompt.get('correct')} | "
        f"{'n/a' if constrained is None else constrained.get('correct')} |\n\n"
        "## Deltas\n\n"
        "- Prompt-control vs baseline accuracy: "
        f"{_format_signed_percent(deltas['prompt_control_vs_baseline_accuracy'])}\n"
        "- Prompt-control vs baseline invalid-output rate: "
        f"{_format_signed_percent(deltas['prompt_control_vs_baseline_invalid_output_rate'])}\n"
        "- Prompt-control vs constrained accuracy: "
        f"{_format_optional_signed_percent(deltas['prompt_control_vs_constrained_accuracy'])}\n\n"
        "## Recommendation\n\n"
        "- Enough before retraining: "
        f"{recommendation['prompt_output_control_enough_before_retraining']}\n"
    )


def _format_percent(value: float) -> str:
    return f"{value:.0%}"


def _format_signed_percent(value: float) -> str:
    return f"{value:+.0%}"


def _format_optional_signed_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return _format_signed_percent(float(value))
