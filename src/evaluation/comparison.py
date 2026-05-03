"""Comparison modules for constrained and prompt-control diagnostics."""

from __future__ import annotations

from typing import Any, Mapping


def build_constrained_comparison(
    constrained_metrics: Mapping[str, Any],
    free_generation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare constrained diagnostic metrics to strict free-generation metrics."""

    required_free_metrics = ("strict_normalized_top1_accuracy", "invalid_output_rate")
    missing_free_metrics = [
        key for key in required_free_metrics if key not in free_generation_metrics
    ]
    if missing_free_metrics:
        raise ValueError(
            "Free-generation metrics missing required fields: "
            + ", ".join(sorted(missing_free_metrics))
        )
    if "constrained_top1_accuracy" not in constrained_metrics:
        raise ValueError("Constrained metrics missing required field: constrained_top1_accuracy")

    baseline_accuracy = float(free_generation_metrics["strict_normalized_top1_accuracy"])
    constrained_accuracy = float(constrained_metrics["constrained_top1_accuracy"])
    invalid_output_rate = float(free_generation_metrics["invalid_output_rate"])
    return {
        "free_generation": {
            "strict_normalized_top1_accuracy": baseline_accuracy,
            "invalid_output_rate": invalid_output_rate,
            "sample_count": free_generation_metrics.get("sample_count"),
            "correct": free_generation_metrics.get("correct"),
        },
        "constrained": {
            "constrained_top1_accuracy": constrained_accuracy,
            "sample_count": constrained_metrics.get("sample_count"),
            "correct": constrained_metrics.get("correct"),
            "mode": constrained_metrics.get("mode"),
        },
        "deltas": {
            "top1_accuracy": round(constrained_accuracy - baseline_accuracy, 6),
            "invalid_output_rate": round(0.0 - invalid_output_rate, 6),
        },
    }


def issue_22_activation_status(
    free_generation_metrics: Mapping[str, Any],
    constrained_comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return whether #21 evidence activates the prompt-control experiment."""

    baseline_accuracy = float(
        free_generation_metrics.get("strict_normalized_top1_accuracy", 0.0)
    )
    invalid_output_rate = float(free_generation_metrics.get("invalid_output_rate", 0.0))
    constrained_accuracy = None
    if constrained_comparison is not None:
        constrained = constrained_comparison.get("constrained", {})
        if isinstance(constrained, Mapping) and "constrained_top1_accuracy" in constrained:
            constrained_accuracy = float(constrained["constrained_top1_accuracy"])

    active = (
        invalid_output_rate > 0.0
        and constrained_accuracy is not None
        and constrained_accuracy > baseline_accuracy
    )
    reason = (
        "active: #21 shows output-control failure with usable constrained signal"
        if active
        else (
            "inactive: #21 evidence does not show both invalid free outputs "
            "and constrained improvement"
        )
    )
    return {
        "active": active,
        "reason": reason,
        "baseline_accuracy": baseline_accuracy,
        "baseline_invalid_output_rate": invalid_output_rate,
        "constrained_top1_accuracy": constrained_accuracy,
    }


def build_prompt_control_comparison(
    prompt_control_metrics: Mapping[str, Any],
    free_generation_metrics: Mapping[str, Any],
    constrained_comparison: Mapping[str, Any] | None = None,
    sample_identity_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare prompt-control free generation against baseline and constrained diagnostics."""

    required_metrics = ("strict_normalized_top1_accuracy", "invalid_output_rate")
    missing_free_metrics = [
        key for key in required_metrics if key not in free_generation_metrics
    ]
    missing_prompt_metrics = [
        key for key in required_metrics if key not in prompt_control_metrics
    ]
    if missing_free_metrics:
        raise ValueError(
            "Free-generation metrics missing required fields: "
            + ", ".join(sorted(missing_free_metrics))
        )
    if missing_prompt_metrics:
        raise ValueError(
            "Prompt-control metrics missing required fields: "
            + ", ".join(sorted(missing_prompt_metrics))
        )

    baseline_accuracy = float(free_generation_metrics["strict_normalized_top1_accuracy"])
    baseline_invalid = float(free_generation_metrics["invalid_output_rate"])
    prompt_accuracy = float(prompt_control_metrics["strict_normalized_top1_accuracy"])
    prompt_invalid = float(prompt_control_metrics["invalid_output_rate"])
    baseline_sample_count = free_generation_metrics.get("sample_count")
    prompt_sample_count = prompt_control_metrics.get("sample_count")
    sample_count_matches_baseline = (
        baseline_sample_count is not None
        and prompt_sample_count is not None
        and baseline_sample_count == prompt_sample_count
    )
    sample_identity_matches_baseline = (
        None
        if sample_identity_validation is None
        else bool(sample_identity_validation.get("matches_baseline"))
    )
    activation = issue_22_activation_status(free_generation_metrics, constrained_comparison)

    constrained_accuracy = None
    if constrained_comparison is not None:
        constrained = constrained_comparison.get("constrained", {})
        if isinstance(constrained, Mapping):
            value = constrained.get("constrained_top1_accuracy")
            constrained_accuracy = float(value) if value is not None else None

    invalid_improved = prompt_invalid < baseline_invalid
    accuracy_preserved = prompt_accuracy >= baseline_accuracy
    enough_before_retraining = bool(
        sample_count_matches_baseline
        and sample_identity_matches_baseline is not False
        and invalid_improved
        and accuracy_preserved
    )
    if not sample_count_matches_baseline:
        recommendation = (
            "Prompt/output-control changes need a full held-out run before deciding: "
            "the prompt-control sample count is missing or does not match the baseline sample count."
        )
    elif sample_identity_matches_baseline is False:
        recommendation = (
            "Prompt/output-control changes are not comparable to the baseline: "
            "the ordered sample_id and expected_gloss identities do not match."
        )
    elif enough_before_retraining:
        recommendation = (
            "Prompt/output-control changes are enough to try before retraining: "
            "invalid outputs fell while strict exact-match accuracy was preserved or improved."
        )
    else:
        recommendation = (
            "Prompt/output-control changes are not enough before retraining: "
            "they did not both reduce invalid outputs and preserve strict exact-match accuracy."
        )

    return {
        "activation": activation,
        "baseline_free_generation": {
            "strict_normalized_top1_accuracy": baseline_accuracy,
            "invalid_output_rate": baseline_invalid,
            "sample_count": baseline_sample_count,
            "correct": free_generation_metrics.get("correct"),
            "invalid": free_generation_metrics.get("invalid"),
        },
        "prompt_control": {
            "strict_normalized_top1_accuracy": prompt_accuracy,
            "invalid_output_rate": prompt_invalid,
            "sample_count": prompt_sample_count,
            "correct": prompt_control_metrics.get("correct"),
            "invalid": prompt_control_metrics.get("invalid"),
        },
        "comparison_scope": {
            "sample_count_matches_baseline": sample_count_matches_baseline,
            "baseline_sample_count": baseline_sample_count,
            "prompt_control_sample_count": prompt_sample_count,
            "sample_identity_matches_baseline": sample_identity_matches_baseline,
            "sample_identity": sample_identity_validation,
        },
        "constrained_diagnostic": (
            None
            if constrained_accuracy is None
            else {
                "constrained_top1_accuracy": constrained_accuracy,
                "invalid_output_rate": 0.0,
                "sample_count": constrained_comparison.get("constrained", {}).get("sample_count")
                if constrained_comparison
                else None,
                "correct": constrained_comparison.get("constrained", {}).get("correct")
                if constrained_comparison
                else None,
            }
        ),
        "deltas": {
            "prompt_control_vs_baseline_accuracy": round(prompt_accuracy - baseline_accuracy, 6),
            "prompt_control_vs_baseline_invalid_output_rate": round(
                prompt_invalid - baseline_invalid,
                6,
            ),
            "prompt_control_vs_constrained_accuracy": None
            if constrained_accuracy is None
            else round(prompt_accuracy - constrained_accuracy, 6),
        },
        "recommendation": {
            "prompt_output_control_enough_before_retraining": enough_before_retraining,
            "text": recommendation,
        },
    }
