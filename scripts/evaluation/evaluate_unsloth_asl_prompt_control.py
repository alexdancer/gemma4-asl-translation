#!/usr/bin/env python3
"""Run a stricter prompt-control free-generation experiment for ASL q64."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.unsloth_asl import (  # noqa: E402
    MockPromptControlledASLGlossPredictor,
    RealUnslothASLGlossPredictor,
    build_prompt_control_comparison,
    build_prompt_control_prompt,
    build_sample_identity_validation,
    evaluate_q64_records,
    issue_22_activation_status,
    load_constrained_comparison,
    load_free_generation_artifacts,
    load_manifest_labels,
    load_prediction_identity_rows,
    load_q64_jsonl,
    write_prompt_control_evaluation_artifacts,
)


DEFAULT_CHECKPOINT = "checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline"
DEFAULT_TEST_FILE = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_FREE_GENERATION_RESULTS_DIR = "evaluation/results/unsloth_top50_q64_full_dashboard_baseline"
DEFAULT_CONSTRAINED_RESULTS_DIR = (
    "evaluation/results/unsloth_top50_q64_full_dashboard_baseline_constrained"
)
DEFAULT_OUT_DIR = "evaluation/results/unsloth_top50_q64_full_dashboard_baseline_prompt_control"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Frozen Unsloth/PEFT checkpoint directory.",
    )
    parser.add_argument(
        "--test-file",
        default=DEFAULT_TEST_FILE,
        help="Held-out q64 JSONL test file with instruction/input/output records.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Top-50 manifest JSON containing the canonical labels list.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Separate directory for prompt-control artifacts.",
    )
    parser.add_argument(
        "--free-generation-results-dir",
        default=DEFAULT_FREE_GENERATION_RESULTS_DIR,
        help="Existing baseline free-generation results directory.",
    )
    parser.add_argument(
        "--constrained-results-dir",
        default=DEFAULT_CONSTRAINED_RESULTS_DIR,
        help="Existing constrained diagnostic results directory with comparison.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests or partial evaluation.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic placeholder predictions without loading a model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when #21 activation evidence is unavailable or inactive.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        labels = load_manifest_labels(args.manifest)
        records = load_q64_jsonl(args.test_file, max_samples=args.max_samples)
        if not records:
            raise ValueError(f"No records found in {args.test_file}")

        free_metrics, _ = load_free_generation_artifacts(args.free_generation_results_dir)
        constrained_comparison = load_constrained_comparison(args.constrained_results_dir)
        activation = issue_22_activation_status(free_metrics, constrained_comparison)
        if not activation["active"] and not args.force:
            raise ValueError(
                f"Issue #22 is not activated: {activation['reason']}. "
                "Use --force to override."
            )

        if args.mock:
            predictor = MockPromptControlledASLGlossPredictor(labels)
        else:
            prompt_builder = lambda record: build_prompt_control_prompt(record, labels)
            predictor = RealUnslothASLGlossPredictor(
                args.checkpoint,
                prompt_builder=prompt_builder,
            )

        rows, metrics = evaluate_q64_records(records, predictor, labels)
        baseline_identity_rows = load_prediction_identity_rows(
            Path(args.free_generation_results_dir) / "predictions.csv"
        )
        sample_identity = build_sample_identity_validation(baseline_identity_rows, rows)
        comparison = build_prompt_control_comparison(
            metrics,
            free_metrics,
            constrained_comparison,
            sample_identity_validation=sample_identity,
        )
        artifacts = write_prompt_control_evaluation_artifacts(
            rows,
            metrics,
            comparison,
            args.out_dir,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Prompt-control evaluation failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "mode": rows[0]["mode"] if rows else None,
        "sample_count": metrics["sample_count"],
        "strict_normalized_top1_accuracy": metrics["strict_normalized_top1_accuracy"],
        "invalid_output_rate": metrics["invalid_output_rate"],
        "baseline_free_generation_accuracy": comparison["baseline_free_generation"][
            "strict_normalized_top1_accuracy"
        ],
        "baseline_free_generation_invalid_output_rate": comparison["baseline_free_generation"][
            "invalid_output_rate"
        ],
        "prompt_control_vs_baseline_accuracy_delta": comparison["deltas"][
            "prompt_control_vs_baseline_accuracy"
        ],
        "prompt_control_vs_baseline_invalid_output_rate_delta": comparison["deltas"][
            "prompt_control_vs_baseline_invalid_output_rate"
        ],
        "recommendation": comparison["recommendation"]["text"],
        "predictions_csv": str(artifacts.predictions_csv),
        "metrics_json": str(artifacts.metrics_json),
        "comparison_json": str(artifacts.comparison_json),
        "report_md": str(artifacts.report_md),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")

    checkpoint = Path(args.checkpoint)
    test_file = Path(args.test_file)
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir).resolve()
    free_generation_results_dir = Path(args.free_generation_results_dir).resolve()
    constrained_results_dir = Path(args.constrained_results_dir).resolve()

    if not args.mock and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_file.exists():
        raise FileNotFoundError(f"q64 test file not found: {test_file}")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if out_dir == free_generation_results_dir:
        raise ValueError(
            "--out-dir must be separate from --free-generation-results-dir to avoid "
            "overwriting baseline artifacts."
        )
    if out_dir == constrained_results_dir:
        raise ValueError(
            "--out-dir must be separate from --constrained-results-dir to avoid "
            "overwriting diagnostic artifacts."
        )


if __name__ == "__main__":
    raise SystemExit(main())
