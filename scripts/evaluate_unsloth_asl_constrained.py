#!/usr/bin/env python3
"""Run constrained Top-50 diagnostics for an Unsloth ASL q64 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.unsloth_asl import (  # noqa: E402
    MockConstrainedGlossScorer,
    RealUnslothASLGlossPredictor,
    build_constrained_comparison,
    evaluate_q64_records_constrained,
    load_free_generation_artifacts,
    load_manifest_labels,
    load_q64_jsonl,
    write_constrained_evaluation_artifacts,
)


DEFAULT_CHECKPOINT = "checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline"
DEFAULT_TEST_FILE = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_FREE_GENERATION_RESULTS_DIR = "evaluation/results/unsloth_top50_q64_full_dashboard_baseline"
DEFAULT_OUT_DIR = "evaluation/results/unsloth_top50_q64_full_dashboard_baseline_constrained"


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
        help="Separate directory for constrained diagnostic artifacts.",
    )
    parser.add_argument(
        "--free-generation-results-dir",
        default=DEFAULT_FREE_GENERATION_RESULTS_DIR,
        help="Existing free-generation results directory containing metrics.json and predictions.csv.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests or partial diagnostic runs.",
    )
    parser.add_argument(
        "--top-scores",
        type=int,
        default=5,
        help="Number of top candidate scores to include as JSON text in each CSV row.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic placeholder candidate scores without loading a model.",
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

        free_metrics, free_predictions = load_free_generation_artifacts(args.free_generation_results_dir)
        if args.mock:
            scorer = MockConstrainedGlossScorer(
                {label: float(len(labels) - index) for index, label in enumerate(labels)}
            )
        else:
            scorer = RealUnslothASLGlossPredictor(args.checkpoint)

        rows, metrics = evaluate_q64_records_constrained(
            records,
            scorer,
            labels,
            free_generation_predictions=free_predictions,
            top_score_count=args.top_scores,
        )
        comparison = build_constrained_comparison(metrics, free_metrics)
        artifacts = write_constrained_evaluation_artifacts(rows, metrics, comparison, args.out_dir)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Constrained evaluation failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "mode": metrics["mode"],
        "sample_count": metrics["sample_count"],
        "constrained_top1_accuracy": metrics["constrained_top1_accuracy"],
        "free_generation_strict_normalized_top1_accuracy": comparison["free_generation"][
            "strict_normalized_top1_accuracy"
        ],
        "top1_accuracy_delta": comparison["deltas"]["top1_accuracy"],
        "constrained_predictions_csv": str(artifacts.constrained_predictions_csv),
        "constrained_metrics_json": str(artifacts.constrained_metrics_json),
        "comparison_json": str(artifacts.comparison_json),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")
    if args.top_scores <= 0:
        raise ValueError("--top-scores must be a positive integer.")

    checkpoint = Path(args.checkpoint)
    test_file = Path(args.test_file)
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir).resolve()
    free_generation_results_dir = Path(args.free_generation_results_dir).resolve()

    if not args.mock and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_file.exists():
        raise FileNotFoundError(f"q64 test file not found: {test_file}")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if out_dir == free_generation_results_dir:
        raise ValueError(
            "--out-dir must be separate from --free-generation-results-dir to avoid overwriting baseline artifacts."
        )


if __name__ == "__main__":
    raise SystemExit(main())
