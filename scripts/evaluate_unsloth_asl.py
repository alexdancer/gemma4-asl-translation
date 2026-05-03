#!/usr/bin/env python3
"""Evaluate an Unsloth ASL q64 JSONL checkpoint on held-out records."""

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
    MockASLGlossPredictor,
    RealUnslothASLGlossPredictor,
    evaluate_q64_records,
    load_manifest_labels,
    load_q64_jsonl,
    write_evaluation_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="Unsloth/PEFT checkpoint directory. Required unless --mock is set.",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="q64 JSONL test file with instruction/input/output records.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Top-50 manifest JSON containing the canonical labels list.",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/results/unsloth_asl",
        help="Directory for predictions.csv and metrics.json.",
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be a positive integer.")
    if not args.mock and not args.checkpoint:
        raise SystemExit("--checkpoint is required for real evaluation. Use --mock for contract testing.")

    labels = load_manifest_labels(args.manifest)
    records = load_q64_jsonl(args.test_file, max_samples=args.max_samples)
    if not records:
        raise SystemExit(f"No records found in {args.test_file}")

    if args.mock:
        predictor = MockASLGlossPredictor(labels)
    else:
        try:
            predictor = RealUnslothASLGlossPredictor(args.checkpoint)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    rows, metrics = evaluate_q64_records(records, predictor, labels)
    artifacts = write_evaluation_artifacts(rows, metrics, args.out_dir)

    summary = {
        "mode": predictor.mode,
        "sample_count": metrics["sample_count"],
        "strict_normalized_top1_accuracy": metrics["strict_normalized_top1_accuracy"],
        "invalid_output_rate": metrics["invalid_output_rate"],
        "predictions_csv": str(artifacts.predictions_csv),
        "metrics_json": str(artifacts.metrics_json),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
