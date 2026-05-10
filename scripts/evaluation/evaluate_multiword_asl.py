#!/usr/bin/env python3
"""Evaluate multi-word ASL predictions from JSONL.

Input JSONL schema per line:
{
  "sample_id": "...",
  "expected_words": ["hello", {"word":"world","start_ms":100,"end_ms":200}],
  "predicted_words": [...]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.multiword_asl import (
    evaluate_multiword_rows,
    load_multiword_predictions_jsonl,
    write_multiword_evaluation_artifacts,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-word ASL model predictions.")
    parser.add_argument("--input-jsonl", required=True, help="Path to JSONL with expected/predicted multi-word rows.")
    parser.add_argument("--out-dir", default="evaluation/results/multiword_asl", help="Output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_multiword_predictions_jsonl(args.input_jsonl)
    rows, metrics = evaluate_multiword_rows(records)
    artifacts = write_multiword_evaluation_artifacts(rows, metrics, args.out_dir)

    print(
        json.dumps(
            {
                "sample_count": metrics["sample_count"],
                "word_error_rate": metrics["word_error_rate"],
                "exact_sequence_accuracy": metrics["exact_sequence_accuracy"],
                "timestamp_boundary_mae_ms": metrics["timestamp_boundary_mae_ms"],
                "predictions_csv": str(artifacts.predictions_csv),
                "metrics_json": str(artifacts.metrics_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
