#!/usr/bin/env python3
"""Run the two-step multi-word eval pipeline in one command.

Steps:
1) build_multiword_eval_rows.py semantics (merge expected + predicted)
2) evaluate_multiword_asl.py semantics (score merged rows)
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

from scripts.evaluation.build_multiword_eval_rows import build_multiword_eval_rows
from src.evaluation.multiword_asl import (
    evaluate_multiword_rows,
    load_multiword_predictions_jsonl,
    write_multiword_evaluation_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-jsonl", required=True)
    parser.add_argument("--predicted-jsonl", required=True)
    parser.add_argument("--merged-out-jsonl", required=True)
    parser.add_argument("--eval-out-dir", required=True)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be JSON object")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_rows = _load_jsonl(Path(args.expected_jsonl))
    predicted_rows = _load_jsonl(Path(args.predicted_jsonl))

    merged_rows = build_multiword_eval_rows(
        expected_rows,
        predicted_rows,
        allow_missing_predictions=args.allow_missing_predictions,
    )
    merged_out = Path(args.merged_out_jsonl)
    _write_jsonl(merged_out, merged_rows)

    scored_rows_input = load_multiword_predictions_jsonl(merged_out)
    rows, metrics = evaluate_multiword_rows(scored_rows_input)
    artifacts = write_multiword_evaluation_artifacts(rows, metrics, args.eval_out_dir)

    print(
        json.dumps(
            {
                "merged_rows": len(merged_rows),
                "merged_out_jsonl": str(merged_out),
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
