#!/usr/bin/env python3
"""Merge expected and runtime-predicted multi-word rows into evaluator JSONL.

Expected JSONL row schema:
{
  "sample_id": "clip_001",
  "expected_words": [...]
}

Predicted JSONL row schema supports either:
- {"sample_id": "clip_001", "predicted_words": [...]}  # preferred
- {"sample_id": "clip_001", "transcript_words": [...]} # API-style alias
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-jsonl", required=True, help="JSONL with sample_id + expected_words")
    parser.add_argument("--predicted-jsonl", required=True, help="JSONL with sample_id + predicted/transcript words")
    parser.add_argument("--out-jsonl", required=True, help="Merged evaluator JSONL output path")
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Allow expected rows with no prediction (fills predicted_words with ['__missing__']).",
    )
    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(payload)
    return rows


def _extract_words(row: dict[str, Any], *, key_candidates: Sequence[str], label: str) -> list[Any]:
    for key in key_candidates:
        value = row.get(key)
        if isinstance(value, list) and value:
            return value
    raise ValueError(f"sample_id={row.get('sample_id', '<missing>')} missing non-empty {label} list")


def _index_by_sample_id(rows: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{source} row missing non-empty sample_id")
        if sample_id in indexed:
            raise ValueError(f"{source} duplicate sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def build_multiword_eval_rows(
    expected_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]],
    *,
    allow_missing_predictions: bool,
) -> list[dict[str, Any]]:
    expected_by_id = _index_by_sample_id(expected_rows, source="expected")
    predicted_by_id = _index_by_sample_id(predicted_rows, source="predicted")

    merged: list[dict[str, Any]] = []
    for sample_id, expected_row in expected_by_id.items():
        expected_words = _extract_words(
            expected_row,
            key_candidates=("expected_words",),
            label="expected_words",
        )

        predicted_row = predicted_by_id.get(sample_id)
        if predicted_row is None:
            if not allow_missing_predictions:
                raise ValueError(f"missing prediction for sample_id={sample_id}")
            predicted_words: list[Any] = ["__missing__"]
        else:
            predicted_words = _extract_words(
                predicted_row,
                key_candidates=("predicted_words", "transcript_words", "words"),
                label="predicted_words/transcript_words/words",
            )

        merged.append(
            {
                "sample_id": sample_id,
                "expected_words": expected_words,
                "predicted_words": predicted_words,
            }
        )

    return merged


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_path = Path(args.expected_jsonl)
    predicted_path = Path(args.predicted_jsonl)
    out_path = Path(args.out_jsonl)

    try:
        expected_rows = _load_jsonl(expected_path)
        predicted_rows = _load_jsonl(predicted_path)
        merged = build_multiword_eval_rows(
            expected_rows,
            predicted_rows,
            allow_missing_predictions=args.allow_missing_predictions,
        )
        _write_jsonl(out_path, merged)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"build_multiword_eval_rows failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "expected_rows": len(expected_rows),
        "predicted_rows": len(predicted_rows),
        "merged_rows": len(merged),
        "out_jsonl": str(out_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
