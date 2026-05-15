#!/usr/bin/env python3
"""Evaluate issue #90 Colab gatekeeper metrics from notebook prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.colab_gatekeeper import evaluate_colab_gatekeeper  # noqa: E402

DEFAULT_CONTRACT = "evaluation/contracts/colab_top50_anchor_contract.json"
DEFAULT_PREDICTIONS = "evaluation/results/colab_issue89_predictions.csv"
DEFAULT_OUT = "evaluation/results/colab_issue90_gatekeeper.json"


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_predictions_csv(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "expected_gloss",
            "first_pass_raw",
            "first_pass_valid",
            "retry_used",
            "final_gloss",
            "final_valid",
            "correct",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"prediction csv missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "expected_gloss": row.get("expected_gloss", ""),
                    "first_pass_raw": row.get("first_pass_raw", ""),
                    "first_pass_valid": _as_bool(row.get("first_pass_valid", "")),
                    "retry_used": _as_bool(row.get("retry_used", "")),
                    "retry_raw": row.get("retry_raw", ""),
                    "retry_valid": _as_bool(row.get("retry_valid", "")),
                    "final_gloss": row.get("final_gloss", ""),
                    "final_valid": _as_bool(row.get("final_valid", "")),
                    "correct": _as_bool(row.get("correct", "")),
                }
            )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--per-anchor-threshold", type=float, default=0.70)
    parser.add_argument("--collapse-threshold", type=float, default=0.40)
    parser.add_argument("--anchor-min-samples", type=int, default=3)
    parser.add_argument("--anchor-max-samples", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        rows = _load_predictions_csv(args.predictions)
        report = evaluate_colab_gatekeeper(
            contract=contract,
            prediction_rows=rows,
            per_anchor_threshold=args.per_anchor_threshold,
            collapse_threshold=args.collapse_threshold,
            anchor_min_samples=args.anchor_min_samples,
            anchor_max_samples=args.anchor_max_samples,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"Failed to evaluate Colab gatekeeper: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "final_pass": report["final_after_retry"]["pass"],
                "final_reasons": report["final_after_retry"]["reasons"],
                "decision_changed": report["retry_effect"]["decision_changed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
