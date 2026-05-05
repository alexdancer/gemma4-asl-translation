#!/usr/bin/env python3
"""Write Phase 2A validation decision artifacts from saved evaluation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.phase2a import Phase2AConfig, build_phase2a_report, write_phase2a_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True, type=Path, help="CSV with truth and prediction columns.")
    parser.add_argument("--train-csv", required=True, type=Path, help="Train split CSV with signer_id.")
    parser.add_argument("--val-csv", required=True, type=Path, help="Validation split CSV with signer_id.")
    parser.add_argument("--test-csv", required=True, type=Path, help="Test split CSV with signer_id.")
    parser.add_argument("--history-json", required=True, type=Path, help="Training history JSON from the training run.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for report artifacts.")
    parser.add_argument("--truth-column", default="truth", help="Ground-truth label column in predictions CSV.")
    parser.add_argument("--prediction-column", default="prediction", help="Predicted label column in predictions CSV.")
    parser.add_argument("--min-macro-f1", default=0.75, type=float, help="Macro F1 threshold to proceed.")
    parser.add_argument("--weak-class-f1", default=0.65, type=float, help="F1 threshold for weak-class reporting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions_csv)
    if args.truth_column not in predictions.columns:
        raise KeyError(f"predictions CSV missing truth column: {args.truth_column}")
    if args.prediction_column not in predictions.columns:
        raise KeyError(f"predictions CSV missing prediction column: {args.prediction_column}")

    history = json.loads(args.history_json.read_text(encoding="utf-8"))
    report = build_phase2a_report(
        truth=predictions[args.truth_column].astype(str).tolist(),
        predictions=predictions[args.prediction_column].astype(str).tolist(),
        train_split=pd.read_csv(args.train_csv),
        val_split=pd.read_csv(args.val_csv),
        test_split=pd.read_csv(args.test_csv),
        training_history=history,
        config=Phase2AConfig(
            min_macro_f1_for_phase2b=args.min_macro_f1,
            weak_class_f1_threshold=args.weak_class_f1,
        ),
    )
    outputs = write_phase2a_artifacts(report, args.output_dir)
    print(f"decision={report.decision}")
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    print(f"loss_curve_csv={outputs['loss_curve_csv']}")


if __name__ == "__main__":
    main()
