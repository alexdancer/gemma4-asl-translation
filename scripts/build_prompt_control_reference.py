#!/usr/bin/env python3
"""Build the prompt-control free-generation reference fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.prompt_control_reference import (  # noqa: E402
    REFERENCE_MODE,
    REFERENCE_SCOPE,
    PromptControlReferenceConfig,
    build_prompt_control_reference_fixture,
    load_prompt_control_prediction_rows,
)
from src.evaluation.unsloth_asl import (  # noqa: E402
    MockPromptControlledASLGlossPredictor,
    RealUnslothASLGlossPredictor,
    build_prompt_control_prompt,
    load_manifest_labels,
)


DEFAULT_CHECKPOINT = "checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline"
DEFAULT_RECORDS = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_OUT_DIR = "evaluation/results/prompt_control_reference"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Validated Top-50 checkpoint directory.",
    )
    parser.add_argument(
        "--records",
        default=DEFAULT_RECORDS,
        help="Held-out q64 JSONL records used for reference selection.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Top-50 manifest JSON containing canonical labels.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Dedicated directory for reference.json.",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=5,
        help="Number of demo samples to select in addition to the smoke sample.",
    )
    parser.add_argument(
        "--predictions-csv",
        default=None,
        help="Optional existing prompt-control predictions.csv to select from without rerunning.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic prompt-control mock predictions without loading the checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        labels = load_manifest_labels(args.manifest)
        prediction_rows = (
            load_prompt_control_prediction_rows(args.predictions_csv)
            if args.predictions_csv is not None
            else None
        )
        predictor = None
        if prediction_rows is None:
            if args.mock:
                predictor = MockPromptControlledASLGlossPredictor(labels)
            else:
                predictor = RealUnslothASLGlossPredictor(
                    args.checkpoint,
                    prompt_builder=lambda record: build_prompt_control_prompt(record, labels),
                )

        result = build_prompt_control_reference_fixture(
            PromptControlReferenceConfig(
                checkpoint_path=args.checkpoint,
                records_path=args.records,
                manifest_path=args.manifest,
                out_dir=args.out_dir,
                demo_count=args.demo_count,
            ),
            predictor=predictor,
            prediction_rows=prediction_rows,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Prompt-control reference fixture failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "scope": REFERENCE_SCOPE,
        "mode": REFERENCE_MODE,
        "artifact_path": str(result.artifact_path),
        "selected_count": result.payload["metadata"]["selected_count"],
        "smoke_sample_id": result.payload["records"][0]["sample_id"],
        "demo_sample_ids": [
            record["sample_id"]
            for record in result.payload["records"]
            if record["selection_role"] == "demo"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.demo_count <= 0:
        raise ValueError("--demo-count must be a positive integer.")
    if args.mock and args.predictions_csv is not None:
        raise ValueError("--mock and --predictions-csv are mutually exclusive.")
    checkpoint = Path(args.checkpoint)
    records = Path(args.records)
    manifest = Path(args.manifest)
    if not args.mock and args.predictions_csv is None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not records.exists():
        raise FileNotFoundError(f"q64 records file not found: {records}")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if args.predictions_csv is not None and not Path(args.predictions_csv).exists():
        raise FileNotFoundError(f"Prompt-control predictions CSV not found: {args.predictions_csv}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
