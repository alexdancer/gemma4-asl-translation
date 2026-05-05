#!/usr/bin/env python3
"""Run demo-safe constrained Top-50 q64 inference for one record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.constrained_top50 import (  # noqa: E402
    DEMO_SCOPE,
    ConstrainedTop50DemoConfig,
    run_constrained_top50_demo,
    select_q64_record_by_sample_id,
)
from src.evaluation.unsloth_asl import (  # noqa: E402
    MockConstrainedGlossScorer,
    load_manifest_labels,
    load_q64_jsonl,
    normalize_gloss,
)


DEFAULT_CHECKPOINT = "checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline"
DEFAULT_RECORDS = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_OUT_DIR = "evaluation/results/demo_safe_constrained_top50"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Top-50 checkpoint directory.")
    parser.add_argument("--records", default=DEFAULT_RECORDS, help="q64 JSONL input records.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Top-50 manifest JSON containing labels.")
    parser.add_argument("--record-id", required=True, help="q64 sample_id to run.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory for demo-safe constrained readiness artifacts.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidate scores to include.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic constrained scores without loading the checkpoint.",
    )
    args = parser.parse_args(argv)

    try:
        scorer = _build_mock_scorer(args.records, args.manifest, args.record_id) if args.mock else None
        result = run_constrained_top50_demo(
            ConstrainedTop50DemoConfig(
                checkpoint_path=args.checkpoint,
                records_path=args.records,
                manifest_path=args.manifest,
                record_id=args.record_id,
                out_dir=args.out_dir,
                top_k=args.top_k,
            ),
            scorer=scorer,
        )
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"Demo-safe constrained Top-50 run failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(_serialize_summary(result), indent=2, sort_keys=True))
    return 0


def _build_mock_scorer(
    records_path: Path | str,
    manifest_path: Path | str,
    record_id: str,
) -> MockConstrainedGlossScorer:
    labels = load_manifest_labels(manifest_path)
    record = select_q64_record_by_sample_id(load_q64_jsonl(records_path), record_id)
    expected = normalize_gloss(str(record.get("output", "")))
    scores = {label: float(len(labels) - index) for index, label in enumerate(labels)}
    if expected in scores:
        scores[expected] = float(len(labels) + 1)
    return MockConstrainedGlossScorer(scores)


def _serialize_summary(result) -> dict[str, object]:
    return {
        "scope": DEMO_SCOPE,
        "claims": result.claims,
        "model_path": result.model_path,
        "input_record_id": result.input_record_id,
        "inference_mode": result.inference_mode,
        "selected_label": result.selected_label,
        "best_label": result.best_label,
        "expected_gloss": result.expected_gloss,
        "correct": result.correct,
        "visible_gloss": result.output.display_text,
        "status": result.output.status,
        "confidence": None,
        "confidence_available": False,
        "top_candidates": [
            {"label": candidate.label, "score": candidate.score}
            for candidate in result.top_candidates
        ],
        "constrained_metadata": result.constrained_metadata,
        "artifact_path": str(result.artifact_path),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
