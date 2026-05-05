#!/usr/bin/env python3
"""Run the demo-scoped prerecorded Top-50 q64 checkpoint path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.demo.prerecorded_q64 import DEMO_SCOPE, PrerecordedQ64DemoConfig, run_prerecorded_q64_demo
from src.evaluation.unsloth_asl import MockASLGlossPredictor, load_manifest_labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Validated Top-50 checkpoint directory.")
    parser.add_argument("--records", required=True, help="Known-good q64 JSONL input records.")
    parser.add_argument("--manifest", required=True, help="Top-50 manifest JSON containing labels.")
    parser.add_argument("--record-id", required=True, help="Known-good q64 sample_id to run.")
    parser.add_argument(
        "--out-dir",
        default="evaluation/results/prerecorded_q64_demo",
        help="Directory for prerecorded q64 demo readiness artifacts.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic q64 contract predictions without loading the checkpoint.",
    )
    args = parser.parse_args(argv)

    predictor = MockASLGlossPredictor(load_manifest_labels(args.manifest)) if args.mock else None
    try:
        result = run_prerecorded_q64_demo(
            PrerecordedQ64DemoConfig(
                checkpoint_path=args.checkpoint,
                records_path=args.records,
                manifest_path=args.manifest,
                record_id=args.record_id,
                out_dir=args.out_dir,
            ),
            predictor=predictor,
        )
    except (FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        print(f"Prerecorded q64 demo checkpoint run failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_serialize_summary(result), indent=2, sort_keys=True))
    return 0


def _serialize_summary(result) -> dict[str, object]:
    return {
        "scope": DEMO_SCOPE,
        "model_path": result.model_path,
        "input_record_id": result.input_record_id,
        "inference_mode": result.inference_mode,
        "raw_prediction": result.raw_prediction,
        "normalized_gloss": result.normalized_gloss,
        "visible_gloss": result.output.display_text,
        "status": result.output.status,
        "valid_label": result.valid_label,
        "confidence": None,
        "artifact_path": str(result.artifact_path),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
