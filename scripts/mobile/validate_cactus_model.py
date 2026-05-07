#!/usr/bin/env python3
"""Validate converted Cactus weights with prompt-control parity smoke checks.

This script is intended to run immediately after `cactus convert ...` finishes.
It performs:
1) Weights directory existence check
2) Real Cactus Engine prompt-control parity run
3) Readiness summary write + pass/fail exit code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mobile.cactus_prompt_control_parity import (  # noqa: E402
    CactusPromptControlParityConfig,
    RealCactusEnginePromptRunner,
    run_cactus_prompt_control_parity,
)

DEFAULT_REFERENCE = Path("evaluation/results/prompt_control_reference/reference.json")
DEFAULT_RECORDS = Path("data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl")
DEFAULT_MANIFEST = Path("data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json")
DEFAULT_OUT_DIR = Path("evaluation/results/cactus_model_validation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate converted Cactus weights via real prompt-control parity checks.")
    parser.add_argument("--cactus-weights", type=Path, required=True, help="Path to converted Cactus weights directory.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Number of prompt-control reference samples to validate (default: smoke=1).",
    )
    return parser.parse_args()


def _build_readiness(payload: dict[str, Any], weights_path: Path) -> dict[str, Any]:
    summary = payload["summary"]
    runtime_mode = payload.get("runtime_mode")
    readiness_passed = (
        runtime_mode == "cactus_engine"
        and payload.get("real_cactus_parity_proven") is True
        and summary.get("sample_count", 0) > 0
        and summary.get("runtime_error_count", 1) == 0
        and summary.get("all_matches") is True
    )
    return {
        "scope": "cactus_model_validation",
        "weights_path": str(weights_path),
        "runtime_mode": runtime_mode,
        "sample_count": summary.get("sample_count", 0),
        "match_count": summary.get("match_count", 0),
        "runtime_error_count": summary.get("runtime_error_count", 0),
        "all_matches": bool(summary.get("all_matches", False)),
        "real_cactus_parity_proven": bool(payload.get("real_cactus_parity_proven", False)),
        "readiness_passed": readiness_passed,
    }


def main() -> int:
    args = _parse_args()

    weights = args.cactus_weights.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights.is_dir():
        payload = {
            "scope": "cactus_model_validation",
            "readiness_passed": False,
            "error": f"Converted Cactus weights directory not found: {weights}",
        }
        readiness_path = out_dir / "readiness.json"
        readiness_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    parity_result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=args.reference,
            records_path=args.records,
            manifest_path=args.manifest,
            cactus_weights_path=weights,
            out_dir=out_dir,
            max_samples=args.max_samples,
        ),
        runner=RealCactusEnginePromptRunner(weights),
    )

    readiness = _build_readiness(parity_result.payload, weights)
    readiness_path = out_dir / "readiness.json"
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "readiness_path": str(readiness_path),
                "parity_report_path": str(parity_result.report_path),
                **readiness,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if readiness["readiness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
