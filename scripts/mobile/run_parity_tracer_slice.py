"""Run issue #34 parity tracer slice for one smoke sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mobile.parity_tracer_slice import (  # noqa: E402
    MockParityPromptRunner,
    ParityTracerSliceConfig,
    RealCactusParityPromptRunner,
    ReferencePythonPromptRunner,
    run_parity_tracer_slice,
)

DEFAULT_REFERENCE = "evaluation/results/prompt_control_reference/reference.json"
DEFAULT_RECORDS = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_OUT_DIR = "artifacts/cactus_tracer"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run parity tracer slice on one smoke sample (Python vs Cactus).")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cactus-weights", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--mock-python-output")
    parser.add_argument("--mock-cactus-output")
    args = parser.parse_args(argv)

    reference_payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    python_runner = (
        MockParityPromptRunner(args.mock_python_output)
        if args.mock_python_output is not None
        else ReferencePythonPromptRunner.from_reference_payload(reference_payload)
    )
    cactus_runner = (
        MockParityPromptRunner(args.mock_cactus_output)
        if args.mock_cactus_output is not None
        else RealCactusParityPromptRunner(args.cactus_weights)
    )

    result = run_parity_tracer_slice(
        ParityTracerSliceConfig(
            reference_path=args.reference,
            records_path=args.records,
            manifest_path=args.manifest,
            cactus_weights_path=args.cactus_weights,
            out_dir=args.out_dir,
            max_samples=args.max_samples,
        ),
        python_runner=python_runner,
        cactus_runner=cactus_runner,
    )

    payload = result.payload
    print(
        json.dumps(
            {
                "scope": payload["scope"],
                "report_path": str(result.report_path),
                "sample_count": payload["summary"]["sample_count"],
                "match_count": payload["summary"]["match_count"],
                "runtime_error_count": payload["summary"]["runtime_error_count"],
                "python_runtime_mode": payload["python_runtime_mode"],
                "cactus_runtime_mode": payload["cactus_runtime_mode"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["summary"]["all_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
