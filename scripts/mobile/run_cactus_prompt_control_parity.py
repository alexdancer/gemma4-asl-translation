"""Run the Cactus prompt-control parity harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mobile.cactus_prompt_control_parity import (  # noqa: E402
    CactusPromptControlParityConfig,
    MockCactusPromptRunner,
    RealCactusEnginePromptRunner,
    run_cactus_prompt_control_parity,
)

DEFAULT_REFERENCE = "evaluation/results/prompt_control_reference/reference.json"
DEFAULT_RECORDS = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_OUT_DIR = "evaluation/results/cactus_prompt_control_parity"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Cactus prompt-control parity against Python reference.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cactus-weights", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--mock-cactus-output")
    args = parser.parse_args(argv)

    runner = (
        MockCactusPromptRunner(args.mock_cactus_output)
        if args.mock_cactus_output is not None
        else RealCactusEnginePromptRunner(args.cactus_weights)
    )
    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=args.reference,
            records_path=args.records,
            manifest_path=args.manifest,
            cactus_weights_path=args.cactus_weights,
            out_dir=args.out_dir,
            max_samples=args.max_samples,
        ),
        runner=runner,
    )

    payload = result.payload
    print(
        json.dumps(
            {
                "scope": payload["scope"],
                "runtime_mode": payload["runtime_mode"],
                "report_path": str(result.report_path),
                "sample_count": payload["summary"]["sample_count"],
                "match_count": payload["summary"]["match_count"],
                "runtime_error_count": payload["summary"]["runtime_error_count"],
                "real_cactus_parity_proven": payload["real_cactus_parity_proven"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["summary"]["all_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
