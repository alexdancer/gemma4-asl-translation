#!/usr/bin/env python3
"""Run issue #32 Cactus tracer slice workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mobile.cactus_tracer_slice import TracerSliceConfig, result_to_dict, run_cactus_tracer_slice

DEFAULT_CHECKPOINT = Path("checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline")
DEFAULT_PROMPT = "You are an ASL gloss classifier. Return exactly one uppercase gloss label and no extra text."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze baseline, convert Cactus weights v1, and run one local completion.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Frozen baseline checkpoint directory.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/cactus_tracer"), help="Artifact root directory.")
    parser.add_argument("--conversion-output-version", default="v1", help="Version tag for converted weights output.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt passed to the local completion step.")
    parser.add_argument("--git-sha", default=None, help="Optional git SHA override for frozen baseline metadata.")
    parser.add_argument(
        "--no-real-export",
        action="store_true",
        help="Disable real Cactus export and force deterministic conversion fallback artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TracerSliceConfig(
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        conversion_output_version=args.conversion_output_version,
        git_sha=args.git_sha,
        prompt=args.prompt,
        allow_real_export=not args.no_real_export,
        repo_root=Path("."),
    )
    result = run_cactus_tracer_slice(config)
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
