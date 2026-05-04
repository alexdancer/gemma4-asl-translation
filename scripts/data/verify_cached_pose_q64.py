#!/usr/bin/env python3
"""Verify that one cached pose archive can emit a q64 JSONL-compatible record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cached_pose_q64 import (  # noqa: E402
    CachedPoseQ64VerificationConfig,
    result_to_dict,
    verify_cached_pose_q64,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-path", required=True, help="Cached/precomputed pose .npz archive.")
    parser.add_argument("--sample-id", required=True, help="Known Top-50 sample_id represented by the pose archive.")
    parser.add_argument("--expected-gloss", required=True, help="Expected canonical Top-50 gloss for sample-id.")
    parser.add_argument(
        "--manifest",
        default="data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json",
        help="Top-50 manifest JSON containing canonical labels.",
    )
    parser.add_argument(
        "--records",
        default="data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl",
        help="Known q64 records used to validate sample_id and expected gloss.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/verification/cached_pose_q64",
        help="Dedicated output directory for cached-pose q64 verification artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        result = verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=args.pose_path,
                sample_id=args.sample_id,
                expected_gloss=args.expected_gloss,
                manifest_path=args.manifest,
                records_path=args.records,
                out_dir=args.out_dir,
            )
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Cached pose q64 verification failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
