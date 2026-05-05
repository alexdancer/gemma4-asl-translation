#!/usr/bin/env python3
"""Run a real video-to-q64 smoke extraction for one known Top-50 clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.video_pose_q64_smoke import (  # noqa: E402
    VideoPoseQ64SmokeConfig,
    VideoPoseQ64SmokeError,
    result_to_dict,
    run_video_pose_q64_smoke,
)


class MockVideoPoseExtractor:
    """Deterministic extractor for lightweight CLI behavior tests."""

    def extract_from_video(self, video_path: Path, max_frames: int | None = None) -> dict[str, np.ndarray]:
        del video_path, max_frames
        return {
            "body": np.ones((4, 17, 4), dtype=np.float32),
            "left_hand": np.ones((4, 21, 4), dtype=np.float32) * 2,
            "right_hand": np.ones((4, 21, 4), dtype=np.float32) * 3,
        }

    def close(self) -> None:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", required=True, help="Known Top-50 video clip to decode and extract.")
    parser.add_argument("--sample-id", required=True, help="Known Top-50 sample_id represented by the video clip.")
    parser.add_argument("--expected-gloss", required=True, help="Expected canonical Top-50 gloss for sample-id.")
    parser.add_argument(
        "--manifest",
        default="data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json",
        help="Top-50 manifest JSON containing canonical labels.",
    )
    parser.add_argument(
        "--records",
        default="data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl",
        help="Known q64 records used to validate sample_id, expected gloss, and q64 shape.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/verification/video_pose_q64_smoke",
        help="Dedicated output directory for video-to-q64 smoke artifacts.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap passed to video extraction.")
    parser.add_argument(
        "--mock-extractor",
        action="store_true",
        help="Use a deterministic lightweight extractor for command/artifact tests.",
    )
    args = parser.parse_args(argv)

    extractor_factory = (lambda: MockVideoPoseExtractor()) if args.mock_extractor else None
    try:
        result = run_video_pose_q64_smoke(
            VideoPoseQ64SmokeConfig(
                video_path=args.video_path,
                sample_id=args.sample_id,
                expected_gloss=args.expected_gloss,
                manifest_path=args.manifest,
                records_path=args.records,
                out_dir=args.out_dir,
                max_frames=args.max_frames,
            ),
            extractor_factory=extractor_factory,
        )
    except (FileNotFoundError, KeyError, ValueError, VideoPoseQ64SmokeError) as exc:
        print(f"Video pose q64 smoke failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
