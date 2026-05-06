#!/usr/bin/env python3
"""Run issue #35 iOS tracer slice scaffold proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mobile.ios_tracer_slice import IOSTracerSliceConfig, result_to_dict, run_ios_tracer_slice

DEFAULT_RESPONSE_FIXTURE = Path("ios/ASLTracerSliceApp/ASLTracerSliceApp/Resources/local_cactus_response.json")
DEFAULT_OUTPUT = Path("artifacts/ios_tracer_slice/local_inference_result.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iOS tracer slice scaffold artifact generation.")
    parser.add_argument(
        "--response-fixture",
        type=Path,
        default=DEFAULT_RESPONSE_FIXTURE,
        help="JSON fixture that represents local Cactus response payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output artifact path.",
    )
    parser.add_argument(
        "--bundle-response-filename",
        default="local_cactus_response.json",
        help="Expected bundled resource filename used by iOS app.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_ios_tracer_slice(
        IOSTracerSliceConfig(
            local_response_fixture=args.response_fixture,
            output_path=args.output,
            bundle_response_filename=args.bundle_response_filename,
            repo_root=Path("."),
        )
    )
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0 if result.acceptance_proof_satisfied else 1


if __name__ == "__main__":
    raise SystemExit(main())
