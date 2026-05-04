"""Run fallback B: precomputed output replay emergency mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.demo.fallback_b import ReplayRunConfig, ReplayRunResult, run_precomputed_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-path",
        required=True,
        help="JSON replay script containing precomputed demo outputs.",
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        help="Honor replay step timing. Omit for instant activation tests.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_false",
        dest="sleep",
        help="Play outputs immediately.",
    )
    parser.set_defaults(sleep=False)
    args = parser.parse_args(argv)

    result = run_precomputed_replay(
        ReplayRunConfig(
            replay_path=args.replay_path,
            sleep=args.sleep,
        )
    )
    print(json.dumps(_serialize_result(result), indent=2, sort_keys=True))
    return 0


def _serialize_result(result: ReplayRunResult) -> dict[str, object]:
    return {
        "mode": result.mode,
        "scenario": result.scenario,
        "replay_path": result.replay_path,
        "observation": result.observation,
        "elapsed_ms": result.elapsed_ms,
        "steps": [
            {
                "at_ms": step.at_ms,
                "status": step.output.status,
                "display_text": step.output.display_text,
                "prediction": step.output.prediction,
                "confidence": step.output.confidence,
                "is_uncertain": step.output.is_uncertain,
                "latency_ms": step.output.latency_ms,
            }
            for step in result.steps
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
