#!/usr/bin/env python3
"""Build deterministic Colab anchor + allowlist contract artifacts for Top-50 eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.colab_anchor_contract import (  # noqa: E402
    build_colab_anchor_contract,
    load_alias_map,
    write_colab_anchor_contract,
)

DEFAULT_RECORDS = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl"
DEFAULT_MANIFEST = "data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json"
DEFAULT_OUT = "evaluation/contracts/colab_top50_anchor_contract.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--alias-map",
        default=None,
        help="Optional JSON file mapping canonical label -> alias list.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        alias_map = load_alias_map(args.alias_map)
        contract = build_colab_anchor_contract(
            manifest_path=args.manifest,
            records_path=args.records,
            top_k=args.top_k,
            canonical_to_aliases=alias_map,
        )
        out_path = write_colab_anchor_contract(args.out, contract)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Failed to build colab anchor contract: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "out": str(out_path),
                "top_k": contract["top_k"],
                "anchor_count": len(contract["anchors"]),
                "allowlist_count": len(contract["allowlist"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
