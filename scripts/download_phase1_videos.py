#!/usr/bin/env python3
"""
Download all phase-1 Zahid videos referenced by train/val/test manifests.

Usage:
  python3 scripts/download_phase1_videos.py \
    --manifest-dir data/phase1_zahid_top50 \
    --out-dir data/phase1_zahid_top50/videos_cache \
    --repo ZahidYasinMittha/American-Sign-Language-Dataset \
    --revision 3a6226f9c8de394a07b6c2e01158f6291897f97b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


def read_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    args = p.parse_args()

    manifests = [
        args.manifest_dir / "train.jsonl",
        args.manifest_dir / "val.jsonl",
        args.manifest_dir / "test.jsonl",
    ]
    for m in manifests:
        if not m.exists():
            raise FileNotFoundError(f"Missing manifest: {m}")

    relpaths: set[str] = set()
    for m in manifests:
        for row in read_jsonl(m):
            rel = str(row.get("video_relpath", "")).strip()
            if not rel:
                raise ValueError(f"Row missing video_relpath in {m}: {row.get('sample_id')}")
            relpaths.add(rel)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = len(relpaths)
    print(f"Found {total} unique videos to download")

    done = 0
    for rel in sorted(relpaths):
        hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            filename=rel,
            revision=args.revision,
            local_dir=str(args.out_dir),
        )
        done += 1
        if done % 25 == 0 or done == total:
            print(f"Downloaded {done}/{total}")

    print("✅ Download complete")
    print(f"Saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
