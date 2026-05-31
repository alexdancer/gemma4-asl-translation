#!/usr/bin/env python3
"""
Extract deterministic frame sequences for phase-1 manifests, then zip outputs.

This mirrors notebook behavior for Zahid phase-1:
- 30 evenly sampled frames per video
- 448x448 JPG output
- frame files: frame_000.jpg ... frame_029.jpg

Expected manifest rows (train/val/test.jsonl):
- sample_id
- split
- video_relpath

Usage:
  python3 scripts/extract_phase1_frames_and_zip.py \
    --manifest-dir data/phase1_zahid_top50 \
    --videos-dir data/phase1_zahid_top50/videos_cache \
    --frames-dir data/phase1_zahid_top50/frames_30x448 \
    --num-frames 30 \
    --size 448 \
    --zip-videos \
    --zip-frames
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python. Install with: python3 -m pip install opencv-python"
    ) from exc

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def extract_frames(video_path: Path, out_dir: Path, *, n_frames: int, size: tuple[int, int], overwrite: bool) -> list[Path]:
    expected = [out_dir / f"frame_{i:03d}.jpg" for i in range(n_frames)]
    if not overwrite and all(p.exists() for p in expected):
        return expected

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("frame_*.jpg"):
        p.unlink()

    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise RuntimeError(f"No readable frames in video: {video_path}")

        indices = np.linspace(0, frame_count - 1, num=n_frames).round().astype(int).tolist()
        written: list[Path] = []
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed reading frame index {idx} from {video_path}")
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            out = out_dir / f"frame_{i:03d}.jpg"
            if not cv2.imwrite(str(out), resized):
                raise RuntimeError(f"Failed writing frame: {out}")
            written.append(out)
    finally:
        cap.release()

    if len(written) != n_frames:
        raise RuntimeError(f"Expected {n_frames} frames, wrote {len(written)} for {video_path}")
    return written


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--videos-dir", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--zip-videos", action="store_true")
    p.add_argument("--zip-frames", action="store_true")
    p.add_argument("--videos-zip-path", type=Path, default=None)
    p.add_argument("--frames-zip-path", type=Path, default=None)
    args = p.parse_args()

    if args.num_frames <= 0:
        p.error("--num-frames must be positive")
    if args.size <= 0:
        p.error("--size must be positive")
    return args


def main() -> None:
    args = parse_args()

    manifests = [args.manifest_dir / "train.jsonl", args.manifest_dir / "val.jsonl", args.manifest_dir / "test.jsonl"]
    for m in manifests:
        if not m.exists():
            raise FileNotFoundError(f"Missing manifest: {m}")
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory: {args.videos_dir}")

    all_rows: list[dict] = []
    for m in manifests:
        all_rows.extend(read_jsonl(m))

    seen: set[str] = set()
    failures: list[dict] = []
    done = 0

    for row in all_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        split = str(row.get("split", "")).strip() or "unknown"
        rel = str(row.get("video_relpath", "")).strip()

        if not sample_id:
            failures.append({"reason": "missing_sample_id", "row": row})
            continue
        if not rel:
            failures.append({"sample_id": sample_id, "reason": "missing_video_relpath"})
            continue
        if sample_id in seen:
            continue
        seen.add(sample_id)

        local_video = args.videos_dir / rel
        if not local_video.exists():
            failures.append({"sample_id": sample_id, "video_relpath": rel, "reason": "video_missing", "path": str(local_video)})
            continue

        sample_dir = args.frames_dir / split / sample_id
        try:
            extract_frames(local_video, sample_dir, n_frames=args.num_frames, size=(args.size, args.size), overwrite=args.overwrite)
            done += 1
            if done % 50 == 0:
                print(f"Extracted {done} samples")
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sample_id, "video_relpath": rel, "reason": f"extract_failed:{exc}"})

    summary = {
        "samples_total": len(seen),
        "samples_ok": done,
        "samples_failed": len(failures),
        "frames_dir": str(args.frames_dir),
    }
    print(json.dumps(summary, indent=2))

    fail_path = args.frames_dir / "extraction_failures.jsonl"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    with fail_path.open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if failures:
        fail_rate = len(failures) / max(len(seen), 1)
        print(f"⚠️ failures written to: {fail_path} (rate={fail_rate:.2%})")

    if args.zip_videos:
        videos_zip = args.videos_zip_path or (args.videos_dir.parent / "videos_cache.zip")
        print(f"Zipping videos: {args.videos_dir} -> {videos_zip}")
        zip_dir(args.videos_dir, videos_zip)
        print(f"✅ wrote {videos_zip}")

    if args.zip_frames:
        frames_zip = args.frames_zip_path or (args.frames_dir.parent / "frames_30x448.zip")
        print(f"Zipping frames: {args.frames_dir} -> {frames_zip}")
        zip_dir(args.frames_dir, frames_zip)
        print(f"✅ wrote {frames_zip}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
