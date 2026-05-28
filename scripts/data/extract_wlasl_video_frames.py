#!/usr/bin/env python3
"""Extract fixed-frame WLASL image sequences for Gemma-4 video fine-tuning.

The script consumes already-preprocessed WLASL clips by default:

    data/WLASL/start_kit/videos/{gloss}/{video_id}.mp4

It writes self-contained dataset-category folders for full WLASL and Top-50:

    data/video_finetune/full/{frames,manifest.jsonl,train.jsonl,val.jsonl,test.jsonl,failures.jsonl}
    data/video_finetune/top50/{frames,manifest.jsonl,train.jsonl,val.jsonl,test.jsonl,failures.jsonl,labels.txt}

Top-50 is computed from available local preprocessed clips per gloss so it reflects
what can actually be extracted on the current machine.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

DatasetName = Literal["full", "top50"]
VALID_SPLITS = {"train", "val", "test"}
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


@dataclass(frozen=True)
class WlaslSample:
    gloss: str
    video_id: str
    split: str
    instance_id: int | None = None
    signer_id: int | None = None
    source: str | None = None

    @property
    def sample_id(self) -> str:
        return f"{self.gloss}_{self.video_id}"


def read_wlasl_samples(metadata_path: Path) -> list[WlaslSample]:
    with metadata_path.open() as f:
        content = json.load(f)

    samples: list[WlaslSample] = []
    for entry in content:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            split = inst.get("split")
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid WLASL split for {gloss}/{inst.get('video_id')}: {split!r}")
            samples.append(
                WlaslSample(
                    gloss=gloss,
                    video_id=str(inst["video_id"]),
                    split=split,
                    instance_id=inst.get("instance_id"),
                    signer_id=inst.get("signer_id"),
                    source=inst.get("source"),
                )
            )
    return samples


def find_video_path(video_root: Path, sample: WlaslSample) -> Path | None:
    gloss_dir = video_root / sample.gloss
    for ext in VIDEO_EXTENSIONS:
        candidate = gloss_dir / f"{sample.video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def available_top_glosses(samples: Iterable[WlaslSample], video_root: Path, top_k: int) -> list[str]:
    counts: Counter[str] = Counter()
    for sample in samples:
        if find_video_path(video_root, sample) is not None:
            counts[sample.gloss] += 1
    return [gloss for gloss, _count in counts.most_common(top_k)]


def _resize_frame(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)


def _read_frame_at(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _decode_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _valid_existing_frame_sequence(output_dir: Path, num_frames: int, image_size: tuple[int, int]) -> list[Path] | None:
    paths = [output_dir / f"frame_{idx:03d}.jpg" for idx in range(num_frames)]
    if not all(path.exists() for path in paths):
        return None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            return None
        if image.shape[:2] != (image_size[1], image_size[0]):
            return None
    return paths


def extract_evenly_sampled_frames(
    video_path: Path,
    output_dir: Path,
    *,
    num_frames: int = 20,
    image_size: tuple[int, int] = (448, 448),
    overwrite: bool = False,
) -> list[Path]:
    """Extract exactly `num_frames` resized JPGs from `video_path`.

    The primary path uses OpenCV frame-count metadata with random access. If that
    path fails, the function decodes the whole clip and samples from decoded
    frames. For very short readable clips, indices may repeat so the output still
    has exactly `num_frames` images. A zero-readable-frame clip raises ValueError.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size dimensions must be positive")

    existing = _valid_existing_frame_sequence(output_dir, num_frames, image_size)
    if not overwrite and existing is not None:
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite or existing is None:
        for old_frame in output_dir.glob("frame_*.jpg"):
            old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    selected_frames: list[np.ndarray] = []
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            indices = np.linspace(0, frame_count - 1, num=num_frames).round().astype(int).tolist()
            for index in indices:
                frame = _read_frame_at(cap, index)
                if frame is None:
                    selected_frames = []
                    break
                selected_frames.append(frame)
    finally:
        cap.release()

    if len(selected_frames) != num_frames:
        decoded = _decode_all_frames(video_path)
        if not decoded:
            raise ValueError("video has zero readable frames")
        indices = np.linspace(0, len(decoded) - 1, num=num_frames).round().astype(int).tolist()
        selected_frames = [decoded[index] for index in indices]

    output_paths: list[Path] = []
    for idx, frame in enumerate(selected_frames):
        resized = _resize_frame(frame, image_size)
        frame_path = output_dir / f"frame_{idx:03d}.jpg"
        if not cv2.imwrite(str(frame_path), resized):
            raise ValueError(f"failed to write frame: {frame_path}")
        output_paths.append(frame_path)
    return output_paths


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _frame_paths_relative(frame_paths: list[Path], dataset_root: Path) -> list[str]:
    return [path.relative_to(dataset_root).as_posix() for path in frame_paths]


def _manifest_row(
    *,
    dataset_root: Path,
    frame_paths: list[Path],
    sample: WlaslSample,
    video_path: Path,
    split_source: str,
    num_frames: int,
    image_size: tuple[int, int],
) -> dict:
    return {
        "sample_id": sample.sample_id,
        "gloss": sample.gloss,
        "video_id": sample.video_id,
        "split": sample.split,
        "split_source": split_source,
        "frame_paths": _frame_paths_relative(frame_paths, dataset_root),
        "num_frames": num_frames,
        "image_size": [image_size[0], image_size[1]],
        "source_video_path": str(video_path),
        "instance_id": sample.instance_id,
        "signer_id": sample.signer_id,
        "source": sample.source,
    }


def _failure_row(dataset: DatasetName, sample: WlaslSample, reason: str, video_path: Path | None = None) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample.sample_id,
        "gloss": sample.gloss,
        "video_id": sample.video_id,
        "split": sample.split,
        "reason": reason,
        "video_path": str(video_path) if video_path is not None else None,
    }


def _copy_existing_frames(
    source_sample_root: Path,
    target_sample_root: Path,
    num_frames: int,
    image_size: tuple[int, int],
    *,
    overwrite: bool,
) -> list[Path] | None:
    source_paths = _valid_existing_frame_sequence(source_sample_root, num_frames, image_size)
    if source_paths is None:
        return None
    target_sample_root.mkdir(parents=True, exist_ok=True)
    target_paths = [target_sample_root / path.name for path in source_paths]
    target_existing = _valid_existing_frame_sequence(target_sample_root, num_frames, image_size)
    if not overwrite and target_existing is not None:
        return target_existing
    if overwrite or target_existing is None:
        for target in target_sample_root.glob("frame_*.jpg"):
            target.unlink()
    for source, target in zip(source_paths, target_paths):
        shutil.copy2(source, target)
    return target_paths


def build_dataset_category(
    *,
    dataset: DatasetName,
    samples: list[WlaslSample],
    video_root: Path,
    output_root: Path,
    selected_glosses: set[str] | None,
    num_frames: int,
    image_size: tuple[int, int],
    split_source: str = "official_wlasl_v0.3",
    max_samples: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    full_frames_root: Path | None = None,
) -> dict:
    dataset_root = output_root / dataset
    frames_root = dataset_root / "frames"
    records: list[dict] = []
    failures: list[dict] = []
    considered = 0

    for sample in samples:
        if selected_glosses is not None and sample.gloss not in selected_glosses:
            continue
        if max_samples is not None and len(records) >= max_samples:
            break
        considered += 1

        video_path = find_video_path(video_root, sample)
        if video_path is None:
            failures.append(_failure_row(dataset, sample, "missing_preprocessed_clip"))
            continue

        sample_frame_root = frames_root / sample.gloss / sample.video_id
        try:
            if dry_run:
                frame_paths = [sample_frame_root / f"frame_{idx:03d}.jpg" for idx in range(num_frames)]
            elif full_frames_root is not None:
                copied = _copy_existing_frames(
                    full_frames_root / sample.gloss / sample.video_id,
                    sample_frame_root,
                    num_frames,
                    image_size,
                    overwrite=overwrite,
                )
                frame_paths = copied if copied is not None else extract_evenly_sampled_frames(
                    video_path,
                    sample_frame_root,
                    num_frames=num_frames,
                    image_size=image_size,
                    overwrite=overwrite,
                )
            else:
                frame_paths = extract_evenly_sampled_frames(
                    video_path,
                    sample_frame_root,
                    num_frames=num_frames,
                    image_size=image_size,
                    overwrite=overwrite,
                )
        except Exception as exc:  # noqa: BLE001 - failure rows are the intended tolerance boundary.
            failures.append(_failure_row(dataset, sample, f"frame_extraction_failed:{exc}", video_path))
            continue

        records.append(
            _manifest_row(
                dataset_root=dataset_root,
                frame_paths=frame_paths,
                sample=sample,
                video_path=video_path,
                split_source=split_source,
                num_frames=num_frames,
                image_size=image_size,
            )
        )

    if not dry_run:
        dataset_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_root / "manifest.jsonl", records)
    for split in ("train", "val", "test"):
        write_jsonl(dataset_root / f"{split}.jsonl", [record for record in records if record["split"] == split])
    write_jsonl(dataset_root / "failures.jsonl", failures)

    summary = {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "considered": considered,
        "written": len(records),
        "failed": len(failures),
        "splits": {split: sum(1 for record in records if record["split"] == split) for split in ("train", "val", "test")},
    }
    (dataset_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_image_size(raw: str) -> tuple[int, int]:
    if "x" in raw.lower():
        width, height = raw.lower().split("x", 1)
        return int(width), int(height)
    size = int(raw)
    return size, size


def default_wlasl_root(repo_root: Path) -> Path:
    candidate = repo_root / "data" / "WLASL"
    return candidate if candidate.exists() else Path("data/WLASL")


def parse_args() -> argparse.Namespace:
    repo_root = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wlasl-root", type=Path, default=default_wlasl_root(repo_root))
    parser.add_argument("--metadata", type=Path, default=None, help="Defaults to <wlasl-root>/start_kit/WLASL_v0.3.json")
    parser.add_argument("--video-root", type=Path, default=None, help="Defaults to <wlasl-root>/start_kit/videos")
    parser.add_argument("--output-root", type=Path, default=Path("data/video_finetune"))
    parser.add_argument("--datasets", choices=["full", "top50", "both"], default="both")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--image-size", type=parse_image_size, default=(448, 448), help="Square size like 448 or WIDTHxHEIGHT")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum successfully written samples per selected dataset category, for smoke runs")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests only; do not decode or write image frames")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing frame_*.jpg files")
    args = parser.parse_args()
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.image_size[0] <= 0 or args.image_size[1] <= 0:
        parser.error("--image-size dimensions must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive when provided")
    return args


def main() -> None:
    args = parse_args()
    wlasl_root = args.wlasl_root
    metadata_path = args.metadata or (wlasl_root / "start_kit" / "WLASL_v0.3.json")
    video_root = args.video_root or (wlasl_root / "start_kit" / "videos")

    samples = read_wlasl_samples(metadata_path)
    selected = ["full", "top50"] if args.datasets == "both" else [args.datasets]
    top_glosses = available_top_glosses(samples, video_root, args.top_k) if "top50" in selected else []

    summaries = []
    if "full" in selected:
        summaries.append(
            build_dataset_category(
                dataset="full",
                samples=samples,
                video_root=video_root,
                output_root=args.output_root,
                selected_glosses=None,
                num_frames=args.num_frames,
                image_size=args.image_size,
                max_samples=args.max_samples,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        )
    if "top50" in selected:
        top50_root = args.output_root / "top50"
        top50_root.mkdir(parents=True, exist_ok=True)
        (top50_root / "labels.txt").write_text("\n".join(top_glosses) + "\n")
        summaries.append(
            build_dataset_category(
                dataset="top50",
                samples=samples,
                video_root=video_root,
                output_root=args.output_root,
                selected_glosses=set(top_glosses),
                num_frames=args.num_frames,
                image_size=args.image_size,
                max_samples=args.max_samples,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                full_frames_root=(args.output_root / "full" / "frames") if "full" in selected and not args.dry_run else None,
            )
        )

    print(json.dumps({"metadata": str(metadata_path), "video_root": str(video_root), "top_glosses": top_glosses, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
