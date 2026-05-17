from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.data.extract_wlasl_video_frames import (
    available_top_glosses,
    build_dataset_category,
    extract_evenly_sampled_frames,
    read_wlasl_samples,
)


def _write_video(path: Path, *, frame_count: int = 8, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((size[1], size[0], 3), idx * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_metadata(path: Path) -> None:
    data = [
        {
            "gloss": "book",
            "instances": [
                {"video_id": "book_train", "split": "train", "instance_id": 1, "signer_id": 10, "source": "fixture"},
                {"video_id": "book_val", "split": "val", "instance_id": 2, "signer_id": 11, "source": "fixture"},
            ],
        },
        {
            "gloss": "drink",
            "instances": [
                {"video_id": "drink_test", "split": "test", "instance_id": 3, "signer_id": 12, "source": "fixture"},
            ],
        },
        {
            "gloss": "missing",
            "instances": [
                {"video_id": "missing_train", "split": "train", "instance_id": 4, "signer_id": 13, "source": "fixture"},
            ],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_extract_evenly_sampled_frames_writes_exact_count_and_size(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_video(video, frame_count=5, size=(40, 30))

    frames = extract_evenly_sampled_frames(video, tmp_path / "frames", num_frames=20, image_size=(16, 16))

    assert len(frames) == 20
    assert frames[0].name == "frame_000.jpg"
    assert frames[-1].name == "frame_019.jpg"
    image = cv2.imread(str(frames[0]))
    assert image.shape[:2] == (16, 16)


def test_extract_evenly_sampled_frames_regenerates_stale_wrong_size_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    _write_video(video, frame_count=5, size=(40, 30))
    for idx in range(4):
        cv2.imwrite(str(frame_dir / f"frame_{idx:03d}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))

    frames = extract_evenly_sampled_frames(video, frame_dir, num_frames=4, image_size=(16, 16), overwrite=False)

    assert len(frames) == 4
    assert cv2.imread(str(frames[0])).shape[:2] == (16, 16)


def test_build_dataset_category_uses_official_splits_and_logs_missing(tmp_path: Path) -> None:
    metadata = tmp_path / "WLASL_v0.3.json"
    videos = tmp_path / "videos"
    output = tmp_path / "video_finetune"
    _write_metadata(metadata)
    _write_video(videos / "book" / "book_train.mp4")
    _write_video(videos / "book" / "book_val.mp4")
    _write_video(videos / "drink" / "drink_test.mp4")

    samples = read_wlasl_samples(metadata)
    summary = build_dataset_category(
        dataset="full",
        samples=samples,
        video_root=videos,
        output_root=output,
        selected_glosses=None,
        num_frames=4,
        image_size=(16, 16),
    )

    assert summary["written"] == 3
    assert summary["failed"] == 1
    assert summary["splits"] == {"train": 1, "val": 1, "test": 1}

    manifest = _read_jsonl(output / "full" / "manifest.jsonl")
    first = manifest[0]
    assert first["split_source"] == "official_wlasl_v0.3"
    assert first["frame_paths"] == [
        "frames/book/book_train/frame_000.jpg",
        "frames/book/book_train/frame_001.jpg",
        "frames/book/book_train/frame_002.jpg",
        "frames/book/book_train/frame_003.jpg",
    ]
    assert all((output / "full" / path).exists() for path in first["frame_paths"])

    failures = _read_jsonl(output / "full" / "failures.jsonl")
    assert failures == [
        {
            "dataset": "full",
            "sample_id": "missing_missing_train",
            "gloss": "missing",
            "video_id": "missing_train",
            "split": "train",
            "reason": "missing_preprocessed_clip",
            "video_path": None,
        }
    ]


def test_top50_is_computed_from_available_local_clips(tmp_path: Path) -> None:
    metadata = tmp_path / "WLASL_v0.3.json"
    videos = tmp_path / "videos"
    _write_metadata(metadata)
    _write_video(videos / "book" / "book_train.mp4")
    _write_video(videos / "book" / "book_val.mp4")
    _write_video(videos / "drink" / "drink_test.mp4")

    samples = read_wlasl_samples(metadata)

    assert available_top_glosses(samples, videos, top_k=2) == ["book", "drink"]
    assert "missing" not in available_top_glosses(samples, videos, top_k=3)


def test_top50_category_can_copy_from_full_frames(tmp_path: Path) -> None:
    metadata = tmp_path / "WLASL_v0.3.json"
    videos = tmp_path / "videos"
    output = tmp_path / "video_finetune"
    _write_metadata(metadata)
    _write_video(videos / "book" / "book_train.mp4")
    _write_video(videos / "book" / "book_val.mp4")
    _write_video(videos / "drink" / "drink_test.mp4")
    samples = read_wlasl_samples(metadata)

    build_dataset_category(
        dataset="full",
        samples=samples,
        video_root=videos,
        output_root=output,
        selected_glosses=None,
        num_frames=4,
        image_size=(16, 16),
    )
    summary = build_dataset_category(
        dataset="top50",
        samples=samples,
        video_root=videos,
        output_root=output,
        selected_glosses={"book"},
        num_frames=4,
        image_size=(16, 16),
        full_frames_root=output / "full" / "frames",
    )

    assert summary["written"] == 2
    top50_manifest = _read_jsonl(output / "top50" / "manifest.jsonl")
    assert {row["gloss"] for row in top50_manifest} == {"book"}
    assert (output / "top50" / "frames" / "book" / "book_train" / "frame_000.jpg").exists()
