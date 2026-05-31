from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

NOTEBOOK12_MODEL_REPO = "AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora"
NOTEBOOK12_BASE_MODEL = "unsloth/gemma-4-26B-A4B-it"
NOTEBOOK12_NUM_FRAMES = 30
NOTEBOOK12_FRAME_SIZE = 448
NOTEBOOK12_MAX_CLIP_SECONDS = 5.0

PredictionStatus = Literal["ok", "uncertain"]


@dataclass(frozen=True)
class Notebook12Config:
    model_repo: str = NOTEBOOK12_MODEL_REPO
    base_model: str = NOTEBOOK12_BASE_MODEL
    num_frames: int = NOTEBOOK12_NUM_FRAMES
    frame_size: int = NOTEBOOK12_FRAME_SIZE
    max_clip_seconds: float = NOTEBOOK12_MAX_CLIP_SECONDS


@dataclass(frozen=True)
class StrictPrediction:
    status: PredictionStatus
    predicted_gloss: str | None
    raw_output: str


class DurationLimitError(ValueError):
    def __init__(self, duration_seconds: float, max_seconds: float) -> None:
        super().__init__(f"Video is {duration_seconds:.2f}s; maximum allowed is {max_seconds:.2f}s.")
        self.duration_seconds = duration_seconds
        self.max_seconds = max_seconds


DEFAULT_NOTEBOOK12_CONFIG = Notebook12Config()


def normalize_gloss(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r"^`+|`+$", "", normalized)
    normalized = re.sub(r"[^a-z0-9_ -]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_top50_labels(path: str | Path) -> tuple[str, ...]:
    labels = [normalize_gloss(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if len(labels) != 50:
        raise ValueError(f"Expected exactly 50 Top-50 labels, found {len(labels)} in {path}.")
    if len(set(labels)) != 50:
        raise ValueError(f"Top-50 labels must be unique in {path}.")
    return tuple(labels)


def strict_top50_prediction(raw_output: str, labels: Sequence[str]) -> StrictPrediction:
    normalized_labels = {normalize_gloss(label) for label in labels}
    first_line = str(raw_output).splitlines()[0] if str(raw_output).splitlines() else ""
    candidate = normalize_gloss(first_line)
    if candidate in normalized_labels:
        return StrictPrediction(status="ok", predicted_gloss=candidate, raw_output=raw_output)
    return StrictPrediction(status="uncertain", predicted_gloss=None, raw_output=raw_output)


def build_top50_prompt(labels: Sequence[str]) -> str:
    normalized_labels = [normalize_gloss(label) for label in labels]
    if not normalized_labels:
        raise ValueError("labels must not be empty.")
    return (
        "Identify the ASL sign shown across these frames. "
        "Return exactly one gloss label from the approved list.\n"
        f"Approved labels: {', '.join(normalized_labels)}"
    )


def deterministic_translation(gloss: str | None) -> str | None:
    if gloss is None:
        return None
    normalized = normalize_gloss(gloss)
    if not normalized:
        return None
    return normalized[0].upper() + normalized[1:] + "."


def validate_clip_duration(duration_seconds: float, *, max_seconds: float = NOTEBOOK12_MAX_CLIP_SECONDS) -> None:
    if duration_seconds > max_seconds:
        raise DurationLimitError(duration_seconds=duration_seconds, max_seconds=max_seconds)


def even_sample_indices(*, source_count: int, target_count: int = NOTEBOOK12_NUM_FRAMES) -> tuple[int, ...]:
    if target_count <= 0:
        raise ValueError("target_count must be positive.")
    if source_count < target_count:
        raise ValueError(f"Cannot sample {target_count} frames from only {source_count} decoded frames.")
    if target_count == 1:
        return (0,)
    return tuple(round(i * (source_count - 1) / (target_count - 1)) for i in range(target_count))


def probe_video_duration_seconds(video_path: str | Path) -> float:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required to probe uploaded videos.") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not determine duration for video: {video_path}")
    return frame_count / fps


def extract_exact_frame_images(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    num_frames: int = NOTEBOOK12_NUM_FRAMES,
    frame_size: int = NOTEBOOK12_FRAME_SIZE,
) -> list[Path]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required to extract uploaded video frames.") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    frames: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    indices = even_sample_indices(source_count=len(frames), target_count=num_frames)
    written: list[Path] = []
    for out_index, source_index in enumerate(indices):
        frame = frames[source_index]
        resized = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        out_path = output / f"frame_{out_index:03d}.jpg"
        if not cv2.imwrite(str(out_path), resized):
            raise RuntimeError(f"Failed to write extracted frame: {out_path}")
        written.append(out_path)
    return written


def build_result_row(
    *,
    video_filename: str,
    duration_seconds: float,
    expected_label: str | None,
    prediction: StrictPrediction,
    config: Notebook12Config = DEFAULT_NOTEBOOK12_CONFIG,
    timestamp: str | None = None,
) -> dict[str, Any]:
    expected = normalize_gloss(expected_label) if expected_label else None
    predicted = prediction.predicted_gloss
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "video_filename": video_filename,
        "duration_seconds": duration_seconds,
        "expected_label": expected,
        "status": prediction.status,
        "predicted_gloss": predicted,
        "translation": deterministic_translation(predicted),
        "raw_output": prediction.raw_output,
        "correct": (predicted == expected) if expected else None,
        "model": config.model_repo,
        "base_model": config.base_model,
        "num_frames": config.num_frames,
        "frame_size": config.frame_size,
    }


def append_result_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_batch_expected_labels(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "expected_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("labels.csv must contain filename and expected_label columns.")
        labels: dict[str, str] = {}
        for row in reader:
            filename = Path(str(row.get("filename") or "").strip()).name
            expected = normalize_gloss(str(row.get("expected_label") or ""))
            if filename and expected:
                labels[filename] = expected
    return labels


def list_video_files(root: str | Path) -> tuple[Path, ...]:
    video_root = Path(root)
    suffixes = {".mp4", ".mov", ".m4v"}
    return tuple(
        sorted(
            (path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes),
            key=lambda path: path.name.lower(),
        )
    )


def build_batch_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    labeled = sum(1 for row in rows if row.get("expected_label"))
    ok = sum(1 for row in rows if row.get("status") == "ok")
    correct = sum(1 for row in rows if row.get("correct") is True)
    uncertain = sum(1 for row in rows if row.get("status") == "uncertain")
    failed = sum(1 for row in rows if row.get("status") == "error")
    incorrect = sum(1 for row in rows if row.get("expected_label") and row.get("correct") is False)
    accuracy = (correct / labeled) if labeled else None
    return {
        "total": total,
        "labeled": labeled,
        "ok": ok,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "failed": failed,
        "accuracy_on_labeled": accuracy,
    }


def config_as_dict(config: Notebook12Config = DEFAULT_NOTEBOOK12_CONFIG) -> dict[str, Any]:
    return asdict(config)
