"""Real video to q64 JSONL smoke path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from src.data.cached_pose_q64 import (
    Q64_FULL_ENCODING,
    _flatten_pose_components,
    _load_source_record,
    _parse_reference_shape,
    _resample_feature_frames,
    _validate_q64_record,
    _validate_reference_q64_metadata,
    build_cached_pose_q64_record,
)
from src.data.q64_encoding import ALPHABET
from src.evaluation.unsloth_asl import load_manifest_labels, normalize_gloss

VIDEO_POSE_Q64_SMOKE_SCOPE = "video_pose_q64_smoke"


class VideoPoseQ64SmokeError(RuntimeError):
    """Raised when the real video-to-q64 smoke path cannot complete."""


class VideoPoseExtractor(Protocol):
    """Small interface needed by the smoke path."""

    def extract_from_video(self, video_path: Path, max_frames: int | None = None) -> dict[str, np.ndarray]:
        """Extract pose components from one video."""

    def close(self) -> None:
        """Release extractor resources."""


@dataclass(frozen=True)
class VideoPoseQ64SmokeConfig:
    """Configuration for one real video-to-q64 smoke run."""

    video_path: Path | str
    sample_id: str
    expected_gloss: str
    manifest_path: Path | str
    records_path: Path | str
    out_dir: Path | str
    max_frames: int | None = None
    encoding: str = Q64_FULL_ENCODING


@dataclass(frozen=True)
class VideoPoseQ64SmokeResult:
    """Artifact paths and generated contract record from one smoke run."""

    scope: str
    sample_id: str
    expected_gloss: str
    encoding: str
    frames: int
    features_per_frame: int
    video_path: str
    jsonl_path: Path
    report_path: Path
    record: dict[str, str]


def run_video_pose_q64_smoke(
    config: VideoPoseQ64SmokeConfig,
    *,
    extractor_factory: Callable[[], VideoPoseExtractor] | None = None,
) -> VideoPoseQ64SmokeResult:
    """Decode one known video, extract poses, and emit a q64 JSONL-compatible record."""

    sample_id = str(config.sample_id).strip()
    if not sample_id:
        raise ValueError("sample_id must not be empty.")
    if config.encoding != Q64_FULL_ENCODING:
        raise ValueError(f"Unsupported video-pose encoding: {config.encoding}")

    expected_gloss = normalize_gloss(str(config.expected_gloss))
    if not expected_gloss:
        raise ValueError("expected_gloss must not be empty.")

    labels = load_manifest_labels(config.manifest_path)
    if expected_gloss not in labels:
        raise ValueError(f"expected_gloss is not in manifest labels: {expected_gloss}")

    source_record = _load_source_record(config.records_path, sample_id)
    source_expected = normalize_gloss(str(source_record.get("output", "")))
    if source_expected != expected_gloss:
        raise ValueError(
            f"expected_gloss mismatch for {sample_id}: records file has {source_expected!r}, got {expected_gloss!r}"
        )

    video_path = Path(config.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        extractor = _build_extractor(extractor_factory)
    except ImportError as exc:
        raise VideoPoseQ64SmokeError(
            "Video pose q64 smoke failed before extraction. Install MediaPipe/OpenCV dependencies "
            "from requirements.txt and verify the environment can import mediapipe and cv2."
        ) from exc
    try:
        try:
            components = extractor.extract_from_video(video_path, max_frames=config.max_frames)
        except (OSError, RuntimeError, ValueError) as exc:
            raise VideoPoseQ64SmokeError(
                f"Video pose q64 smoke failed while decoding or extracting poses from {video_path}. "
                "Verify the video file is present/readable and MediaPipe can process this clip."
            ) from exc
    finally:
        try:
            extractor.close()
        except Exception:
            pass

    q64_components = _q64_pose_components(components)
    features = _flatten_pose_components(
        q64_components,
        include_components=("body", "left_hand", "right_hand", "face"),
        normalize=True,
    )
    source_frame_count = int(features.shape[0])
    feature_count = int(features.shape[1])
    source_shape = _parse_reference_shape(str(source_record.get("input", "")))
    if source_shape is None:
        raise ValueError(f"reference q64 record missing shape metadata for {sample_id}.")
    target_frame_count, expected_feature_count = source_shape
    if expected_feature_count != feature_count:
        raise ValueError(
            f"reference q64 feature mismatch for {sample_id}: records file has {expected_feature_count}, generated {feature_count}"
        )
    features = _resample_feature_frames(features, target_frame_count)

    frame_count = int(features.shape[0])
    record = build_cached_pose_q64_record(
        sample_id=sample_id,
        expected_gloss=expected_gloss,
        frames=features.tolist(),
        encoding=config.encoding,
    )
    _validate_q64_record(record, sample_id=sample_id, expected_gloss=expected_gloss, frames=frame_count, features=feature_count)
    _validate_reference_q64_metadata(
        source_record,
        sample_id=sample_id,
        expected_encoding=config.encoding,
        expected_frames=frame_count,
        expected_features=feature_count,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{sample_id}_video_pose_q64_smoke.jsonl"
    report_path = out_dir / "video_pose_q64_smoke_report.json"
    jsonl_path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")

    report = {
        "scope": VIDEO_POSE_Q64_SMOKE_SCOPE,
        "sample_id": sample_id,
        "expected_gloss": expected_gloss,
        "video_path": str(video_path),
        "manifest_path": str(config.manifest_path),
        "records_path": str(config.records_path),
        "extraction": {
            "extractor": type(extractor).__name__,
            "requested_max_frames": config.max_frames,
        },
        "coverage": {
            "source_frames": source_frame_count,
            "q64_frames": frame_count,
            "features_per_frame": feature_count,
            "components": _component_coverage(components),
        },
        "q64": {
            "encoding": config.encoding,
            "alphabet": ALPHABET,
            "clip": 4,
            "stride": 1,
            "jsonl_path": str(jsonl_path),
        },
        "status": "ok",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return VideoPoseQ64SmokeResult(
        scope=VIDEO_POSE_Q64_SMOKE_SCOPE,
        sample_id=sample_id,
        expected_gloss=expected_gloss,
        encoding=config.encoding,
        frames=frame_count,
        features_per_frame=feature_count,
        video_path=str(video_path),
        jsonl_path=jsonl_path,
        report_path=report_path,
        record=record,
    )


def result_to_dict(result: VideoPoseQ64SmokeResult) -> dict[str, Any]:
    """Serialize a smoke result for CLI output."""

    payload = asdict(result)
    payload["jsonl_path"] = str(result.jsonl_path)
    payload["report_path"] = str(result.report_path)
    return payload


def _build_extractor(extractor_factory: Callable[[], VideoPoseExtractor] | None) -> VideoPoseExtractor:
    if extractor_factory is not None:
        return extractor_factory()

    from src.data.pose_extractor import PoseExtractor

    return PoseExtractor(include_face=False)


def _q64_pose_components(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    q64_components: dict[str, np.ndarray] = {}
    for name, component in components.items():
        if component.ndim == 3 and component.shape[2] > 3:
            q64_components[name] = component[:, :, :3]
        else:
            q64_components[name] = component
    return q64_components


def _component_coverage(components: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for name, component in components.items():
        if component.ndim != 3:
            continue
        frame_presence = np.any(np.abs(component) > 0, axis=(1, 2))
        coverage[name] = {
            "frames": int(component.shape[0]),
            "joints": int(component.shape[1]),
            "coordinates": int(component.shape[2]),
            "covered_frames": int(np.count_nonzero(frame_presence)),
        }
    return coverage
