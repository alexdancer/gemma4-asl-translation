"""Cached pose archive to q64 JSONL verification path."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.q64_encoding import ALPHABET, encode_frames_q64
from src.evaluation.unsloth_asl import load_manifest_labels, load_q64_jsonl, normalize_gloss

Q64_FULL_ENCODING = "q64_full"
Q64_INSTRUCTION = "Classify this compact ASL pose encoding into its WLASL gloss. Reply with only the gloss word."
VERIFICATION_SCOPE = "cached_pose_q64_verification"


@dataclass(frozen=True)
class CachedPoseQ64VerificationConfig:
    """Configuration for one cached-pose q64 compatibility verification."""

    pose_path: Path | str
    sample_id: str
    expected_gloss: str
    manifest_path: Path | str
    out_dir: Path | str
    records_path: Path | str | None = None
    encoding: str = Q64_FULL_ENCODING


@dataclass(frozen=True)
class CachedPoseQ64VerificationResult:
    """Artifact paths and generated contract record from one verification run."""

    scope: str
    sample_id: str
    expected_gloss: str
    encoding: str
    frames: int
    features_per_frame: int
    pose_path: str
    jsonl_path: Path
    report_path: Path
    record: dict[str, str]


def verify_cached_pose_q64(config: CachedPoseQ64VerificationConfig) -> CachedPoseQ64VerificationResult:
    """Convert a cached pose archive into a q64 JSONL record and verify compatibility."""

    sample_id = str(config.sample_id).strip()
    if not sample_id:
        raise ValueError("sample_id must not be empty.")
    if config.encoding != Q64_FULL_ENCODING:
        raise ValueError(f"Unsupported cached-pose encoding: {config.encoding}")

    expected_gloss = normalize_gloss(str(config.expected_gloss))
    if not expected_gloss:
        raise ValueError("expected_gloss must not be empty.")

    labels = load_manifest_labels(config.manifest_path)
    if expected_gloss not in labels:
        raise ValueError(f"expected_gloss is not in manifest labels: {expected_gloss}")

    source_record = _load_source_record(config.records_path, sample_id) if config.records_path is not None else None
    if source_record is not None:
        source_expected = normalize_gloss(str(source_record.get("output", "")))
        if source_expected != expected_gloss:
            raise ValueError(
                f"expected_gloss mismatch for {sample_id}: "
                f"records file has {source_expected!r}, got {expected_gloss!r}"
            )

    pose_path = Path(config.pose_path)
    components, archive_sample_id = _load_pose_archive(pose_path)
    if archive_sample_id is not None and archive_sample_id != sample_id:
        raise ValueError(f"pose archive sample_id mismatch: archive has {archive_sample_id!r}, got {sample_id!r}")

    features = _flatten_pose_components(
        components,
        include_components=("body", "left_hand", "right_hand", "face"),
        normalize=True,
    )
    feature_count = int(features.shape[1])
    source_frame_count = int(features.shape[0])
    target_frame_count = source_frame_count
    if source_record is not None:
        source_shape = _parse_reference_shape(str(source_record.get("input", "")))
        if source_shape is None:
            raise ValueError(f"reference q64 record missing shape metadata for {sample_id}.")
        target_frame_count, expected_feature_count = source_shape
        if expected_feature_count != feature_count:
            raise ValueError(
                f"reference q64 feature mismatch for {sample_id}: "
                f"records file has {expected_feature_count}, generated {feature_count}"
            )
        features = _resample_feature_frames(features, target_frame_count)

    frames = features.tolist()
    frame_count = int(features.shape[0])
    if frame_count <= 0 or feature_count <= 0:
        raise ValueError(f"Pose archive must contain at least one non-empty frame: {pose_path}")

    record = build_cached_pose_q64_record(
        sample_id=sample_id,
        expected_gloss=expected_gloss,
        frames=frames,
        encoding=config.encoding,
    )
    _validate_q64_record(record, sample_id=sample_id, expected_gloss=expected_gloss, frames=frame_count, features=feature_count)
    if source_record is not None:
        _validate_reference_q64_metadata(
            source_record,
            sample_id=sample_id,
            expected_encoding=config.encoding,
            expected_frames=frame_count,
            expected_features=feature_count,
        )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{sample_id}_cached_pose_q64.jsonl"
    report_path = out_dir / "cached_pose_q64_verification_report.json"
    jsonl_path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")

    report = {
        "scope": VERIFICATION_SCOPE,
        "sample_id": sample_id,
        "expected_gloss": expected_gloss,
        "encoding": config.encoding,
        "pose_path": str(pose_path),
        "manifest_path": str(config.manifest_path),
        "records_path": None if config.records_path is None else str(config.records_path),
        "source_frames": source_frame_count,
        "frames": frame_count,
        "features_per_frame": feature_count,
        "q64_jsonl_path": str(jsonl_path),
        "schema": {
            "required_fields": ["instruction", "input", "output"],
            "input_metadata": ["sample_id", "encoding", "frames", "features_per_frame", "pose_q64"],
        },
        "status": "ok",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return CachedPoseQ64VerificationResult(
        scope=VERIFICATION_SCOPE,
        sample_id=sample_id,
        expected_gloss=expected_gloss,
        encoding=config.encoding,
        frames=frame_count,
        features_per_frame=feature_count,
        pose_path=str(pose_path),
        jsonl_path=jsonl_path,
        report_path=report_path,
        record=record,
    )


def build_cached_pose_q64_record(
    *,
    sample_id: str,
    expected_gloss: str,
    frames: Sequence[Sequence[float]],
    encoding: str = Q64_FULL_ENCODING,
) -> dict[str, str]:
    """Build one q64 JSONL-compatible instruction/input/output record."""

    frame_rows = [[float(value) for value in frame] for frame in frames]
    feature_count = min(len(frame) for frame in frame_rows) if frame_rows else 0
    pose_q64 = encode_frames_q64(frame_rows, stride=1)
    input_text = (
        f"sample_id={sample_id}\n"
        f"encoding={encoding} clip=4 alphabet={ALPHABET}\n"
        f"frames={len(frame_rows)} features_per_frame={feature_count}\n"
        f"pose_q64={pose_q64}"
    )
    return {
        "instruction": Q64_INSTRUCTION,
        "input": input_text,
        "output": expected_gloss,
    }


def _flatten_pose_components(
    pose_components: Mapping[str, np.ndarray],
    *,
    include_components: Sequence[str],
    normalize: bool,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    timesteps: int | None = None

    for name in include_components:
        component = pose_components.get(name)
        if component is None:
            continue
        if component.ndim != 3:
            raise ValueError(f"Pose component '{name}' must have shape (T, J, C).")
        if timesteps is None:
            timesteps = int(component.shape[0])
        elif int(component.shape[0]) != timesteps:
            raise ValueError("All pose components must share the same sequence length.")
        arrays.append(component.reshape(component.shape[0], -1))

    if not arrays or timesteps is None:
        raise ValueError("No pose components were available to flatten.")

    flattened = np.concatenate(arrays, axis=1).astype(np.float32)
    return _normalize_pose_embeddings(flattened) if normalize else flattened


def _normalize_pose_embeddings(pose_embeddings: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    centered = pose_embeddings.astype(np.float32) - pose_embeddings.mean(axis=0, keepdims=True)
    scale = pose_embeddings.std(axis=0, keepdims=True)
    normalized = centered / np.maximum(scale, epsilon)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def _resample_feature_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError(f"reference q64 frame count must be positive, got {target_frames}.")
    source_frames = int(features.shape[0])
    if source_frames == target_frames:
        return features
    if source_frames == 1:
        return np.repeat(features, target_frames, axis=0).astype(np.float32)

    source_positions = np.arange(source_frames, dtype=np.float32)
    target_positions = np.linspace(0, source_frames - 1, target_frames, dtype=np.float32)
    resampled = np.empty((target_frames, features.shape[1]), dtype=np.float32)
    for feature_index in range(features.shape[1]):
        resampled[:, feature_index] = np.interp(target_positions, source_positions, features[:, feature_index])
    return resampled


def _load_pose_archive(pose_path: Path) -> tuple[dict[str, np.ndarray], str | None]:
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose archive not found: {pose_path}")

    try:
        archive = np.load(pose_path, allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"Pose archive is unreadable or corrupt: {pose_path}") from exc

    components: dict[str, np.ndarray] = {}
    for component_name in ("body", "left_hand", "right_hand", "face"):
        if component_name in archive:
            component = archive[component_name].astype(np.float32)
            if component.ndim != 3:
                raise ValueError(f"Pose component '{component_name}' must have shape (T, J, C): {pose_path}")
            if component.shape[1] <= 0 or component.shape[2] <= 0:
                raise ValueError(f"Pose component '{component_name}' must have non-empty joint and coordinate axes: {pose_path}")
            if not np.isfinite(component).all():
                raise ValueError(f"Pose component '{component_name}' contains non-finite values: {pose_path}")
            components[component_name] = component

    missing = [name for name in ("body", "left_hand", "right_hand") if name not in components]
    if missing:
        raise ValueError(f"Pose archive missing required components {missing}: {pose_path}")

    lengths = {name: int(component.shape[0]) for name, component in components.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Pose components must share one frame count: {lengths}")
    if next(iter(lengths.values())) <= 0:
        raise ValueError(f"Pose archive contains no frames: {pose_path}")

    archive_sample_id = _read_optional_archive_text(archive, "sample_id")
    return components, archive_sample_id


def _read_optional_archive_text(archive: Mapping[str, Any], key: str) -> str | None:
    if key not in archive:
        return None
    value = archive[key]
    try:
        text = str(np.asarray(value).item()).strip()
    except ValueError as exc:
        raise ValueError(f"Pose archive field '{key}' must be a scalar string.") from exc
    return text or None


def _load_source_record(records_path: Path | str, sample_id: str) -> Mapping[str, Any]:
    for record in load_q64_jsonl(records_path):
        if _record_sample_id(record) == sample_id:
            return record
    raise ValueError(f"sample_id not found in q64 records: {sample_id}")


def _record_sample_id(record: Mapping[str, Any]) -> str | None:
    match = re.search(r"^sample_id=([^\n]+)", str(record.get("input", "")), flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _input_line_value(input_text: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in input_text.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return None


def _parse_reference_shape(input_text: str) -> tuple[int, int] | None:
    shape_line = _input_line_value(input_text, "frames")
    if shape_line is None:
        return None
    match = re.fullmatch(r"(\d+)\s+features_per_frame=(\d+)", shape_line)
    if match is None:
        raise ValueError(f"Reference q64 record has malformed shape metadata: frames={shape_line}")
    return int(match.group(1)), int(match.group(2))


def _validate_reference_q64_metadata(
    source_record: Mapping[str, Any],
    *,
    sample_id: str,
    expected_encoding: str,
    expected_frames: int,
    expected_features: int,
) -> None:
    input_text = str(source_record.get("input", ""))
    source_encoding = _input_line_value(input_text, "encoding")
    expected_encoding_line = f"{expected_encoding} clip=4 alphabet={ALPHABET}"
    if source_encoding != expected_encoding_line:
        raise ValueError(
            f"reference q64 encoding mismatch for {sample_id}: "
            f"records file has {source_encoding!r}, expected {expected_encoding_line!r}"
        )

    source_shape = _parse_reference_shape(input_text)
    expected_shape = (expected_frames, expected_features)
    if source_shape != expected_shape:
        raise ValueError(
            f"reference q64 shape mismatch for {sample_id}: "
            f"records file has {source_shape}, generated {expected_shape}"
        )

    pose_q64 = _input_line_value(input_text, "pose_q64")
    if not pose_q64:
        raise ValueError(f"reference q64 record missing pose_q64 payload for {sample_id}.")
    rows = pose_q64.split("|")
    if len(rows) != expected_frames or any(len(row) != expected_features for row in rows):
        raise ValueError(
            f"reference q64 payload shape mismatch for {sample_id}: "
            f"expected {expected_frames} rows of {expected_features} characters."
        )


def _validate_q64_record(
    record: Mapping[str, Any],
    *,
    sample_id: str,
    expected_gloss: str,
    frames: int,
    features: int,
) -> None:
    for field in ("instruction", "input", "output"):
        if field not in record or not str(record[field]).strip():
            raise ValueError(f"Generated q64 record missing required field: {field}")

    input_text = str(record["input"])
    required_lines = {
        "sample_id": f"sample_id={sample_id}",
        "encoding": f"encoding={Q64_FULL_ENCODING} clip=4 alphabet={ALPHABET}",
        "shape": f"frames={frames} features_per_frame={features}",
    }
    for name, line in required_lines.items():
        if line not in input_text.splitlines():
            raise ValueError(f"Generated q64 record missing {name} metadata line: {line}")
    if not any(line.startswith("pose_q64=") and line.removeprefix("pose_q64=") for line in input_text.splitlines()):
        raise ValueError("Generated q64 record missing pose_q64 payload.")
    if normalize_gloss(str(record["output"])) != expected_gloss:
        raise ValueError(f"Generated q64 record output mismatch for {sample_id}.")


def result_to_dict(result: CachedPoseQ64VerificationResult) -> dict[str, Any]:
    """Serialize a verification result for CLI output."""

    payload = asdict(result)
    payload["jsonl_path"] = str(result.jsonl_path)
    payload["report_path"] = str(result.report_path)
    return payload
