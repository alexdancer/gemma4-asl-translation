"""Synthetic multi-word phrase generation from single-word Top-50 pose clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.create_splits import DEFAULT_TOP50_GLOSSES

REQUIRED_COMPONENTS: Tuple[str, ...] = ("body", "left_hand", "right_hand")
OPTIONAL_COMPONENTS: Tuple[str, ...] = ("face",)


@dataclass(frozen=True)
class SyntheticPhraseConfig:
    min_words: int = 2
    max_words: int = 3
    repeat_probability: float = 0.10
    gap_frames_min: int = 4
    gap_frames_max: int = 10
    fps: float = 30.0
    include_face: bool = False
    top50_glosses: Tuple[str, ...] = tuple(DEFAULT_TOP50_GLOSSES)
    random_seed: Optional[int] = None


@dataclass(frozen=True)
class WordBoundary:
    word: str
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SyntheticPhraseSample:
    sample_id: str
    phrase: str
    pose_path: Path
    source_sample_ids: Tuple[str, ...]
    boundaries: Tuple[WordBoundary, ...]
    gap_frames: Tuple[int, ...]
    total_frames: int


def _validate_config(config: SyntheticPhraseConfig) -> None:
    if config.min_words < 1 or config.max_words < config.min_words:
        raise ValueError("Invalid word-count range in SyntheticPhraseConfig.")
    if not (0.0 <= config.repeat_probability <= 1.0):
        raise ValueError("repeat_probability must be within [0.0, 1.0].")
    if config.gap_frames_min < 0 or config.gap_frames_max < config.gap_frames_min:
        raise ValueError("Invalid gap frame range.")
    if config.fps <= 0:
        raise ValueError("fps must be positive.")


def _resolve_pose_path(row: Mapping[str, object], pose_root: Path) -> Path:
    raw = row.get("pose_path")
    if raw is None or pd.isna(raw) or str(raw).strip() == "":
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise KeyError("Each annotation row must include 'pose_path' or 'sample_id'.")
        raw = f"{sample_id}.npz"

    path = Path(str(raw))
    if path.is_absolute():
        raise ValueError("Absolute pose_path values are not allowed; paths must resolve under pose_root.")

    pose_root_resolved = pose_root.resolve(strict=False)
    resolved = (pose_root_resolved / path).resolve(strict=False)
    if pose_root_resolved != resolved and pose_root_resolved not in resolved.parents:
        raise ValueError(f"pose_path escapes pose_root: {path}")
    return resolved


def _load_pose_components(path: Path, include_face: bool) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Pose archive not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        components: Dict[str, np.ndarray] = {}
        for name in REQUIRED_COMPONENTS + OPTIONAL_COMPONENTS:
            if name in archive:
                components[name] = archive[name].astype(np.float32)

    for name in REQUIRED_COMPONENTS:
        if name not in components:
            raise KeyError(f"Missing required component '{name}' in pose archive: {path}")

    if include_face and "face" not in components:
        raise KeyError(f"include_face=True but pose archive has no 'face' component: {path}")

    frame_lengths = {name: int(value.shape[0]) for name, value in components.items() if name in REQUIRED_COMPONENTS or (include_face and name == "face")}
    if any(length <= 0 for length in frame_lengths.values()):
        raise ValueError(f"Pose archive contains an empty component sequence: {path}")
    if len(set(frame_lengths.values())) != 1:
        raise ValueError(f"Pose component frame lengths must match in {path}: {frame_lengths}")

    return components


def _empty_gap_like(reference: np.ndarray, gap_frames: int) -> np.ndarray:
    if gap_frames <= 0:
        return np.zeros((0, *reference.shape[1:]), dtype=np.float32)
    return np.zeros((gap_frames, *reference.shape[1:]), dtype=np.float32)


def _sample_gloss_sequence(
    glosses: Sequence[str],
    word_count: int,
    repeat_probability: float,
    rng: np.random.Generator,
) -> List[str]:
    if not glosses:
        raise ValueError("No available glosses for phrase sampling.")

    sequence = [str(rng.choice(glosses))]
    for _ in range(word_count - 1):
        if rng.random() < repeat_probability:
            sequence.append(sequence[-1])
            continue
        candidates = [g for g in glosses if g != sequence[-1]]
        sequence.append(str(rng.choice(candidates if candidates else glosses)))
    return sequence


def generate_synthetic_phrases(
    annotations: pd.DataFrame,
    pose_root: Path,
    output_dir: Path,
    num_phrases: int,
    config: Optional[SyntheticPhraseConfig] = None,
) -> Tuple[pd.DataFrame, List[SyntheticPhraseSample]]:
    """Create synthetic multi-word pose clips and return a manifest + rich samples."""

    cfg = config or SyntheticPhraseConfig()
    _validate_config(cfg)

    if annotations.empty:
        raise ValueError("annotations must not be empty.")
    if "gloss" not in annotations.columns:
        raise KeyError("annotations must contain a 'gloss' column.")
    if num_phrases <= 0:
        raise ValueError("num_phrases must be positive.")

    pose_root = Path(pose_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)

    top50_set = {str(g) for g in cfg.top50_glosses}
    filtered = annotations.loc[annotations["gloss"].astype(str).isin(top50_set)].copy()
    if filtered.empty:
        raise ValueError("No annotations matched configured top-50 gloss set.")

    by_gloss: Dict[str, List[MutableMapping[str, object]]] = {}
    for row in filtered.to_dict(orient="records"):
        gloss = str(row["gloss"])
        by_gloss.setdefault(gloss, []).append(row)

    available_glosses = sorted(by_gloss)
    manifest_rows: List[Dict[str, object]] = []
    samples: List[SyntheticPhraseSample] = []

    for phrase_index in range(num_phrases):
        word_count = int(rng.integers(cfg.min_words, cfg.max_words + 1))
        phrase_glosses = _sample_gloss_sequence(available_glosses, word_count, cfg.repeat_probability, rng)

        selected_rows: List[MutableMapping[str, object]] = [dict(rng.choice(by_gloss[word])) for word in phrase_glosses]

        merged_components: Dict[str, List[np.ndarray]] = {name: [] for name in REQUIRED_COMPONENTS}
        if cfg.include_face:
            merged_components["face"] = []

        boundaries: List[WordBoundary] = []
        gap_frames_used: List[int] = []
        source_sample_ids: List[str] = []
        timeline_frame = 0

        for idx, row in enumerate(selected_rows):
            pose_path = _resolve_pose_path(row, pose_root)
            components = _load_pose_components(pose_path, include_face=cfg.include_face)
            source_sample_ids.append(str(row.get("sample_id", pose_path.stem)))

            source_len = int(components["body"].shape[0])
            start_frame = timeline_frame
            end_frame = start_frame + source_len - 1
            boundaries.append(
                WordBoundary(
                    word=str(row["gloss"]),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=int(round((start_frame / cfg.fps) * 1000.0)),
                    end_ms=int(round(((end_frame + 1) / cfg.fps) * 1000.0)),
                )
            )

            for name in merged_components:
                merged_components[name].append(components[name])

            timeline_frame = end_frame + 1

            if idx < len(selected_rows) - 1:
                gap = int(rng.integers(cfg.gap_frames_min, cfg.gap_frames_max + 1))
                gap_frames_used.append(gap)
                for name in merged_components:
                    merged_components[name].append(_empty_gap_like(components[name], gap))
                timeline_frame += gap

        stacked_components = {name: np.concatenate(parts, axis=0).astype(np.float32) for name, parts in merged_components.items()}
        sample_id = f"synthetic_phrase_{phrase_index:06d}"
        pose_path = output_dir / f"{sample_id}.npz"
        np.savez_compressed(pose_path, **stacked_components)

        phrase_text = " ".join(phrase_glosses)
        sample = SyntheticPhraseSample(
            sample_id=sample_id,
            phrase=phrase_text,
            pose_path=pose_path,
            source_sample_ids=tuple(source_sample_ids),
            boundaries=tuple(boundaries),
            gap_frames=tuple(gap_frames_used),
            total_frames=int(stacked_components["body"].shape[0]),
        )
        samples.append(sample)

        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "gloss": sample.phrase,
                "pose_path": str(sample.pose_path),
                "word_count": len(sample.boundaries),
                "total_frames": sample.total_frames,
                "source_sample_ids_json": json.dumps(sample.source_sample_ids),
                "boundaries_json": json.dumps([b.__dict__ for b in sample.boundaries]),
                "gap_frames_json": json.dumps(sample.gap_frames),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    return manifest, samples


def summarize_synthetic_distribution(samples: Sequence[SyntheticPhraseSample]) -> Dict[str, object]:
    """Return lightweight distribution stats for phrase-length and gap randomization checks."""

    if not samples:
        return {
            "num_samples": 0,
            "word_count_distribution": {},
            "gap_frames_min": None,
            "gap_frames_max": None,
            "gap_frames_unique": [],
            "adjacent_repeat_rate": 0.0,
        }

    word_counts: Dict[int, int] = {}
    gaps: List[int] = []
    repeated_edges = 0
    total_edges = 0

    for sample in samples:
        wc = len(sample.boundaries)
        word_counts[wc] = word_counts.get(wc, 0) + 1
        gaps.extend(int(g) for g in sample.gap_frames)

        words = [b.word for b in sample.boundaries]
        for left, right in zip(words[:-1], words[1:]):
            total_edges += 1
            if left == right:
                repeated_edges += 1

    return {
        "num_samples": len(samples),
        "word_count_distribution": {str(k): v for k, v in sorted(word_counts.items())},
        "gap_frames_min": min(gaps) if gaps else None,
        "gap_frames_max": max(gaps) if gaps else None,
        "gap_frames_unique": sorted(set(gaps)),
        "adjacent_repeat_rate": (repeated_edges / total_edges) if total_edges else 0.0,
    }
