from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_phrase_generator import (
    SyntheticPhraseConfig,
    generate_synthetic_phrases,
    summarize_synthetic_distribution,
)


def _write_pose(path: Path, frames: int) -> None:
    np.savez(
        path,
        body=np.ones((frames, 17, 4), dtype=np.float32),
        left_hand=np.ones((frames, 21, 4), dtype=np.float32),
        right_hand=np.ones((frames, 21, 4), dtype=np.float32),
    )


def _base_annotations(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    pose_root = tmp_path / "poses"
    pose_root.mkdir()

    rows = []
    for gloss, lengths in {"hello": [4, 5], "thanks": [3, 4], "yes": [5, 6]}.items():
        for idx, frames in enumerate(lengths, start=1):
            sample_id = f"{gloss}_{idx:03d}"
            _write_pose(pose_root / f"{sample_id}.npz", frames=frames)
            rows.append({"sample_id": sample_id, "gloss": gloss})

    return pd.DataFrame(rows), pose_root


def test_generate_synthetic_phrases_emits_boundary_metadata_and_word_counts(tmp_path: Path) -> None:
    annotations, pose_root = _base_annotations(tmp_path)
    output_dir = tmp_path / "synthetic"

    manifest, samples = generate_synthetic_phrases(
        annotations=annotations,
        pose_root=pose_root,
        output_dir=output_dir,
        num_phrases=12,
        config=SyntheticPhraseConfig(random_seed=7, min_words=2, max_words=3, gap_frames_min=2, gap_frames_max=5),
    )

    assert len(manifest) == 12
    assert len(samples) == 12
    assert manifest["word_count"].between(2, 3).all()

    first = samples[0]
    assert first.pose_path.exists()
    assert first.total_frames > 0
    assert len(first.boundaries) in (2, 3)

    parsed_boundaries = json.loads(manifest.iloc[0]["boundaries_json"])
    assert len(parsed_boundaries) == manifest.iloc[0]["word_count"]
    assert parsed_boundaries[0]["start_frame"] == 0
    assert parsed_boundaries[0]["end_frame"] >= parsed_boundaries[0]["start_frame"]
    assert parsed_boundaries[-1]["end_ms"] > 0


def test_gap_randomization_distribution_is_non_degenerate(tmp_path: Path) -> None:
    annotations, pose_root = _base_annotations(tmp_path)

    _, samples = generate_synthetic_phrases(
        annotations=annotations,
        pose_root=pose_root,
        output_dir=tmp_path / "synthetic",
        num_phrases=80,
        config=SyntheticPhraseConfig(random_seed=42, gap_frames_min=2, gap_frames_max=8),
    )

    summary = summarize_synthetic_distribution(samples)
    assert summary["num_samples"] == 80
    assert summary["gap_frames_min"] == 2
    assert summary["gap_frames_max"] == 8
    assert len(summary["gap_frames_unique"]) >= 3


def test_repeat_probability_is_configurable(tmp_path: Path) -> None:
    annotations, pose_root = _base_annotations(tmp_path)

    _, no_repeat_samples = generate_synthetic_phrases(
        annotations=annotations,
        pose_root=pose_root,
        output_dir=tmp_path / "synthetic_no_repeat",
        num_phrases=40,
        config=SyntheticPhraseConfig(random_seed=1, repeat_probability=0.0),
    )
    no_repeat_summary = summarize_synthetic_distribution(no_repeat_samples)
    assert no_repeat_summary["adjacent_repeat_rate"] == 0.0

    _, repeat_samples = generate_synthetic_phrases(
        annotations=annotations,
        pose_root=pose_root,
        output_dir=tmp_path / "synthetic_repeat",
        num_phrases=40,
        config=SyntheticPhraseConfig(random_seed=1, repeat_probability=1.0),
    )
    repeat_summary = summarize_synthetic_distribution(repeat_samples)
    assert repeat_summary["adjacent_repeat_rate"] == 1.0


def test_rejects_pose_paths_outside_pose_root(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    pose_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    _write_pose(external / "outside.npz", frames=5)

    attack = pd.DataFrame([{"sample_id": "x", "gloss": "hello", "pose_path": "../external/outside.npz"}])

    with pytest.raises(ValueError, match="escapes pose_root"):
        generate_synthetic_phrases(
            annotations=attack,
            pose_root=pose_root,
            output_dir=tmp_path / "synthetic_attack",
            num_phrases=1,
            config=SyntheticPhraseConfig(random_seed=0),
        )


def test_rejects_mismatched_component_lengths(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    pose_root.mkdir()
    np.savez(
        pose_root / "broken_001.npz",
        body=np.ones((4, 17, 4), dtype=np.float32),
        left_hand=np.ones((3, 21, 4), dtype=np.float32),
        right_hand=np.ones((4, 21, 4), dtype=np.float32),
    )

    annotations = pd.DataFrame([{"sample_id": "broken_001", "gloss": "hello"}])

    with pytest.raises(ValueError, match="frame lengths must match"):
        generate_synthetic_phrases(
            annotations=annotations,
            pose_root=pose_root,
            output_dir=tmp_path / "synthetic_broken",
            num_phrases=1,
            config=SyntheticPhraseConfig(random_seed=0),
        )


def test_rejects_zero_frame_components(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    pose_root.mkdir()
    np.savez(
        pose_root / "empty_001.npz",
        body=np.ones((0, 17, 4), dtype=np.float32),
        left_hand=np.ones((0, 21, 4), dtype=np.float32),
        right_hand=np.ones((0, 21, 4), dtype=np.float32),
    )

    annotations = pd.DataFrame([{"sample_id": "empty_001", "gloss": "hello"}])

    with pytest.raises(ValueError, match="empty component sequence"):
        generate_synthetic_phrases(
            annotations=annotations,
            pose_root=pose_root,
            output_dir=tmp_path / "synthetic_empty",
            num_phrases=1,
            config=SyntheticPhraseConfig(random_seed=0),
        )
