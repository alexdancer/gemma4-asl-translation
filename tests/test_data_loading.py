"""Unit tests for WLASL metadata loading and split creation."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.create_splits import save_splits, stratified_split
from src.data.wlasl_loader import load_wlasl_metadata


def test_load_wlasl_metadata_flattens_instances(tmp_path: Path) -> None:
    payload = [
        {
            "gloss": "hello",
            "instances": [
                {"video_id": "001", "split": "train", "signer_id": 1},
                {"video_id": "002", "split": "val", "signer_id": 2},
            ],
        }
    ]
    metadata_path = tmp_path / "WLASL_v0.3.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    frame = load_wlasl_metadata(metadata_path)

    assert len(frame) == 2
    assert set(frame["sample_id"]) == {"hello_001", "hello_002"}


def test_stratified_split_and_save_outputs(tmp_path: Path) -> None:
    payload = [
        {
            "gloss": "hello",
            "instances": [
                {"video_id": "001", "split": "train"},
                {"video_id": "002", "split": "train"},
                {"video_id": "003", "split": "train"},
            ],
        },
        {
            "gloss": "thanks",
            "instances": [
                {"video_id": "004", "split": "train"},
                {"video_id": "005", "split": "train"},
                {"video_id": "006", "split": "train"},
            ],
        },
    ]
    metadata_path = tmp_path / "WLASL_v0.3.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    metadata = load_wlasl_metadata(metadata_path)
    train_df, val_df, test_df = stratified_split(metadata, seed=7)
    outputs = save_splits(train_df, val_df, test_df, tmp_path / "splits")

    assert len(train_df) + len(val_df) + len(test_df) == len(metadata)
    assert outputs["manifest_json"].exists()
    assert outputs["train_csv"].exists()
