"""Behavior tests for pose-to-text dataset manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pose_to_text_dataset import PoseToTextDataset


def test_dataset_resolves_pose_archive_from_sample_id_when_pose_path_is_absent(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    pose_root.mkdir()
    np.savez(
        pose_root / "hello_001.npz",
        body=np.ones((2, 17, 3), dtype=np.float32),
        left_hand=np.ones((2, 21, 3), dtype=np.float32),
        right_hand=np.ones((2, 21, 3), dtype=np.float32),
    )
    annotations = pd.DataFrame(
        {
            "sample_id": ["hello_001"],
            "gloss": ["hello"],
            "signer_id": ["signer-a"],
        }
    )

    dataset = PoseToTextDataset(annotations=annotations, pose_root=pose_root)
    sample = dataset[0]

    assert sample["sample_id"] == "hello_001"
    assert sample["gloss"] == "hello"
    assert sample["pose_path"] == str(pose_root / "hello_001.npz")
    assert tuple(sample["pose_features"].shape) == (2, 177)
