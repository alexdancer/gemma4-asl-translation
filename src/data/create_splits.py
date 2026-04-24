"""Utilities to generate train, validation, and test splits for ASL datasets."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, received {total}")


def stratified_split(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a simple per-gloss stratified split without sklearn."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if "gloss" not in metadata.columns:
        raise KeyError("Metadata must contain a 'gloss' column for stratified splitting.")

    rng = random.Random(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in metadata.groupby("gloss", sort=False):
        indices = list(group.index)
        rng.shuffle(indices)
        count = len(indices)

        train_end = max(1, int(round(count * train_ratio))) if count >= 3 else max(1, count - 1)
        remaining = count - train_end
        val_count = int(round(count * val_ratio)) if remaining > 1 else min(1, remaining)
        test_count = count - train_end - val_count

        if test_count < 0:
            test_count = 0
            val_count = count - train_end

        train_idx = indices[:train_end]
        val_idx = indices[train_end : train_end + val_count]
        test_idx = indices[train_end + val_count :]

        train_parts.append(metadata.loc[train_idx])
        if val_idx:
            val_parts.append(metadata.loc[val_idx])
        if test_idx:
            test_parts.append(metadata.loc[test_idx])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True) if val_parts else metadata.iloc[0:0].copy()
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True) if test_parts else metadata.iloc[0:0].copy()

    LOGGER.info(
        "Created dataset splits: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def create_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    """Create a dataset manifest summarizing split composition."""

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest = {
        "num_samples": int(len(all_df)),
        "num_glosses": int(all_df["gloss"].nunique()) if not all_df.empty else 0,
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    return manifest


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Save split CSVs, index files, and a JSON manifest to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train_csv": output_dir / "train.csv",
        "val_csv": output_dir / "val.csv",
        "test_csv": output_dir / "test.csv",
        "manifest_json": output_dir / "manifest.json",
        "train_indices_json": output_dir / "train_indices.json",
        "val_indices_json": output_dir / "val_indices.json",
        "test_indices_json": output_dir / "test_indices.json",
    }

    train_df.to_csv(paths["train_csv"], index=False)
    val_df.to_csv(paths["val_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)

    paths["train_indices_json"].write_text(json.dumps(train_df["sample_id"].tolist(), indent=2), encoding="utf-8")
    paths["val_indices_json"].write_text(json.dumps(val_df["sample_id"].tolist(), indent=2), encoding="utf-8")
    paths["test_indices_json"].write_text(json.dumps(test_df["sample_id"].tolist(), indent=2), encoding="utf-8")
    paths["manifest_json"].write_text(json.dumps(create_manifest(train_df, val_df, test_df), indent=2), encoding="utf-8")

    LOGGER.info("Saved split artifacts to %s", output_dir)
    return paths
