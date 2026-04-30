"""Utilities to generate train, validation, and test splits for ASL datasets."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

TOP50_CONTRACT_VERSION = "asl-top50-v1"
DEFAULT_TOP50_GLOSSES = [
    "hello",
    "thanks",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "want",
    "need",
    "more",
    "finished",
    "again",
    "good",
    "bad",
    "fine",
    "like",
    "don't like",
    "understand",
    "don't understand",
    "know",
    "don't know",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "name",
    "me",
    "you",
    "he",
    "she",
    "we",
    "they",
    "go",
    "come",
    "home",
    "school",
    "work",
    "eat",
    "drink",
    "water",
    "bathroom",
    "friend",
    "family",
    "mother",
    "father",
    "today",
    "tomorrow",
    "yesterday",
]


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, received {total}")


def create_top50_contract(glosses: Sequence[str] | None = None) -> Dict[str, object]:
    """Return the versioned Top-50 ASL gloss contract."""

    contract_glosses = list(glosses or DEFAULT_TOP50_GLOSSES)
    if len(contract_glosses) != 50:
        raise ValueError(f"Top-50 contract must contain exactly 50 glosses, received {len(contract_glosses)}")
    if len(set(contract_glosses)) != 50:
        raise ValueError("Top-50 contract glosses must be unique.")

    return {
        "version": TOP50_CONTRACT_VERSION,
        "description": "Fixed Top-50 ASL gloss list for the Gemma 4 ASL hackathon v1 demo scope.",
        "glosses": contract_glosses,
    }


def save_top50_contract(output_path: Path, overwrite: bool = False) -> Path:
    """Write the default Top-50 contract JSON."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(create_top50_contract(), indent=2), encoding="utf-8")
    return output_path


def load_top50_contract(contract_path: Path) -> Dict[str, object]:
    """Load and validate a Top-50 contract from disk."""

    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    version = contract.get("version")
    glosses = contract.get("glosses")
    if version != TOP50_CONTRACT_VERSION:
        raise ValueError(f"Unsupported Top-50 contract version: {version}")
    if not isinstance(glosses, list):
        raise ValueError("Top-50 contract must contain a 'glosses' list.")
    create_top50_contract([str(gloss) for gloss in glosses])
    return contract


def filter_to_gloss_contract(metadata: pd.DataFrame, glosses: Iterable[str]) -> pd.DataFrame:
    """Filter metadata to the configured gloss contract, preserving contract order."""

    if "gloss" not in metadata.columns:
        raise KeyError("Metadata must contain a 'gloss' column for Top-50 filtering.")

    gloss_order = {gloss: index for index, gloss in enumerate(glosses)}
    frame = metadata.loc[metadata["gloss"].isin(gloss_order)].copy()
    frame["_gloss_order"] = frame["gloss"].map(gloss_order)
    frame = frame.sort_values(["_gloss_order", "sample_id"]).drop(columns=["_gloss_order"])
    return frame.reset_index(drop=True)


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


def signer_independent_split(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata by signer so train, validation, and test signers never overlap."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if "signer_id" not in metadata.columns:
        raise KeyError("Metadata must contain a 'signer_id' column for signer-independent splitting.")
    if metadata["signer_id"].isna().any():
        missing = metadata.loc[metadata["signer_id"].isna(), "sample_id"].head(5).tolist()
        raise ValueError(f"Signer-independent split requires signer_id for every sample; missing examples: {missing}")

    rng = random.Random(seed)
    signers: List[object] = list(metadata["signer_id"].drop_duplicates())
    rng.shuffle(signers)
    if len(signers) < 3:
        raise ValueError("Signer-independent split requires at least 3 unique signers.")

    train_count = max(1, int(round(len(signers) * train_ratio)))
    val_count = max(1, int(round(len(signers) * val_ratio)))
    if train_count + val_count >= len(signers):
        val_count = 1
        train_count = max(1, len(signers) - 2)

    train_signers = set(signers[:train_count])
    val_signers = set(signers[train_count : train_count + val_count])
    test_signers = set(signers[train_count + val_count :])

    train_df = metadata.loc[metadata["signer_id"].isin(train_signers)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = metadata.loc[metadata["signer_id"].isin(val_signers)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = metadata.loc[metadata["signer_id"].isin(test_signers)].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    LOGGER.info(
        "Created signer-independent splits: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def create_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_strategy: str | None = None,
    contract_version: str | None = None,
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
    if split_strategy is not None:
        manifest["split_strategy"] = split_strategy
    if contract_version is not None:
        manifest["contract_version"] = contract_version
    return manifest


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    split_strategy: str | None = None,
    contract_version: str | None = None,
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
    paths["manifest_json"].write_text(
        json.dumps(
            create_manifest(
                train_df,
                val_df,
                test_df,
                split_strategy=split_strategy,
                contract_version=contract_version,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Saved split artifacts to %s", output_dir)
    return paths


def create_top50_split_artifacts(
    metadata: pd.DataFrame,
    output_dir: Path,
    contract_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, Path] | Path]:
    """Create reproducible Top-50 random and signer-independent split artifacts."""

    contract = load_top50_contract(contract_path)
    top50_metadata = filter_to_gloss_contract(metadata, [str(gloss) for gloss in contract["glosses"]])
    if top50_metadata.empty:
        raise ValueError("No metadata rows matched the Top-50 contract.")

    output_dir = Path(output_dir)
    contract_output = output_dir / Path(contract_path).name
    contract_output.parent.mkdir(parents=True, exist_ok=True)
    contract_output.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    random_train, random_val, random_test = stratified_split(
        top50_metadata,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    signer_train, signer_val, signer_test = signer_independent_split(
        top50_metadata,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    return {
        "contract_json": contract_output,
        "random": save_splits(
            random_train,
            random_val,
            random_test,
            output_dir=output_dir / "random",
            split_strategy="random_stratified",
            contract_version=str(contract["version"]),
        ),
        "signer_independent": save_splits(
            signer_train,
            signer_val,
            signer_test,
            output_dir=output_dir / "signer_independent",
            split_strategy="signer_independent",
            contract_version=str(contract["version"]),
        ),
    }
