"""Unit tests for WLASL metadata loading and split creation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.create_splits import (
    TOP50_CONTRACT_VERSION,
    create_top50_contract,
    create_top50_split_artifacts,
    filter_to_gloss_contract,
    save_splits,
    save_top50_contract,
    signer_independent_split,
    stratified_split,
)
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


def test_top50_contract_filters_metadata_in_contract_order() -> None:
    metadata = _make_split_metadata(["thanks", "hello", "outside"])
    contract = create_top50_contract(["hello", "thanks", *[f"gloss-{index}" for index in range(48)]])

    filtered = filter_to_gloss_contract(metadata, contract["glosses"])

    assert filtered["gloss"].tolist() == ["hello", "thanks"]
    assert contract["version"] == TOP50_CONTRACT_VERSION


def test_signer_independent_split_has_no_signer_overlap() -> None:
    metadata = _make_split_metadata(["hello", "thanks", "yes", "no", "please", "sorry"], samples_per_gloss=3)

    train_df, val_df, test_df = signer_independent_split(metadata, seed=3)

    train_signers = set(train_df["signer_id"])
    val_signers = set(val_df["signer_id"])
    test_signers = set(test_df["signer_id"])
    assert train_signers.isdisjoint(val_signers)
    assert train_signers.isdisjoint(test_signers)
    assert val_signers.isdisjoint(test_signers)
    assert len(train_df) + len(val_df) + len(test_df) == len(metadata)


def test_create_top50_split_artifacts_writes_random_and_signer_independent_outputs(tmp_path: Path) -> None:
    glosses = ["hello", "thanks", "yes", "no", "please", "sorry"]
    contract_glosses = [*glosses, *[f"unused-{index}" for index in range(44)]]
    metadata = _make_split_metadata([*glosses, "outside"], samples_per_gloss=3)
    contract_path = save_top50_contract(tmp_path / "asl_top50_glosses_v1.json", overwrite=True)
    contract_path.write_text(
        json.dumps({"version": TOP50_CONTRACT_VERSION, "description": "unit test", "glosses": contract_glosses}),
        encoding="utf-8",
    )

    outputs = create_top50_split_artifacts(
        metadata,
        output_dir=tmp_path / "splits",
        contract_path=contract_path,
        seed=11,
    )

    assert Path(outputs["contract_json"]).exists()
    assert Path(outputs["random"]["train_csv"]).exists()
    assert Path(outputs["signer_independent"]["test_csv"]).exists()

    signer_manifest = json.loads(Path(outputs["signer_independent"]["manifest_json"]).read_text(encoding="utf-8"))
    random_manifest = json.loads(Path(outputs["random"]["manifest_json"]).read_text(encoding="utf-8"))
    assert signer_manifest["contract_version"] == TOP50_CONTRACT_VERSION
    assert signer_manifest["split_strategy"] == "signer_independent"
    assert random_manifest["split_strategy"] == "random_stratified"

    signer_sample_ids = []
    for split_name in ["train_indices_json", "val_indices_json", "test_indices_json"]:
        signer_sample_ids.extend(json.loads(Path(outputs["signer_independent"][split_name]).read_text(encoding="utf-8")))
    assert all("outside" not in sample_id for sample_id in signer_sample_ids)


def _make_split_metadata(glosses: list[str], samples_per_gloss: int = 1) -> pd.DataFrame:
    rows = []
    signer_id = 100
    for gloss in glosses:
        for index in range(samples_per_gloss):
            video_id = f"{gloss}-{index}"
            rows.append(
                {
                    "sample_id": f"{gloss}_{video_id}",
                    "gloss": gloss,
                    "split": "train",
                    "signer_id": signer_id,
                    "video_id": video_id,
                    "video_path": None,
                }
            )
            signer_id += 1
    return pd.DataFrame(rows)
