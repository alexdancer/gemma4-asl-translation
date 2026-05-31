"""Tests for Colab Top-50 anchor + alias contract builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.build_colab_anchor_contract import main as build_colab_anchor_contract_main
from src.evaluation.colab_anchor_contract import (
    build_colab_anchor_contract,
    flatten_alias_map,
    top_k_anchors,
)


def _write_manifest(path: Path, labels: list[str]) -> Path:
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return path


def _write_records(path: Path, outputs: list[str]) -> Path:
    records = [
        {
            "instruction": "Classify",
            "input": f"sample_id=s_{idx}",
            "output": output,
        }
        for idx, output in enumerate(outputs)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_top_k_anchors_is_deterministic_for_frequency_ties() -> None:
    records = [
        {"output": "yes"},
        {"output": "hello"},
        {"output": "yes"},
        {"output": "hello"},
        {"output": "thanks"},
        {"output": "thanks"},
    ]
    labels = ["hello", "thanks", "yes"]

    anchors = top_k_anchors(records, labels, top_k=3)

    assert anchors == [
        {"gloss": "hello", "count": 2},
        {"gloss": "thanks", "count": 2},
        {"gloss": "yes", "count": 2},
    ]


def test_flatten_alias_map_rejects_collisions() -> None:
    labels = ["thank you", "yes"]
    with pytest.raises(ValueError, match="Alias collision"):
        flatten_alias_map(
            labels,
            {
                "thank you": ["ty"],
                "yes": ["ty"],
            },
        )


def test_build_colab_anchor_contract_enforces_manifest_membership(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json", ["yes", "thank you"])
    records = _write_records(tmp_path / "records.jsonl", ["yes", "oops"])

    with pytest.raises(ValueError, match="not in manifest labels"):
        build_colab_anchor_contract(
            manifest_path=manifest,
            records_path=records,
            top_k=2,
            canonical_to_aliases={},
        )


def test_build_colab_anchor_contract_cli_writes_artifact(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json", ["yes", "thank you", "hello"])
    records = _write_records(
        tmp_path / "records.jsonl",
        ["yes", "thank you", "yes", "hello", "yes", "hello"],
    )
    alias_map = tmp_path / "aliases.json"
    alias_map.write_text(json.dumps({"thank you": ["thankyou"]}), encoding="utf-8")
    out = tmp_path / "contract.json"

    status = build_colab_anchor_contract_main(
        [
            "--manifest",
            str(manifest),
            "--records",
            str(records),
            "--alias-map",
            str(alias_map),
            "--out",
            str(out),
            "--top-k",
            "2",
        ]
    )

    assert status == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["contract_version"] == "colab_top50_anchor_contract_v1"
    assert payload["anchors"] == [
        {"gloss": "yes", "count": 3},
        {"gloss": "hello", "count": 2},
    ]
    assert {entry["alias"]: entry["canonical"] for entry in payload["alias_map"]}["thankyou"] == "thank you"
