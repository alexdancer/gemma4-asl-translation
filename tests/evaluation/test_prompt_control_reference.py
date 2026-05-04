"""Behavior tests for prompt-control reference fixture selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluation.build_prompt_control_reference import main as build_prompt_control_reference_main
from src.demo.prompt_control_reference import (
    REFERENCE_MODE,
    REFERENCE_SCOPE,
    PromptControlReferenceConfig,
    build_prompt_control_reference_fixture,
)


class ExactOutputPredictor:
    """Deterministic predictor that makes selected records correct without a model."""

    mode = "mock"

    def predict_raw(self, record: Mapping[str, Any]) -> str:
        return str(record["output"]).upper()


def test_prompt_control_reference_writes_expected_fixture_shape(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "missing-checkpoint"
    out_dir = tmp_path / "reference"
    _write_records(
        records,
        [
            ("smoke_001", "hello"),
            ("demo_001", "thanks"),
            ("demo_002", "yes"),
        ],
    )
    manifest.write_text(json.dumps({"labels": ["hello", "thanks", "yes"]}), encoding="utf-8")

    result = build_prompt_control_reference_fixture(
        PromptControlReferenceConfig(
            checkpoint_path=checkpoint,
            records_path=records,
            manifest_path=manifest,
            out_dir=out_dir,
            demo_count=2,
            generated_at="2026-05-03T00:00:00+00:00",
        ),
        predictor=ExactOutputPredictor(),
    )

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.artifact_path == out_dir / "reference.json"
    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-05-03T00:00:00+00:00"
    assert payload["scope"] == REFERENCE_SCOPE
    assert payload["mode"] == REFERENCE_MODE
    assert "constrained Top-50 inference is not part" in payload["inference_contract"]
    assert payload["checkpoint_path"] == str(checkpoint)
    assert payload["manifest_path"] == str(manifest)
    assert payload["records_path"] == str(records)
    assert payload["sample_ids"] == ["smoke_001", "demo_001", "demo_002"]
    assert payload["metadata"] == {
        "requested_demo_count": 2,
        "selected_count": 3,
        "smoke_count": 1,
        "demo_count": 2,
        "evaluated_row_count": 3,
    }
    assert [record["selection_role"] for record in payload["records"]] == [
        "smoke",
        "demo",
        "demo",
    ]
    assert all(record["valid_label"] is True for record in payload["records"])
    assert all(record["correct"] is True for record in payload["records"])
    assert payload["records"][0]["raw_model_output"] == "HELLO"
    assert payload["records"][0]["normalized_gloss"] == "hello"
    assert sorted(path.name for path in out_dir.iterdir()) == ["reference.json"]


def test_prompt_control_reference_selects_only_correct_prediction_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    records = tmp_path / "records.jsonl"
    out_dir = tmp_path / "reference"
    manifest.write_text(json.dumps({"labels": ["hello", "thanks", "yes"]}), encoding="utf-8")
    _write_records(records, [("ignored", "hello")])

    result = build_prompt_control_reference_fixture(
        PromptControlReferenceConfig(
            checkpoint_path="checkpoints/top50",
            records_path=records,
            manifest_path=manifest,
            out_dir=out_dir,
            demo_count=1,
            generated_at="2026-05-03T00:00:00+00:00",
        ),
        prediction_rows=[
            {
                "sample_id": "bad_invalid",
                "expected_gloss": "hello",
                "predicted_gloss": "hello maybe",
                "raw_model_output": "hello maybe",
                "valid_label": False,
                "correct": False,
                "mode": "real",
            },
            {
                "sample_id": "smoke",
                "expected_gloss": "thanks",
                "predicted_gloss": "thanks",
                "raw_model_output": "THANKS",
                "valid_label": True,
                "correct": True,
                "mode": "real",
            },
            {
                "sample_id": "demo",
                "expected_gloss": "yes",
                "predicted_gloss": "yes",
                "raw_model_output": "yes",
                "valid_label": True,
                "correct": True,
                "mode": "real",
            },
        ],
    )

    assert result.payload["sample_ids"] == ["smoke", "demo"]
    assert [record["selection_role"] for record in result.payload["records"]] == ["smoke", "demo"]
    assert result.payload["records"][0]["inference_mode"] == "real"


def test_prompt_control_reference_cli_mock_writes_only_reference_json(
    tmp_path: Path,
    capsys,
) -> None:
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "prompt_control_reference"
    _write_records(records, [("a", "hello"), ("b", "hello"), ("c", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    exit_code = build_prompt_control_reference_main(
        [
            "--mock",
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--records",
            str(records),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--demo-count",
            "2",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    artifact = json.loads((out_dir / "reference.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["scope"] == REFERENCE_SCOPE
    assert summary["mode"] == REFERENCE_MODE
    assert summary["artifact_path"] == str(out_dir / "reference.json")
    assert summary["selected_count"] == 3
    assert summary["smoke_sample_id"] == "a"
    assert summary["demo_sample_ids"] == ["b", "c"]
    assert artifact["records"][0]["selection_role"] == "smoke"
    assert sorted(path.name for path in out_dir.iterdir()) == ["reference.json"]


def _write_records(path: Path, samples: list[tuple[str, str]]) -> None:
    rows = [
        {
            "instruction": "Classify this compact ASL pose encoding.",
            "input": f"sample_id={sample_id}\nencoding=q64_full\npose_q64=abc",
            "output": gloss,
        }
        for sample_id, gloss in samples
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
