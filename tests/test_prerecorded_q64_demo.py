"""Behavior tests for prerecorded q64 Top-50 demo integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.run_prerecorded_q64_demo import main as run_prerecorded_q64_demo_main
from src.demo.prerecorded_q64 import PrerecordedQ64DemoConfig, run_prerecorded_q64_demo


class FakeCheckpointPredictor:
    """Small checkpoint stand-in that exercises the q64 prediction contract."""

    mode = "real"

    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output
        self.records: list[Mapping[str, Any]] = []

    def predict_raw(self, record: Mapping[str, Any]) -> str:
        self.records.append(record)
        return self.raw_output


def test_prerecorded_q64_demo_writes_scoped_visible_gloss_artifact(tmp_path: Path) -> None:
    record_path = tmp_path / "known_good.jsonl"
    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "demo_artifacts"
    checkpoint_path.mkdir()
    record_path.write_text(
        json.dumps(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": "sample_id=hearing_26986\nencoding=q64_full\npose_q64=abc",
                "output": "hearing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"labels": ["hearing", "drink"]}), encoding="utf-8")
    predictor = FakeCheckpointPredictor("HEARING\nextra text")

    result = run_prerecorded_q64_demo(
        PrerecordedQ64DemoConfig(
            checkpoint_path=checkpoint_path,
            records_path=record_path,
            manifest_path=manifest_path,
            record_id="hearing_26986",
            out_dir=out_dir,
        ),
        predictor=predictor,
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.output.display_text == "hearing"
    assert result.normalized_gloss == "hearing"
    assert result.valid_label is True
    assert predictor.records and predictor.records[0]["output"] == "hearing"
    assert artifact["scope"] == "demo_top50_prerecorded_q64"
    assert artifact["claims"] == "Demo-scoped Top-50 q64 pose-to-gloss checkpoint path; not production ASL recognition."
    assert artifact["model_path"] == str(checkpoint_path)
    assert artifact["input_record_id"] == "hearing_26986"
    assert artifact["inference_mode"] == "real"
    assert artifact["raw_prediction"] == "HEARING\nextra text"
    assert artifact["normalized_gloss"] == "hearing"
    assert artifact["visible_gloss"] == "hearing"
    assert artifact["valid_label"] is True
    assert artifact["confidence"] is None


def test_prerecorded_q64_demo_cli_runs_with_mock_checkpoint(tmp_path: Path, capsys) -> None:
    record_path = tmp_path / "known_good.jsonl"
    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "demo_artifacts"
    checkpoint_path.mkdir()
    record_path.write_text(
        json.dumps(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": "sample_id=drink_001\nencoding=q64_full\npose_q64=abc",
                "output": "drink",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"labels": ["drink"]}), encoding="utf-8")

    exit_code = run_prerecorded_q64_demo_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--records",
            str(record_path),
            "--manifest",
            str(manifest_path),
            "--record-id",
            "drink_001",
            "--out-dir",
            str(out_dir),
            "--mock",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    artifact = json.loads((out_dir / "prerecorded_q64_demo_readiness.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["scope"] == "demo_top50_prerecorded_q64"
    assert summary["model_path"] == str(checkpoint_path)
    assert summary["input_record_id"] == "drink_001"
    assert summary["inference_mode"] == "mock"
    assert summary["visible_gloss"] == "drink"
    assert summary["valid_label"] is True
    assert summary["artifact_path"] == str(out_dir / "prerecorded_q64_demo_readiness.json")
    assert artifact["claims"].endswith("not production ASL recognition.")


def test_prerecorded_q64_demo_cli_reports_real_checkpoint_load_failure(
    tmp_path: Path,
    capsys,
) -> None:
    record_path = tmp_path / "known_good.jsonl"
    manifest_path = tmp_path / "manifest.json"
    record_path.write_text(
        json.dumps(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": "sample_id=drink_001\nencoding=q64_full\npose_q64=abc",
                "output": "drink",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"labels": ["drink"]}), encoding="utf-8")

    exit_code = run_prerecorded_q64_demo_main(
        [
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--records",
            str(record_path),
            "--manifest",
            str(manifest_path),
            "--record-id",
            "drink_001",
            "--out-dir",
            str(tmp_path / "demo_artifacts"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Prerecorded q64 demo checkpoint run failed:" in captured.err
    assert "Checkpoint not found" in captured.err
