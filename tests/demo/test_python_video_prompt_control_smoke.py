"""Behavior tests for Python video -> q64 -> prompt-control smoke path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.demo.run_python_video_prompt_control_smoke import main as python_video_prompt_control_main
from src.data.q64_encoding import ALPHABET
from src.demo.python_video_prompt_control import (
    PythonVideoPromptControlSmokeConfig,
    run_python_video_prompt_control_smoke,
)


def _reference_input(*, sample_id: str = "hearing_26986", frames: int = 2, features: int = 177) -> str:
    return (
        f"sample_id={sample_id}\n"
        f"encoding=q64_full clip=4 alphabet={ALPHABET}\n"
        f"frames={frames} features_per_frame={features}\n"
        f"pose_q64={'|'.join('W' * features for _ in range(frames))}"
    )


def _write_contract_files(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    manifest_path.write_text(json.dumps({"labels": ["hearing", "drink"]}), encoding="utf-8")
    records_path.write_text(
        json.dumps(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": _reference_input(),
                "output": "hearing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, records_path


class MockExtractor:
    def extract_from_video(self, video_path: Path, max_frames: int | None = None) -> dict[str, np.ndarray]:
        assert video_path.name == "hearing_26986.mp4"
        assert max_frames == 8
        return {
            "body": np.ones((4, 17, 4), dtype=np.float32),
            "left_hand": np.ones((4, 21, 4), dtype=np.float32) * 2,
            "right_hand": np.ones((4, 21, 4), dtype=np.float32) * 3,
        }

    def close(self) -> None:
        pass


class PromptControlPredictor:
    mode = "real"

    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output
        self.records: list[Mapping[str, Any]] = []

    def predict_raw(self, record: Mapping[str, Any]) -> str:
        self.records.append(record)
        return self.raw_output


def test_python_video_prompt_control_smoke_writes_scoped_readiness_artifact(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    video_path = tmp_path / "hearing_26986.mp4"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "python_video_prompt_control"
    video_path.write_bytes(b"mock video path used by injected extractor")
    checkpoint_path.mkdir()
    predictor = PromptControlPredictor("HEARING\nextra")

    result = run_python_video_prompt_control_smoke(
        PythonVideoPromptControlSmokeConfig(
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            sample_id="hearing_26986",
            expected_gloss="hearing",
            manifest_path=manifest_path,
            records_path=records_path,
            out_dir=out_dir,
            max_frames=8,
        ),
        extractor_factory=lambda: MockExtractor(),
        predictor=predictor,
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert predictor.records and "sample_id=hearing_26986" in predictor.records[0]["input"]
    assert result.correct is True
    assert artifact["scope"] == "python_video_prompt_control_smoke"
    assert artifact["claims"].endswith("one known video sample only; not production ASL recognition.")
    assert artifact["model_path"] == str(checkpoint_path)
    assert artifact["input_record_id"] == "hearing_26986"
    assert artifact["video_path"] == str(video_path)
    assert artifact["q64_jsonl_path"].endswith("_video_pose_q64_smoke.jsonl")
    assert artifact["inference_mode"] == "real"
    assert artifact["raw_model_output"] == "HEARING\nextra"
    assert artifact["normalized_gloss"] == "hearing"
    assert artifact["expected_gloss"] == "hearing"
    assert artifact["valid_label"] is True
    assert artifact["correct"] is True
    assert artifact["prompt_control"]["contract"] == "q64_prompt_control_free_generation"
    assert artifact["prompt_control"]["valid_labels"] == ["hearing", "drink"]
    assert artifact["timing_ms"]["total"] >= 0.0
    assert artifact["timing_ms"]["video_to_q64"] >= 0.0
    assert artifact["timing_ms"]["prompt_control_inference"] >= 0.0
    assert artifact["diagnostics"]["video_pose_q64_scope"] == "video_pose_q64_smoke"
    assert artifact["diagnostics"]["q64_encoding"] == "q64_full"


def test_python_video_prompt_control_cli_runs_with_mock_seams(tmp_path: Path, capsys) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    video_path = tmp_path / "hearing_26986.mp4"
    checkpoint_path = tmp_path / "checkpoint"
    out_dir = tmp_path / "python_video_prompt_control"
    video_path.write_bytes(b"mock video")
    checkpoint_path.mkdir()

    exit_code = python_video_prompt_control_main(
        [
            "--video-path",
            str(video_path),
            "--checkpoint",
            str(checkpoint_path),
            "--sample-id",
            "hearing_26986",
            "--expected-gloss",
            "hearing",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "8",
            "--mock-extractor",
            "--mock-model-output",
            "HEARING\nextra",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    artifact = json.loads((out_dir / "python_video_prompt_control_smoke_readiness.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["scope"] == "python_video_prompt_control_smoke"
    assert summary["input_record_id"] == "hearing_26986"
    assert summary["normalized_gloss"] == "hearing"
    assert summary["correct"] is True
    assert summary["artifact_path"] == str(out_dir / "python_video_prompt_control_smoke_readiness.json")
    assert artifact["raw_model_output"] == "HEARING\nextra"
