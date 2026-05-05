"""Behavior tests for real video to q64 smoke artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.smoke_video_pose_q64 import main as smoke_video_pose_q64_main
from src.data.q64_encoding import ALPHABET
from src.data.video_pose_q64_smoke import (
    VideoPoseQ64SmokeConfig,
    VideoPoseQ64SmokeError,
    run_video_pose_q64_smoke,
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
        body = np.ones((4, 17, 4), dtype=np.float32)
        left_hand = np.ones((4, 21, 4), dtype=np.float32) * 2
        right_hand = np.ones((4, 21, 4), dtype=np.float32) * 3
        return {"body": body, "left_hand": left_hand, "right_hand": right_hand}

    def close(self) -> None:
        pass


def test_video_pose_q64_smoke_writes_record_and_extraction_report(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    video_path = tmp_path / "hearing_26986.mp4"
    video_path.write_bytes(b"mock video path used by injected extractor")
    out_dir = tmp_path / "video_smoke"

    result = run_video_pose_q64_smoke(
        VideoPoseQ64SmokeConfig(
            video_path=video_path,
            sample_id="hearing_26986",
            expected_gloss="hearing",
            manifest_path=manifest_path,
            records_path=records_path,
            out_dir=out_dir,
            max_frames=8,
        ),
        extractor_factory=lambda: MockExtractor(),
    )

    generated = json.loads(result.jsonl_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert set(generated) == {"instruction", "input", "output"}
    assert generated["output"] == "hearing"
    assert "sample_id=hearing_26986" in generated["input"]
    assert "encoding=q64_full clip=4 alphabet=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_" in generated["input"]
    assert "frames=2 features_per_frame=177" in generated["input"]
    assert "pose_q64=" in generated["input"]

    assert report["scope"] == "video_pose_q64_smoke"
    assert report["status"] == "ok"
    assert report["sample_id"] == "hearing_26986"
    assert report["expected_gloss"] == "hearing"
    assert report["video_path"] == str(video_path)
    assert report["extraction"]["extractor"] == "MockExtractor"
    assert report["extraction"]["requested_max_frames"] == 8
    assert report["coverage"]["source_frames"] == 4
    assert report["coverage"]["q64_frames"] == 2
    assert report["coverage"]["features_per_frame"] == 177
    assert report["coverage"]["components"] == {
        "body": {"frames": 4, "joints": 17, "coordinates": 4, "covered_frames": 4},
        "left_hand": {"frames": 4, "joints": 21, "coordinates": 4, "covered_frames": 4},
        "right_hand": {"frames": 4, "joints": 21, "coordinates": 4, "covered_frames": 4},
    }
    assert report["q64"] == {
        "encoding": "q64_full",
        "alphabet": ALPHABET,
        "clip": 4,
        "stride": 1,
        "jsonl_path": str(result.jsonl_path),
    }


def test_video_pose_q64_smoke_reports_actionable_extractor_dependency_failure(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    video_path = tmp_path / "hearing_26986.mp4"
    video_path.write_bytes(b"mock video")

    def missing_mediapipe() -> MockExtractor:
        raise ImportError("mediapipe is required for PoseExtractor")

    with pytest.raises(VideoPoseQ64SmokeError, match="Install MediaPipe/OpenCV dependencies"):
        run_video_pose_q64_smoke(
            VideoPoseQ64SmokeConfig(
                video_path=video_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "video_smoke",
            ),
            extractor_factory=missing_mediapipe,
        )


def test_video_pose_q64_smoke_cli_writes_artifacts_with_mock_extractor(tmp_path: Path, capsys) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    video_path = tmp_path / "hearing_26986.mp4"
    video_path.write_bytes(b"mock video")
    out_dir = tmp_path / "video_smoke"

    exit_code = smoke_video_pose_q64_main(
        [
            "--video-path",
            str(video_path),
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
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    report = json.loads((out_dir / "video_pose_q64_smoke_report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["scope"] == "video_pose_q64_smoke"
    assert summary["sample_id"] == "hearing_26986"
    assert summary["frames"] == 2
    assert summary["features_per_frame"] == 177
    assert report["extraction"]["extractor"] == "MockVideoPoseExtractor"


def test_video_pose_q64_smoke_cli_reports_actionable_missing_video(tmp_path: Path, capsys) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)

    exit_code = smoke_video_pose_q64_main(
        [
            "--video-path",
            str(tmp_path / "missing.mp4"),
            "--sample-id",
            "hearing_26986",
            "--expected-gloss",
            "hearing",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--out-dir",
            str(tmp_path / "video_smoke"),
            "--mock-extractor",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Video pose q64 smoke failed:" in captured.err
    assert "Video file not found" in captured.err
