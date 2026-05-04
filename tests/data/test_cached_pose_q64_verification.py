"""Behavior tests for cached pose to q64 verification artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.q64_encoding import ALPHABET
from scripts.data.verify_cached_pose_q64 import main as verify_cached_pose_q64_main
from src.data.cached_pose_q64 import CachedPoseQ64VerificationConfig, verify_cached_pose_q64


def _reference_input(*, sample_id: str = "hearing_26986", frames: int = 2, features: int = 177) -> str:
    return (
        f"sample_id={sample_id}\n"
        f"encoding=q64_full clip=4 alphabet={ALPHABET}\n"
        f"frames={frames} features_per_frame={features}\n"
        f"pose_q64={'|'.join('W' * features for _ in range(frames))}"
    )


def _write_contract_files(tmp_path: Path, *, reference_input: str | None = None) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    manifest_path.write_text(json.dumps({"labels": ["hearing", "drink"]}), encoding="utf-8")
    records_path.write_text(
        json.dumps(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": reference_input or _reference_input(),
                "output": "hearing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, records_path


def _write_pose_archive(path: Path, *, sample_id: str = "hearing_26986", frames: int = 2) -> None:
    np.savez(
        path,
        sample_id=np.asarray(sample_id),
        body=np.ones((frames, 17, 3), dtype=np.float32),
        left_hand=np.ones((frames, 21, 3), dtype=np.float32) * 2,
        right_hand=np.ones((frames, 21, 3), dtype=np.float32) * 3,
    )


def test_cached_pose_verification_writes_q64_record_and_dedicated_report(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    pose_path = tmp_path / "hearing_26986.npz"
    out_dir = tmp_path / "verification_artifacts"
    _write_pose_archive(pose_path)

    result = verify_cached_pose_q64(
        CachedPoseQ64VerificationConfig(
            pose_path=pose_path,
            sample_id="hearing_26986",
            expected_gloss="hearing",
            manifest_path=manifest_path,
            records_path=records_path,
            out_dir=out_dir,
        )
    )

    generated = json.loads(result.jsonl_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert set(generated) == {"instruction", "input", "output"}
    assert generated["output"] == "hearing"
    assert "sample_id=hearing_26986" in generated["input"]
    assert "encoding=q64_full clip=4 alphabet=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_" in generated["input"]
    assert "frames=2 features_per_frame=177" in generated["input"]
    assert "pose_q64=" in generated["input"]
    assert report["scope"] == "cached_pose_q64_verification"
    assert report["status"] == "ok"
    assert result.jsonl_path.parent == out_dir
    assert result.report_path.parent == out_dir
    assert result.report_path.name != "metrics.json"
    assert "readiness" not in result.report_path.name


def test_cached_pose_verification_fails_for_missing_pose_archive(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)

    with pytest.raises(FileNotFoundError, match="Pose archive not found"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=tmp_path / "missing.npz",
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_fails_for_malformed_pose_components(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    pose_path = tmp_path / "hearing_26986.npz"
    np.savez(
        pose_path,
        sample_id=np.asarray("hearing_26986"),
        body=np.ones((2, 17, 3), dtype=np.float32),
        left_hand=np.ones((1, 21, 3), dtype=np.float32),
        right_hand=np.ones((2, 21, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="share one frame count"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=pose_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_fails_for_empty_or_non_finite_component_axes(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    empty_component_path = tmp_path / "empty_component.npz"
    np.savez(
        empty_component_path,
        sample_id=np.asarray("hearing_26986"),
        body=np.ones((2, 0, 3), dtype=np.float32),
        left_hand=np.ones((2, 21, 3), dtype=np.float32),
        right_hand=np.ones((2, 21, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="non-empty joint and coordinate axes"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=empty_component_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )

    non_finite_path = tmp_path / "non_finite_component.npz"
    body = np.ones((2, 17, 3), dtype=np.float32)
    body[0, 0, 0] = np.nan
    np.savez(
        non_finite_path,
        sample_id=np.asarray("hearing_26986"),
        body=body,
        left_hand=np.ones((2, 21, 3), dtype=np.float32),
        right_hand=np.ones((2, 21, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="contains non-finite values"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=non_finite_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_fails_for_identity_and_gloss_mismatch(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    pose_path = tmp_path / "wrong_sample.npz"
    _write_pose_archive(pose_path, sample_id="wrong_sample")

    with pytest.raises(ValueError, match="sample_id mismatch"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=pose_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )

    _write_pose_archive(pose_path, sample_id="hearing_26986")
    with pytest.raises(ValueError, match="expected_gloss mismatch"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=pose_path,
                sample_id="hearing_26986",
                expected_gloss="drink",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_resamples_cached_frames_to_reference_q64_shape(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path, reference_input=_reference_input(frames=12, features=177))
    pose_path = tmp_path / "hearing_26986.npz"
    _write_pose_archive(pose_path, frames=60)

    result = verify_cached_pose_q64(
        CachedPoseQ64VerificationConfig(
            pose_path=pose_path,
            sample_id="hearing_26986",
            expected_gloss="hearing",
            manifest_path=manifest_path,
            records_path=records_path,
            out_dir=tmp_path / "verification_artifacts",
        )
    )

    generated = json.loads(result.jsonl_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert result.frames == 12
    assert "frames=12 features_per_frame=177" in generated["input"]
    assert len(generated["input"].split("pose_q64=", 1)[1].split("|")) == 12
    assert report["source_frames"] == 60


def test_cached_pose_verification_fails_for_reference_q64_feature_mismatch(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path, reference_input=_reference_input(frames=2, features=176))
    pose_path = tmp_path / "hearing_26986.npz"
    _write_pose_archive(pose_path)

    with pytest.raises(ValueError, match="reference q64 feature mismatch"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=pose_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_fails_for_reference_q64_payload_shape_mismatch(tmp_path: Path) -> None:
    bad_payload = (
        f"sample_id=hearing_26986\n"
        f"encoding=q64_full clip=4 alphabet={ALPHABET}\n"
        f"frames=2 features_per_frame=177\n"
        f"pose_q64={'W' * 176}|{'W' * 177}"
    )
    manifest_path, records_path = _write_contract_files(tmp_path, reference_input=bad_payload)
    pose_path = tmp_path / "hearing_26986.npz"
    _write_pose_archive(pose_path)

    with pytest.raises(ValueError, match="reference q64 payload shape mismatch"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=pose_path,
                sample_id="hearing_26986",
                expected_gloss="hearing",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_fails_for_invalid_label(tmp_path: Path) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)
    pose_path = tmp_path / "hearing_26986.npz"
    _write_pose_archive(pose_path)

    with pytest.raises(ValueError, match="not in manifest labels"):
        verify_cached_pose_q64(
            CachedPoseQ64VerificationConfig(
                pose_path=pose_path,
                sample_id="hearing_26986",
                expected_gloss="outside-label",
                manifest_path=manifest_path,
                records_path=records_path,
                out_dir=tmp_path / "verification_artifacts",
            )
        )


def test_cached_pose_verification_cli_reports_actionable_failure(tmp_path: Path, capsys) -> None:
    manifest_path, records_path = _write_contract_files(tmp_path)

    exit_code = verify_cached_pose_q64_main(
        [
            "--pose-path",
            str(tmp_path / "missing.npz"),
            "--sample-id",
            "hearing_26986",
            "--expected-gloss",
            "hearing",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--out-dir",
            str(tmp_path / "verification_artifacts"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Cached pose q64 verification failed:" in captured.err
    assert "Pose archive not found" in captured.err
