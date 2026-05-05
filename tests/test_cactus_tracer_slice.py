"""Behavior tests for issue #32 Cactus tracer slice artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.mobile.cactus_tracer_slice import (
    TracerSliceConfig,
    run_cactus_tracer_slice,
    run_local_completion,
)


def test_run_cactus_tracer_slice_writes_required_artifacts_with_fallback(tmp_path: Path) -> None:
    checkpoint = tmp_path / "baseline-checkpoint"
    checkpoint.mkdir()

    config = TracerSliceConfig(
        checkpoint_path=checkpoint,
        output_root=tmp_path / "artifacts",
        conversion_output_version="v1",
        git_sha="abc123deadbeef",
        allow_real_export=False,
        prompt="Return one ASL gloss label.",
    )

    result = run_cactus_tracer_slice(config)

    freeze_payload = json.loads(Path(result.freeze_metadata_path).read_text(encoding="utf-8"))
    assert freeze_payload["checkpoint_id"] == "baseline-checkpoint"
    assert freeze_payload["git_sha"] == "abc123deadbeef"
    assert freeze_payload["conversion_output_version"] == "v1"

    converted_dir = Path(result.converted_weights_dir)
    assert converted_dir.name == "v1"
    assert (converted_dir / "conversion_manifest.json").exists()

    completion_payload = json.loads(Path(result.completion_artifact_path).read_text(encoding="utf-8"))
    assert completion_payload["runtime_mode"] == "deterministic_fallback"
    assert completion_payload["success"] is False
    assert isinstance(completion_payload["response"], str)
    assert isinstance(completion_payload["error"], str)
    assert completion_payload["timing_ms"] >= 0

    summary_payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary_payload["acceptance_proof_satisfied"] is False
    assert result.acceptance_proof_satisfied is False


def test_local_completion_fallback_reports_failure_for_acceptance(tmp_path: Path) -> None:
    artifact_path = tmp_path / "completion.json"
    payload = run_local_completion(
        weights_dir=tmp_path / "weights",
        prompt="Classify this sample.",
        artifact_path=artifact_path,
        repo_root=tmp_path,
        prefer_cactus_engine=True,
    )

    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved == payload
    assert {"success", "error", "response", "timing_ms", "runtime_mode"}.issubset(saved)
    assert saved["runtime_mode"] == "deterministic_fallback"
    assert saved["success"] is False
    assert isinstance(saved["error"], str)


def test_local_completion_without_cactus_attempt_is_not_success(tmp_path: Path) -> None:
    artifact_path = tmp_path / "completion.json"
    payload = run_local_completion(
        weights_dir=tmp_path / "weights",
        prompt="Classify this sample.",
        artifact_path=artifact_path,
        repo_root=tmp_path,
        prefer_cactus_engine=False,
    )

    assert payload["success"] is False
    assert payload["runtime_mode"] == "deterministic_fallback"
    assert payload["error"] == "Cactus engine not attempted."
