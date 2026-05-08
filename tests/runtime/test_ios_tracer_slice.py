"""Behavior tests for issue #35 iOS tracer slice scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mobile.ios_tracer_slice import IOSTracerSliceConfig, run_ios_tracer_slice


def test_ios_tracer_slice_records_button_to_local_inference_and_ui_payload(tmp_path: Path) -> None:
    response_fixture = tmp_path / "local_cactus_response.json"
    response_fixture.write_text(
        json.dumps({"gloss": "HELLO", "confidence": 0.912, "runtime_mode": "local_cactus_mock"}),
        encoding="utf-8",
    )

    config = IOSTracerSliceConfig(
        local_response_fixture=response_fixture,
        output_path=tmp_path / "ios_tracer_result.json",
        bundle_response_filename="local_cactus_response.json",
    )

    result = run_ios_tracer_slice(config)

    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    assert artifact["scope"] == "ios_tracer_slice_1"
    assert artifact["user_action"] == {
        "type": "button_tap",
        "control_id": "run_local_cactus_inference",
        "label": "Run Local Cactus Inference",
    }

    inference = artifact["inference"]
    assert inference["triggered_by"] == "button_tap"
    assert inference["runtime_mode"] == "local_cactus_mock"
    assert inference["success"] is True

    ui = artifact["ui"]
    assert ui["predicted_gloss"] == "HELLO"
    assert ui["confidence"] == pytest.approx(0.912)
    assert ui["confidence_text"] == "91.2%"

    acceptance = artifact["acceptance"]
    assert acceptance["button_triggers_local_inference"] is True
    assert acceptance["ui_displays_gloss_and_confidence"] is True


def test_ios_tracer_slice_rejects_confidence_outside_unit_interval(tmp_path: Path) -> None:
    response_fixture = tmp_path / "local_cactus_response.json"
    response_fixture.write_text(json.dumps({"gloss": "HELLO", "confidence": 1.2}), encoding="utf-8")

    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        run_ios_tracer_slice(
            IOSTracerSliceConfig(
                local_response_fixture=response_fixture,
                output_path=tmp_path / "ios_tracer_result.json",
                bundle_response_filename="local_cactus_response.json",
            )
        )


def test_ios_swiftui_scaffold_contains_button_and_result_text_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    content_view = repo_root / "ios" / "ASL-App" / "ASL-App" / "ContentView.swift"
    inference_client = repo_root / "ios" / "ASL-App" / "ASL-App" / "LocalCactusInferenceClient.swift"

    assert content_view.exists(), "Expected iOS ContentView scaffold for issue #35"
    assert inference_client.exists(), "Expected local inference adapter scaffold for issue #35"

    content_source = content_view.read_text(encoding="utf-8")
    inference_source = inference_client.read_text(encoding="utf-8")

    assert 'Button("Run Cloud Translation")' in content_source
    assert "Primary Output" in content_source
    assert "Translation" in content_source
    assert "Confidence" in content_source

    assert "func infer(" in inference_source
    assert "uploadVideoData: Data?" in inference_source
    assert "uploadFilename: String?" in inference_source
    assert "request.setValue(requestID, forHTTPHeaderField: \"X-Request-ID\")" in inference_source
    assert "multipart/form-data; boundary=" in inference_source
    assert "cloud_endpoint_success" in inference_source
    assert "cloud_endpoint_error" in inference_source
    assert "cloud_endpoint_unreachable" in inference_source
    assert "maxRetryAttempts = 1" in inference_source
    assert "cloud_endpoint_retry_exhausted" in inference_source
    assert "for attempt in 0...maxRetryAttempts" in inference_source
    assert "failure?.retryable" in inference_source
    assert "mapCloudErrorMessage(errorCode: failure?.errorCode, fallbackMessage: failure?.message, exhaustedRetry: attempt >= maxRetryAttempts)" in inference_source
    assert "let mimeType = ext == \"mp4\" ? \"video/mp4\" : \"video/quicktime\"" in inference_source
    assert "Button(\"Real Proof Run\")" in content_source
    assert "private actor InferenceArtifactLogger" in content_source
    assert "session_index.json" in content_source
    assert "UUID().uuidString" in content_source
