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
    pytest.skip("Removed: iOS SwiftUI client was hard-deleted; RN app is primary client.")
