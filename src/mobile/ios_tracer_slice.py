"""Issue #35 tracer slice: button-to-local-Cactus inference scaffold for iOS."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACER_SCOPE = "ios_tracer_slice_1"


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = repo_root.expanduser().resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class IOSTracerSliceConfig:
    local_response_fixture: Path
    output_path: Path = Path("artifacts/ios_tracer_slice/local_inference_result.json")
    bundle_response_filename: str = "local_cactus_response.json"
    repo_root: Path = Path(".")


@dataclass(frozen=True)
class IOSTracerSliceResult:
    artifact_path: str
    acceptance_proof_satisfied: bool


def _load_local_response(path: Path) -> tuple[str, float, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gloss = str(payload.get("gloss", "")).strip()
    if not gloss:
        raise ValueError("gloss must be a non-empty string")

    confidence_raw = payload.get("confidence")
    confidence = float(confidence_raw)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    runtime_mode = str(payload.get("runtime_mode", "local_cactus_mock")).strip() or "local_cactus_mock"
    return gloss, confidence, runtime_mode


def run_ios_tracer_slice(config: IOSTracerSliceConfig) -> IOSTracerSliceResult:
    gloss, confidence, runtime_mode = _load_local_response(config.local_response_fixture)

    artifact_payload = {
        "scope": TRACER_SCOPE,
        "captured_at": _utc_now(),
        "user_action": {
            "type": "button_tap",
            "control_id": "run_local_cactus_inference",
            "label": "Run Local Cactus Inference",
        },
        "inference": {
            "triggered_by": "button_tap",
            "runtime_mode": runtime_mode,
            "success": True,
            "response_fixture": _to_repo_relative(config.local_response_fixture, config.repo_root),
            "bundle_response_filename": config.bundle_response_filename,
        },
        "ui": {
            "predicted_gloss": gloss,
            "confidence": confidence,
            "confidence_text": f"{confidence * 100:.1f}%",
        },
        "acceptance": {
            "button_triggers_local_inference": True,
            "ui_displays_gloss_and_confidence": True,
            "device_runtime_validated": False,
            "device_runtime_follow_up": "Run ASLTracerSliceApp on target iPhone and capture screenshot/video proof.",
        },
    }
    _write_json(config.output_path, artifact_payload)

    return IOSTracerSliceResult(
        artifact_path=str(config.output_path.resolve()),
        acceptance_proof_satisfied=False,
    )


def result_to_dict(result: IOSTracerSliceResult) -> dict[str, Any]:
    return asdict(result)
