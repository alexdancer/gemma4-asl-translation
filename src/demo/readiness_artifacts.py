"""Demo readiness artifact module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEMO_SCOPE_PRERECORDED_Q64 = "demo_top50_prerecorded_q64"
DEMO_CLAIMS_PRERECORDED_Q64 = (
    "Demo-scoped Top-50 q64 pose-to-gloss checkpoint path; "
    "not production ASL recognition."
)
DEMO_SCOPE_CONSTRAINED_TOP50 = "demo_safe_constrained_top50"
DEMO_CLAIMS_CONSTRAINED_TOP50 = (
    "Diagnostic/demo-safe constrained Top-50 fallback; returns only canonical "
    "Top-50 labels, is not production ASL recognition, and is not the primary "
    "free-generation proof metric."
)


def write_prerecorded_q64_readiness_artifact(
    *,
    out_dir: Path | str,
    model_path: Path | str,
    input_record_id: str,
    inference_mode: str,
    raw_prediction: str,
    normalized_gloss: str | None,
    visible_gloss: str,
    status: str,
    valid_label: bool,
    confidence_proxy_used_for_display: float,
) -> Path:
    """Write the demo-scoped prerecorded q64 readiness artifact."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "prerecorded_q64_demo_readiness.json"
    payload = {
        "scope": DEMO_SCOPE_PRERECORDED_Q64,
        "claims": DEMO_CLAIMS_PRERECORDED_Q64,
        "model_path": str(model_path),
        "input_record_id": input_record_id,
        "inference_mode": inference_mode,
        "raw_prediction": raw_prediction,
        "normalized_gloss": normalized_gloss,
        "visible_gloss": visible_gloss,
        "status": status,
        "valid_label": valid_label,
        "confidence": None,
        "confidence_available": False,
        "confidence_proxy_used_for_display": confidence_proxy_used_for_display,
    }
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def write_constrained_top50_readiness_artifact(
    *,
    out_dir: Path | str,
    model_path: Path | str,
    input_record_id: str,
    inference_mode: str,
    selected_label: str,
    expected_gloss: str | None,
    correct: bool | None,
    visible_gloss: str,
    status: str,
    top_candidates: list[dict[str, float | str]],
    constrained_metadata: dict[str, Any],
    confidence_proxy_used_for_display: float,
) -> Path:
    """Write a separate demo-safe constrained Top-50 readiness artifact."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "demo_safe_constrained_top50_readiness.json"
    payload = {
        "scope": DEMO_SCOPE_CONSTRAINED_TOP50,
        "claims": DEMO_CLAIMS_CONSTRAINED_TOP50,
        "model_path": str(model_path),
        "input_record_id": input_record_id,
        "inference_mode": inference_mode,
        "selected_label": selected_label,
        "best_label": selected_label,
        "expected_gloss": expected_gloss,
        "correct": correct,
        "visible_gloss": visible_gloss,
        "status": status,
        "top_candidates": top_candidates,
        "confidence": None,
        "confidence_available": False,
        "confidence_proxy_used_for_display": confidence_proxy_used_for_display,
        "constrained_metadata": constrained_metadata,
    }
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artifact_path
