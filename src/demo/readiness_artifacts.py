"""Demo readiness artifact module."""

from __future__ import annotations

import json
from pathlib import Path

DEMO_SCOPE_PRERECORDED_Q64 = "demo_top50_prerecorded_q64"
DEMO_CLAIMS_PRERECORDED_Q64 = (
    "Demo-scoped Top-50 q64 pose-to-gloss checkpoint path; "
    "not production ASL recognition."
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
