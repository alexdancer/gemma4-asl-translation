"""Prerecorded Top-50 q64 checkpoint demo path.

This module is intentionally demo-scoped: it proves a known-good q64 record can
flow through a checkpoint predictor and the shared q64 prediction contract into
visible gloss output. It is not a production ASL recognizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.demo.output_contract import DemoOutput, DemoOutputConfig, format_demo_output
from src.evaluation.unsloth_asl import (
    GlossPredictor,
    RealUnslothASLGlossPredictor,
    infer_q64_record,
    load_manifest_labels,
    load_q64_jsonl,
)

DEMO_SCOPE = "demo_top50_prerecorded_q64"
DEMO_CLAIMS = (
    "Demo-scoped Top-50 q64 pose-to-gloss checkpoint path; "
    "not production ASL recognition."
)


@dataclass(frozen=True)
class PrerecordedQ64DemoConfig:
    """Configuration for one known-good prerecorded q64 demo run."""

    checkpoint_path: Path | str
    records_path: Path | str
    manifest_path: Path | str
    record_id: str
    out_dir: Path | str
    output_config: DemoOutputConfig = DemoOutputConfig()

    def __post_init__(self) -> None:
        if not str(self.record_id).strip():
            raise ValueError("record_id is required.")


@dataclass(frozen=True)
class PrerecordedQ64DemoResult:
    """Observable prerecorded q64 demo result for UI, logs, and readiness checks."""

    model_path: str
    input_record_id: str
    raw_prediction: str
    normalized_gloss: str | None
    valid_label: bool
    inference_mode: str
    output: DemoOutput
    artifact_path: Path


@dataclass(frozen=True)
class _Q64DemoPrediction:
    ok: bool
    prediction: str | None
    confidence: float
    latency_ms: float = 0.0
    latency_target_ms: float = 800.0
    error: str | None = None


def run_prerecorded_q64_demo(
    config: PrerecordedQ64DemoConfig,
    *,
    predictor: GlossPredictor | None = None,
) -> PrerecordedQ64DemoResult:
    """Run one known-good q64 record through checkpoint inference to demo output."""

    checkpoint_path = Path(config.checkpoint_path)
    records_path = Path(config.records_path)
    manifest_path = Path(config.manifest_path)
    out_dir = Path(config.out_dir)

    labels = load_manifest_labels(manifest_path)
    record = _select_record(load_q64_jsonl(records_path), config.record_id)
    checkpoint_predictor = predictor or RealUnslothASLGlossPredictor(checkpoint_path)
    inference = infer_q64_record(record, checkpoint_predictor, labels)

    output = format_demo_output(
        _Q64DemoPrediction(
            ok=inference.valid_label,
            prediction=inference.predicted_gloss,
            confidence=1.0 if inference.valid_label else 0.0,
            error=None if inference.valid_label else "prediction is outside the Top-50 demo label contract",
        ),
        config.output_config,
    )

    artifact_path = _write_demo_artifact(
        out_dir=out_dir,
        model_path=checkpoint_path,
        input_record_id=config.record_id,
        inference_mode=inference.mode,
        raw_prediction=inference.raw_model_output,
        normalized_gloss=inference.predicted_gloss,
        valid_label=inference.valid_label,
        output=output,
    )
    return PrerecordedQ64DemoResult(
        model_path=str(checkpoint_path),
        input_record_id=config.record_id,
        raw_prediction=inference.raw_model_output,
        normalized_gloss=inference.predicted_gloss,
        valid_label=inference.valid_label,
        inference_mode=inference.mode,
        output=output,
        artifact_path=artifact_path,
    )


def _select_record(records: list[dict[str, Any]], record_id: str) -> Mapping[str, Any]:
    for record in records:
        input_text = str(record.get("input", ""))
        if any(line == f"sample_id={record_id}" for line in input_text.splitlines()):
            return record
    raise ValueError(f"record_id not found in q64 records: {record_id}")


def _write_demo_artifact(
    *,
    out_dir: Path,
    model_path: Path,
    input_record_id: str,
    inference_mode: str,
    raw_prediction: str,
    normalized_gloss: str | None,
    valid_label: bool,
    output: DemoOutput,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "prerecorded_q64_demo_readiness.json"
    payload = {
        "scope": DEMO_SCOPE,
        "claims": DEMO_CLAIMS,
        "model_path": str(model_path),
        "input_record_id": input_record_id,
        "inference_mode": inference_mode,
        "raw_prediction": raw_prediction,
        "normalized_gloss": normalized_gloss,
        "visible_gloss": output.display_text,
        "status": output.status,
        "valid_label": valid_label,
        "confidence": None,
        "confidence_available": False,
        "confidence_proxy_used_for_display": output.confidence,
    }
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artifact_path
