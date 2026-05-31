"""Prompt-control reference fixture selection for stable Top-50 demos.

This module is intentionally scoped to prompt-control free generation. It
selects known-correct samples and writes a compact JSON fixture for later video
prompt-control checks. It does not run or encode constrained Top-50
inference.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.unsloth_asl import (
    GlossPredictor,
    RealUnslothASLGlossPredictor,
    build_prompt_control_prompt,
    evaluate_q64_records,
    load_manifest_labels,
    load_q64_jsonl,
)

REFERENCE_SCHEMA_VERSION = 1
REFERENCE_SCOPE = "prompt_control_reference_fixture"
REFERENCE_MODE = "prompt_control_free_generation"
REFERENCE_ARTIFACT_NAME = "reference.json"


@dataclass(frozen=True)
class PromptControlReferenceConfig:
    """Configuration for selecting a prompt-control reference fixture."""

    checkpoint_path: Path | str
    records_path: Path | str
    manifest_path: Path | str
    out_dir: Path | str
    demo_count: int = 5
    generated_at: str | None = None

    def __post_init__(self) -> None:
        if self.demo_count <= 0:
            raise ValueError("demo_count must be a positive integer.")


@dataclass(frozen=True)
class PromptControlReferenceResult:
    """Paths and payload produced by the prompt-control fixture builder."""

    artifact_path: Path
    payload: dict[str, Any]


def build_prompt_control_reference_fixture(
    config: PromptControlReferenceConfig,
    *,
    predictor: GlossPredictor | None = None,
    prediction_rows: Sequence[Mapping[str, Any]] | None = None,
) -> PromptControlReferenceResult:
    """Select correct prompt-control samples and write the reference artifact."""

    checkpoint_path = Path(config.checkpoint_path)
    records_path = Path(config.records_path)
    manifest_path = Path(config.manifest_path)
    out_dir = Path(config.out_dir)

    labels = load_manifest_labels(manifest_path)
    if prediction_rows is None:
        records = load_q64_jsonl(records_path)
        if not records:
            raise ValueError(f"No records found in {records_path}")
        checkpoint_predictor = predictor or RealUnslothASLGlossPredictor(
            checkpoint_path,
            prompt_builder=lambda record: build_prompt_control_prompt(record, labels),
        )
        rows, _ = evaluate_q64_records(records, checkpoint_predictor, labels)
    else:
        rows = [_normalize_prediction_row(row) for row in prediction_rows]

    selected = _select_reference_rows(rows, demo_count=config.demo_count)
    payload = _build_payload(
        checkpoint_path=checkpoint_path,
        records_path=records_path,
        manifest_path=manifest_path,
        rows=rows,
        selected=selected,
        demo_count=config.demo_count,
        generated_at=config.generated_at,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / REFERENCE_ARTIFACT_NAME
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PromptControlReferenceResult(artifact_path=artifact_path, payload=payload)


def load_prompt_control_prediction_rows(predictions_csv: Path | str) -> list[dict[str, Any]]:
    """Load prompt-control evaluator predictions for fixture selection."""

    path = Path(predictions_csv)
    required = {
        "sample_id",
        "expected_gloss",
        "predicted_gloss",
        "raw_model_output",
        "valid_label",
        "correct",
        "mode",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        return [_normalize_prediction_row(row) for row in reader]


def _select_reference_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    demo_count: int,
) -> list[dict[str, Any]]:
    target_count = demo_count + 1
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in rows:
        normalized = _normalize_prediction_row(row)
        sample_id = str(normalized["sample_id"])
        if sample_id in seen:
            continue
        if normalized["valid_label"] is True and normalized["correct"] is True:
            seen.add(sample_id)
            selected.append(normalized)
        if len(selected) == target_count:
            break

    if len(selected) < target_count:
        raise ValueError(
            "Not enough correct prompt-control samples for reference fixture: "
            f"needed {target_count}, found {len(selected)}."
        )
    return selected


def _build_payload(
    *,
    checkpoint_path: Path,
    records_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    demo_count: int,
    generated_at: str | None,
) -> dict[str, Any]:
    records = []
    for index, row in enumerate(selected):
        role = "smoke" if index == 0 else "demo"
        records.append(
            {
                "selection_role": role,
                "sample_id": str(row["sample_id"]),
                "expected_gloss": str(row["expected_gloss"]),
                "raw_model_output": str(row["raw_model_output"]),
                "normalized_gloss": str(row["predicted_gloss"]),
                "valid_label": bool(row["valid_label"]),
                "correct": bool(row["correct"]),
                "inference_mode": str(row["mode"]),
            }
        )

    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": REFERENCE_SCOPE,
        "mode": REFERENCE_MODE,
        "inference_contract": (
            "Prompt-control free generation only; constrained Top-50 inference is not part "
            "of this fixture."
        ),
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "records_path": str(records_path),
        "metadata": {
            "requested_demo_count": demo_count,
            "selected_count": len(records),
            "smoke_count": 1,
            "demo_count": len(records) - 1,
            "evaluated_row_count": len(rows),
        },
        "sample_ids": [record["sample_id"] for record in records],
        "records": records,
    }


def _normalize_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "expected_gloss": str(row["expected_gloss"]),
        "predicted_gloss": str(row["predicted_gloss"]),
        "raw_model_output": str(row["raw_model_output"]),
        "valid_label": _coerce_bool(row["valid_label"]),
        "correct": _coerce_bool(row["correct"]),
        "mode": str(row["mode"]),
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")
