"""Issue #34 parity tracer slice: one smoke sample Python vs Cactus report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.unsloth_asl import (
    build_prompt_control_prompt,
    load_manifest_labels,
    load_q64_jsonl,
    normalize_model_output,
)
from src.mobile.cactus_prompt_control_parity import RealCactusEnginePromptRunner

PARITY_SCOPE = "parity_tracer_slice"
PARITY_REPORT_NAME = "parity_report_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ParityTracerSliceConfig:
    reference_path: Path | str
    records_path: Path | str
    manifest_path: Path | str
    cactus_weights_path: Path | str
    out_dir: Path | str = Path("artifacts/cactus_tracer")
    max_samples: int = 1

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")


@dataclass(frozen=True)
class ParityCompletionResult:
    raw_model_output: str
    success: bool = True
    error: str | None = None
    runtime_metadata: dict[str, Any] | None = None


class ParityPromptRunner(Protocol):
    runtime_mode: str

    def complete(self, prompt: str, *, sample_id: str) -> ParityCompletionResult:
        """Run one completion for the prepared prompt."""


@dataclass(frozen=True)
class ParityTracerSliceResult:
    report_path: Path
    payload: dict[str, Any]


class MockParityPromptRunner:
    runtime_mode = "mock"

    def __init__(self, output: str) -> None:
        self.output = output

    def complete(self, prompt: str, *, sample_id: str) -> ParityCompletionResult:
        _ = prompt
        _ = sample_id
        return ParityCompletionResult(raw_model_output=self.output, runtime_metadata={})


class ReferencePythonPromptRunner:
    """Python-path runner backed by frozen Python reference outputs."""

    runtime_mode = "python_reference"

    def __init__(self, rows_by_sample_id: dict[str, dict[str, Any]]) -> None:
        self.rows_by_sample_id = rows_by_sample_id

    @classmethod
    def from_reference_payload(cls, payload: dict[str, Any]) -> ReferencePythonPromptRunner:
        rows_by_sample_id: dict[str, dict[str, Any]] = {}
        for row in payload.get("records", []):
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                rows_by_sample_id[sample_id] = row
        return cls(rows_by_sample_id)

    def complete(self, prompt: str, *, sample_id: str) -> ParityCompletionResult:
        _ = prompt
        row = self.rows_by_sample_id.get(sample_id)
        if row is None:
            return ParityCompletionResult(
                raw_model_output="",
                success=False,
                error=f"prompt: python reference missing for sample_id {sample_id}",
                runtime_metadata={"source": "reference_fixture"},
            )
        return ParityCompletionResult(
            raw_model_output=str(row.get("raw_model_output", "")),
            runtime_metadata={"source": "reference_fixture"},
        )


class RealCactusParityPromptRunner:
    """Adapter for the real Cactus Engine runner."""

    runtime_mode = "cactus_engine"

    def __init__(self, cactus_weights_path: Path | str) -> None:
        self._runner = RealCactusEnginePromptRunner(cactus_weights_path)

    def complete(self, prompt: str, *, sample_id: str) -> ParityCompletionResult:
        result = self._runner.complete(prompt, sample_id=sample_id)
        return ParityCompletionResult(
            raw_model_output=result.raw_model_output,
            success=result.success,
            error=result.error,
            runtime_metadata=result.response_metadata,
        )


def run_parity_tracer_slice(
    config: ParityTracerSliceConfig,
    *,
    python_runner: ParityPromptRunner,
    cactus_runner: ParityPromptRunner,
) -> ParityTracerSliceResult:
    reference_path = Path(config.reference_path)
    records_path = Path(config.records_path)
    manifest_path = Path(config.manifest_path)
    out_dir = Path(config.out_dir)

    labels = load_manifest_labels(manifest_path)
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    records_by_sample_id = _index_records_by_sample_id(load_q64_jsonl(records_path))
    selected_references = _select_reference_records(reference_payload, config.max_samples)

    samples: list[dict[str, Any]] = []
    for reference_record in selected_references:
        sample_id = str(reference_record["sample_id"])
        expected_gloss = str(reference_record["expected_gloss"])
        runtime_errors: list[str] = []

        python_raw = ""
        python_metadata: dict[str, Any] = {}
        cactus_raw = ""
        cactus_metadata: dict[str, Any] = {}

        try:
            q64_record = records_by_sample_id.get(sample_id)
            if q64_record is None:
                raise ValueError(f"q64 record not found for reference sample_id {sample_id}")
            prompt = build_prompt_control_prompt(q64_record, labels)
        except Exception as exc:
            prompt = None
            runtime_errors.append(_normalize_runtime_error(str(exc), prefix="prompt:"))

        if prompt is not None:
            python_raw, python_metadata, python_error = _run_runner(
                python_runner,
                prompt,
                sample_id=sample_id,
                runtime_subject="Python",
            )
            cactus_raw, cactus_metadata, cactus_error = _run_runner(
                cactus_runner,
                prompt,
                sample_id=sample_id,
                runtime_subject="Cactus",
            )
            if python_error is not None:
                runtime_errors.append(python_error)
            if cactus_error is not None:
                runtime_errors.append(cactus_error)

        python_gloss, python_valid = normalize_model_output(python_raw, labels)
        cactus_gloss, cactus_valid = normalize_model_output(cactus_raw, labels)

        normalized_gloss_match = python_gloss == cactus_gloss
        valid_label_match = python_valid == cactus_valid
        python_matches_expected = python_valid and python_gloss == expected_gloss
        cactus_matches_expected = cactus_valid and cactus_gloss == expected_gloss
        runtime_error = "; ".join(runtime_errors) if runtime_errors else None
        parity_pass = (
            runtime_error is None
            and normalized_gloss_match
            and valid_label_match
            and python_matches_expected
            and cactus_matches_expected
        )

        samples.append(
            {
                "sample_id": sample_id,
                "selection_role": str(reference_record.get("selection_role", "unknown")),
                "expected_gloss": expected_gloss,
                "python": {
                    "raw_model_output": python_raw,
                    "normalized_gloss": python_gloss,
                    "valid_label": python_valid,
                },
                "cactus": {
                    "raw_model_output": cactus_raw,
                    "normalized_gloss": cactus_gloss,
                    "valid_label": cactus_valid,
                },
                "normalized_gloss_match": normalized_gloss_match,
                "valid_label_match": valid_label_match,
                "python_matches_expected": python_matches_expected,
                "cactus_matches_expected": cactus_matches_expected,
                "parity_pass": parity_pass,
                "runtime_error": runtime_error,
                "runtime": {
                    "python": {
                        "runtime_mode": python_runner.runtime_mode,
                        "runtime_metadata": python_metadata,
                    },
                    "cactus": {
                        "runtime_mode": cactus_runner.runtime_mode,
                        "runtime_metadata": cactus_metadata,
                    },
                },
            }
        )

    summary = _build_summary(samples)
    payload = {
        "scope": PARITY_SCOPE,
        "python_runtime_mode": python_runner.runtime_mode,
        "cactus_runtime_mode": cactus_runner.runtime_mode,
        "reference_checkpoint_path": str(reference_payload.get("checkpoint_path", "")),
        "cactus_weights_path": str(config.cactus_weights_path),
        "reference_path": str(reference_path),
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "summary": summary,
        "samples": samples,
        "captured_at": _utc_now(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / PARITY_REPORT_NAME
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ParityTracerSliceResult(report_path=report_path, payload=payload)


def _run_runner(
    runner: ParityPromptRunner,
    prompt: str,
    *,
    sample_id: str,
    runtime_subject: str,
) -> tuple[str, dict[str, Any], str | None]:
    try:
        completion = runner.complete(prompt, sample_id=sample_id)
        metadata = completion.runtime_metadata or {}
        error = None
        if not completion.success:
            error = completion.error or f"runtime: {runtime_subject} completion returned success=false"
        return completion.raw_model_output, metadata, _normalize_runtime_error(error) if error else None
    except Exception as exc:
        return "", {}, _normalize_runtime_error(str(exc), prefix="runtime:")


def _normalize_runtime_error(error: str, *, prefix: str = "runtime:") -> str:
    normalized = error.strip()
    if not normalized:
        return prefix
    if normalized.startswith(("prompt:", "runtime:", "export:", "decoding:", "output-parsing:", "tokenizer:")):
        return normalized
    return f"{prefix} {normalized}" if not prefix.endswith(" ") else f"{prefix}{normalized}"


def _select_reference_records(reference_payload: dict[str, Any], max_samples: int) -> list[dict[str, Any]]:
    records = list(reference_payload.get("records", []))
    smoke = [record for record in records if record.get("selection_role") == "smoke"]
    selected = smoke[:1] + [record for record in records if record.get("selection_role") != "smoke"]
    return selected[:max_samples]


def _build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    match_count = sum(1 for sample in samples if sample["parity_pass"])
    runtime_error_count = sum(1 for sample in samples if sample["runtime_error"] is not None)
    return {
        "sample_count": len(samples),
        "match_count": match_count,
        "mismatch_count": len(samples) - match_count,
        "runtime_error_count": runtime_error_count,
        "all_matches": len(samples) > 0 and match_count == len(samples),
    }


def _index_records_by_sample_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records_by_sample_id: dict[str, dict[str, Any]] = {}
    seen_indices: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        sample_id = _record_sample_id(record)
        if sample_id in records_by_sample_id:
            raise ValueError(
                f"duplicate q64 sample_id {sample_id!r} at records {seen_indices[sample_id]} and {index}."
            )
        records_by_sample_id[sample_id] = record
        seen_indices[sample_id] = index
    return records_by_sample_id


def _record_sample_id(record: dict[str, Any]) -> str:
    input_text = str(record.get("input", ""))
    for line in input_text.splitlines():
        if line.startswith("sample_id="):
            return line.split("=", 1)[1].strip()
    raise ValueError("q64 record is missing sample_id in input.")
