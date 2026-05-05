"""Cactus Engine prompt-control parity harness for q64 ASL records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.unsloth_asl import (
    build_prompt_control_prompt,
    load_manifest_labels,
    load_q64_jsonl,
    normalize_model_output,
)

PARITY_SCOPE = "cactus_prompt_control_parity"
PARITY_REPORT_NAME = "parity_report.json"


@dataclass(frozen=True)
class CactusPromptControlParityConfig:
    """Configuration for one Cactus prompt-control parity report run."""

    reference_path: Path | str
    records_path: Path | str
    manifest_path: Path | str
    cactus_weights_path: Path | str
    out_dir: Path | str
    max_samples: int = 1

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")


@dataclass(frozen=True)
class CactusCompletionResult:
    """Cactus completion result normalized for parity report construction."""

    raw_model_output: str
    success: bool = True
    error: str | None = None
    response_metadata: dict[str, Any] | None = None


class CactusPromptRunner(Protocol):
    """Prompt completion seam for mocked tests and real Cactus Engine runs."""

    runtime_mode: str

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        """Return one completion for the prompt-control prompt."""


@dataclass(frozen=True)
class CactusPromptControlParityResult:
    """Paths and payload produced by the Cactus parity harness."""

    report_path: Path
    payload: dict[str, Any]


class MockCactusPromptRunner:
    """Deterministic Cactus prompt runner for CI-safe parity tests."""

    runtime_mode = "mock"

    def __init__(self, output: str) -> None:
        self.output = output

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        return CactusCompletionResult(raw_model_output=self.output, response_metadata={})


class RealCactusEnginePromptRunner:
    """Cactus Engine Python SDK prompt runner.

    Missing weights or SDK/runtime failures are returned as completion errors so
    the harness can still write an honest report artifact.
    """

    runtime_mode = "cactus_engine"

    def __init__(self, cactus_weights_path: Path | str) -> None:
        self.cactus_weights_path = Path(cactus_weights_path)

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        if not self.cactus_weights_path.is_dir():
            return CactusCompletionResult(
                raw_model_output="",
                success=False,
                error=f"export: Cactus weights directory not found: {self.cactus_weights_path}",
                response_metadata={},
            )
        try:
            from src.cactus import cactus_complete, cactus_destroy, cactus_init
        except Exception as exc:
            return CactusCompletionResult(
                raw_model_output="",
                success=False,
                error=f"runtime: failed to import Cactus Engine Python SDK: {exc}",
                response_metadata={},
            )

        model = None
        try:
            model = cactus_init(str(self.cactus_weights_path), None, False)
            messages = json.dumps([{"role": "user", "content": prompt}])
            options = json.dumps({"temperature": 0.0, "max_tokens": 8})
            raw_response = cactus_complete(model, messages, options, None, None)
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return CactusCompletionResult(
                raw_model_output="",
                success=False,
                error=f"output-parsing: Cactus response was not valid JSON: {exc}",
                response_metadata={},
            )
        except Exception as exc:
            return CactusCompletionResult(
                raw_model_output="",
                success=False,
                error=f"runtime: Cactus completion failed: {exc}",
                response_metadata={},
            )
        finally:
            if model is not None:
                try:
                    cactus_destroy(model)
                except Exception:
                    pass

        success = bool(response_payload.get("success", True))
        response = response_payload.get("response", "")
        raw_model_output = "" if response is None else str(response)
        metadata = {
            key: value
            for key, value in response_payload.items()
            if key not in {"response", "success", "error"}
        }
        error = None if success else f"decoding: {response_payload.get('error', 'Cactus completion failed')}"
        return CactusCompletionResult(
            raw_model_output=raw_model_output,
            success=success,
            error=error,
            response_metadata=metadata,
        )


def run_cactus_prompt_control_parity(
    config: CactusPromptControlParityConfig,
    *,
    runner: CactusPromptRunner,
) -> CactusPromptControlParityResult:
    """Run selected prompt-control reference samples through a Cactus runner."""

    reference_path = Path(config.reference_path)
    records_path = Path(config.records_path)
    manifest_path = Path(config.manifest_path)
    out_dir = Path(config.out_dir)

    labels = load_manifest_labels(manifest_path)
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    records_by_sample_id = _index_records_by_sample_id(load_q64_jsonl(records_path))
    selected_references = _select_reference_records(reference_payload, config.max_samples)

    samples = []
    for reference_record in selected_references:
        sample_id = str(reference_record["sample_id"])
        python_gloss = reference_record.get("normalized_gloss")
        python_valid = bool(reference_record.get("valid_label"))
        expected_gloss = str(reference_record["expected_gloss"])
        runtime_error = None
        response_metadata: dict[str, Any] = {}
        try:
            q64_record = records_by_sample_id.get(sample_id)
            if q64_record is None:
                raise ValueError(f"q64 record not found for reference sample_id {sample_id}")
            prompt = build_prompt_control_prompt(q64_record, labels)
        except Exception as exc:
            prompt = None
            raw_model_output = ""
            runtime_error = str(exc)
            if not runtime_error.startswith("prompt:"):
                runtime_error = f"prompt: {runtime_error}"

        if prompt is not None:
            try:
                completion = runner.complete(prompt, sample_id=sample_id)
                raw_model_output = completion.raw_model_output
                response_metadata = completion.response_metadata or {}
                if not completion.success:
                    runtime_error = completion.error or "runtime: Cactus completion returned success=false"
                elif response_metadata.get("cloud_handoff") is True:
                    runtime_error = (
                        "runtime: Cactus completion used cloud_handoff=true; "
                        "local Cactus parity is not proven."
                    )
            except Exception as exc:
                raw_model_output = ""
                runtime_error = str(exc)
                if not runtime_error.startswith(
                    ("export:", "tokenizer:", "runtime:", "decoding:", "output-parsing:")
                ):
                    runtime_error = f"runtime: {runtime_error}"

        cactus_gloss, cactus_valid = normalize_model_output(raw_model_output, labels)
        normalized_matches = cactus_gloss == python_gloss
        valid_matches = cactus_valid == python_valid
        correct_matches = runtime_error is None and cactus_valid and cactus_gloss == expected_gloss

        samples.append(
            {
                "sample_id": sample_id,
                "selection_role": str(reference_record["selection_role"]),
                "expected_gloss": expected_gloss,
                "python_reference": {
                    "raw_model_output": str(reference_record["raw_model_output"]),
                    "normalized_gloss": python_gloss,
                    "valid_label": python_valid,
                },
                "cactus": {
                    "raw_model_output": raw_model_output,
                    "normalized_gloss": cactus_gloss,
                    "valid_label": cactus_valid,
                },
                "normalized_gloss_matches_python": normalized_matches,
                "valid_label_matches_python": valid_matches,
                "correct_matches_expected": correct_matches,
                "runtime_error": runtime_error,
                "cactus_response_metadata": response_metadata,
            }
        )

    summary = _build_summary(samples)
    payload = {
        "scope": PARITY_SCOPE,
        "runtime_mode": runner.runtime_mode,
        "real_cactus_parity_proven": runner.runtime_mode == "cactus_engine" and summary["all_matches"],
        "reference_checkpoint_path": str(reference_payload.get("checkpoint_path", "")),
        "cactus_weights_path": str(config.cactus_weights_path),
        "reference_path": str(reference_path),
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "summary": summary,
        "samples": samples,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / PARITY_REPORT_NAME
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CactusPromptControlParityResult(report_path=report_path, payload=payload)


def _select_reference_records(
    reference_payload: dict[str, Any],
    max_samples: int,
) -> list[dict[str, Any]]:
    records = list(reference_payload.get("records", []))
    smoke = [record for record in records if record.get("selection_role") == "smoke"]
    selected = smoke[:1] + [record for record in records if record.get("selection_role") != "smoke"]
    return selected[:max_samples]


def _build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    match_count = sum(
        1
        for sample in samples
        if sample["normalized_gloss_matches_python"]
        and sample["valid_label_matches_python"]
        and sample["correct_matches_expected"]
        and sample["runtime_error"] is None
    )
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
