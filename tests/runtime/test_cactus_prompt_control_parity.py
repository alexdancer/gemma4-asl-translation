"""Behavior tests for the Cactus prompt-control parity harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.mobile.run_cactus_prompt_control_parity import main as cactus_parity_main
from src.mobile.cactus_prompt_control_parity import (
    CactusCompletionResult,
    CactusPromptControlParityConfig,
    MockCactusPromptRunner,
    RealCactusEnginePromptRunner,
    run_cactus_prompt_control_parity,
)


def test_matching_mock_cactus_output_writes_passing_report(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello", "thanks"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=MockCactusPromptRunner("hello"),
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert result.report_path == out_dir / "parity_report.json"
    assert payload["scope"] == "cactus_prompt_control_parity"
    assert payload["runtime_mode"] == "mock"
    assert payload["real_cactus_parity_proven"] is False
    assert payload["reference_checkpoint_path"] == "checkpoints/reference"
    assert payload["cactus_weights_path"] == "mock-cactus-weights"
    assert payload["reference_path"] == str(reference)
    assert payload["records_path"] == str(records)
    assert payload["manifest_path"] == str(manifest)
    assert payload["summary"] == {
        "sample_count": 1,
        "match_count": 1,
        "mismatch_count": 0,
        "runtime_error_count": 0,
        "all_matches": True,
    }
    assert payload["samples"] == [
        {
            "sample_id": "smoke_001",
            "selection_role": "smoke",
            "expected_gloss": "hello",
            "python_reference": {
                "raw_model_output": "HELLO",
                "normalized_gloss": "hello",
                "valid_label": True,
            },
            "cactus": {
                "raw_model_output": "hello",
                "normalized_gloss": "hello",
                "valid_label": True,
            },
            "normalized_gloss_matches_python": True,
            "valid_label_matches_python": True,
            "correct_matches_expected": True,
            "runtime_error": None,
            "cactus_response_metadata": {},
        }
    ]


def test_mismatched_normalized_output_writes_failing_report(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello", "thanks"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=MockCactusPromptRunner("thanks"),
    )

    sample = result.payload["samples"][0]
    assert result.payload["summary"]["all_matches"] is False
    assert result.payload["summary"]["match_count"] == 0
    assert result.payload["summary"]["mismatch_count"] == 1
    assert sample["cactus"]["normalized_gloss"] == "thanks"
    assert sample["cactus"]["valid_label"] is True
    assert sample["normalized_gloss_matches_python"] is False
    assert sample["valid_label_matches_python"] is True
    assert sample["correct_matches_expected"] is False
    assert sample["runtime_error"] is None


def test_runner_exception_is_captured_as_runtime_error(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=FailingCactusPromptRunner(),
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    assert payload["summary"]["all_matches"] is False
    assert payload["summary"]["runtime_error_count"] == 1
    assert sample["cactus"] == {
        "raw_model_output": "",
        "normalized_gloss": None,
        "valid_label": False,
    }
    assert sample["normalized_gloss_matches_python"] is False
    assert sample["valid_label_matches_python"] is False
    assert sample["correct_matches_expected"] is False
    assert sample["runtime_error"] == "runtime: Cactus completion failed"
    assert sample["cactus_response_metadata"] == {}


def test_invalid_cactus_output_marks_invalid_label_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=MockCactusPromptRunner("hello maybe"),
    )

    sample = result.payload["samples"][0]
    assert sample["cactus"]["normalized_gloss"] == "hello maybe"
    assert sample["cactus"]["valid_label"] is False
    assert sample["normalized_gloss_matches_python"] is False
    assert sample["valid_label_matches_python"] is False
    assert sample["correct_matches_expected"] is False
    assert result.payload["summary"]["all_matches"] is False


def test_max_samples_starts_with_smoke_then_expands_to_demo_rows(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(
        reference,
        [
            ("demo_001", "demo", "thanks", "THANKS"),
            ("smoke_001", "smoke", "hello", "HELLO"),
            ("demo_002", "demo", "yes", "YES"),
        ],
    )
    _write_records(records, [("smoke_001", "hello"), ("demo_001", "thanks"), ("demo_002", "yes")])
    manifest.write_text(json.dumps({"labels": ["hello", "thanks", "yes"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
            max_samples=2,
        ),
        runner=MappingCactusPromptRunner({"smoke_001": "hello", "demo_001": "thanks"}),
    )

    assert [sample["sample_id"] for sample in result.payload["samples"]] == ["smoke_001", "demo_001"]
    assert result.payload["summary"]["sample_count"] == 2
    assert result.payload["summary"]["all_matches"] is True


def test_cli_mock_run_prints_summary_and_writes_report(tmp_path: Path, capsys) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    exit_code = cactus_parity_main(
        [
            "--reference",
            str(reference),
            "--records",
            str(records),
            "--manifest",
            str(manifest),
            "--cactus-weights",
            "mock-cactus-weights",
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "1",
            "--mock-cactus-output",
            "hello",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    artifact = json.loads((out_dir / "parity_report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary == {
        "scope": "cactus_prompt_control_parity",
        "runtime_mode": "mock",
        "report_path": str(out_dir / "parity_report.json"),
        "sample_count": 1,
        "match_count": 1,
        "runtime_error_count": 0,
        "real_cactus_parity_proven": False,
    }
    assert artifact["summary"]["all_matches"] is True


def test_real_cactus_runner_missing_weights_reports_export_failure_without_parity_claim(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    missing_weights = tmp_path / "missing-cactus-weights"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path=missing_weights,
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=RealCactusEnginePromptRunner(missing_weights),
    )

    sample = result.payload["samples"][0]
    assert result.payload["runtime_mode"] == "cactus_engine"
    assert result.payload["real_cactus_parity_proven"] is False
    assert result.payload["summary"]["runtime_error_count"] == 1
    assert sample["runtime_error"] == f"export: Cactus weights directory not found: {missing_weights}"
    assert sample["cactus"]["raw_model_output"] == ""


def test_cactus_cloud_handoff_does_not_claim_local_parity(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path=tmp_path / "converted-cactus-weights",
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=CloudHandoffCactusPromptRunner(),
    )

    sample = result.payload["samples"][0]
    assert result.payload["runtime_mode"] == "cactus_engine"
    assert result.payload["real_cactus_parity_proven"] is False
    assert result.payload["summary"]["all_matches"] is False
    assert result.payload["summary"]["runtime_error_count"] == 1
    assert sample["cactus"]["normalized_gloss"] == "hello"
    assert sample["cactus_response_metadata"]["cloud_handoff"] is True
    assert sample["runtime_error"] == (
        "runtime: Cactus completion used cloud_handoff=true; local Cactus parity is not proven."
    )


def test_missing_q64_record_is_reported_as_prompt_failure(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("other_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    result = run_cactus_prompt_control_parity(
        CactusPromptControlParityConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
            max_samples=1,
        ),
        runner=MockCactusPromptRunner("hello"),
    )

    sample = result.payload["samples"][0]
    assert result.payload["summary"]["runtime_error_count"] == 1
    assert sample["runtime_error"] == "prompt: q64 record not found for reference sample_id smoke_001"
    assert sample["cactus"] == {
        "raw_model_output": "",
        "normalized_gloss": None,
        "valid_label": False,
    }


def test_duplicate_q64_record_ids_fail_clearly(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "parity"
    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello"), ("smoke_001", "thanks")])
    manifest.write_text(json.dumps({"labels": ["hello", "thanks"]}), encoding="utf-8")

    try:
        run_cactus_prompt_control_parity(
            CactusPromptControlParityConfig(
                reference_path=reference,
                records_path=records,
                manifest_path=manifest,
                cactus_weights_path="mock-cactus-weights",
                out_dir=out_dir,
                max_samples=1,
            ),
            runner=MockCactusPromptRunner("hello"),
        )
    except ValueError as exc:
        assert str(exc) == "duplicate q64 sample_id 'smoke_001' at records 1 and 2."
    else:
        raise AssertionError("expected duplicate sample_id failure")


class FailingCactusPromptRunner:
    runtime_mode = "mock"

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        raise RuntimeError("Cactus completion failed")


class MappingCactusPromptRunner:
    runtime_mode = "mock"

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        return CactusCompletionResult(raw_model_output=self.outputs[sample_id])


class CloudHandoffCactusPromptRunner:
    runtime_mode = "cactus_engine"

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        return CactusCompletionResult(
            raw_model_output="hello",
            response_metadata={"cloud_handoff": True, "total_tokens": 1},
        )


def _write_reference(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint_path": "checkpoints/reference",
                "records": [
                    {
                        "sample_id": sample_id,
                        "selection_role": role,
                        "expected_gloss": expected,
                        "raw_model_output": raw,
                        "normalized_gloss": expected,
                        "valid_label": True,
                    }
                    for sample_id, role, expected, raw in rows
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_records(path: Path, samples: list[tuple[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    for sample_id, gloss in samples:
        rows.append(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": f"sample_id={sample_id}\nencoding=q64_full\npose_q64=abc",
                "output": gloss,
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
