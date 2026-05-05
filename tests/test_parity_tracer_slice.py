"""Behavior tests for issue #34 parity tracer slice artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.mobile.run_parity_tracer_slice import main as parity_tracer_main
from src.mobile.parity_tracer_slice import (
    ParityCompletionResult,
    ParityTracerSliceConfig,
    run_parity_tracer_slice,
)


class SpyRunner:
    def __init__(self, *, runtime_mode: str, raw_output: str, metadata: dict[str, Any] | None = None) -> None:
        self.runtime_mode = runtime_mode
        self.raw_output = raw_output
        self.metadata = metadata or {}
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, sample_id: str) -> ParityCompletionResult:
        self.calls.append((sample_id, prompt))
        return ParityCompletionResult(raw_model_output=self.raw_output, runtime_metadata=self.metadata)


class FailingRunner:
    runtime_mode = "mock"

    def complete(self, prompt: str, *, sample_id: str) -> ParityCompletionResult:
        raise RuntimeError("Cactus completion failed")


def test_parity_tracer_slice_runs_smoke_sample_in_python_and_cactus_paths(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "artifacts"

    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello", "thanks"]}), encoding="utf-8")

    python_runner = SpyRunner(runtime_mode="python", raw_output="HELLO", metadata={"latency_ms": 11.2})
    cactus_runner = SpyRunner(runtime_mode="cactus_engine", raw_output="hello", metadata={"latency_ms": 4.8})

    result = run_parity_tracer_slice(
        ParityTracerSliceConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
        ),
        python_runner=python_runner,
        cactus_runner=cactus_runner,
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    assert result.report_path == out_dir / "parity_report_v1.json"
    assert payload["scope"] == "parity_tracer_slice"
    assert payload["summary"]["sample_count"] == 1
    assert payload["summary"]["match_count"] == 1
    assert payload["summary"]["runtime_error_count"] == 0

    assert sample["python"]["normalized_gloss"] == "hello"
    assert sample["python"]["valid_label"] is True
    assert sample["cactus"]["normalized_gloss"] == "hello"
    assert sample["cactus"]["valid_label"] is True
    assert sample["normalized_gloss_match"] is True
    assert sample["valid_label_match"] is True
    assert sample["runtime_error"] is None
    assert sample["runtime"]["python"]["runtime_mode"] == "python"
    assert sample["runtime"]["cactus"]["runtime_mode"] == "cactus_engine"

    assert [call[0] for call in python_runner.calls] == ["smoke_001"]
    assert [call[0] for call in cactus_runner.calls] == ["smoke_001"]


def test_parity_tracer_slice_cli_writes_report_and_summary(tmp_path: Path, capsys) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "artifacts"

    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    exit_code = parity_tracer_main(
        [
            "--reference",
            str(reference),
            "--records",
            str(records),
            "--manifest",
            str(manifest),
            "--cactus-weights",
            "mock-cactus-weights",
            "--mock-cactus-output",
            "hello",
            "--out-dir",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    artifact = json.loads((out_dir / "parity_report_v1.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary == {
        "cactus_runtime_mode": "mock",
        "match_count": 1,
        "python_runtime_mode": "python_reference",
        "report_path": str(out_dir / "parity_report_v1.json"),
        "runtime_error_count": 0,
        "sample_count": 1,
        "scope": "parity_tracer_slice",
    }
    assert artifact["summary"]["all_matches"] is True


def test_parity_tracer_slice_records_runtime_error_in_artifact(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "artifacts"

    _write_reference(reference, [("smoke_001", "smoke", "hello", "HELLO")])
    _write_records(records, [("smoke_001", "hello")])
    manifest.write_text(json.dumps({"labels": ["hello"]}), encoding="utf-8")

    python_runner = SpyRunner(runtime_mode="python", raw_output="HELLO")

    result = run_parity_tracer_slice(
        ParityTracerSliceConfig(
            reference_path=reference,
            records_path=records,
            manifest_path=manifest,
            cactus_weights_path="mock-cactus-weights",
            out_dir=out_dir,
        ),
        python_runner=python_runner,
        cactus_runner=FailingRunner(),
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    assert payload["summary"]["runtime_error_count"] == 1
    assert payload["summary"]["all_matches"] is False
    assert sample["runtime_error"] == "runtime: Cactus completion failed"
    assert sample["cactus"] == {
        "raw_model_output": "",
        "normalized_gloss": None,
        "valid_label": False,
    }


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
