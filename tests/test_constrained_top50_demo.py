"""Behavior tests for demo-safe constrained Top-50 inference."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.demo.run_constrained_top50_demo import main as run_constrained_top50_demo_main
from src.demo.constrained_top50 import ConstrainedTop50DemoConfig, run_constrained_top50_demo
from src.evaluation.unsloth_asl import MockConstrainedGlossScorer


def _write_demo_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    record_path = tmp_path / "known_good.jsonl"
    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    record_path.write_text(
        json.dumps(
            {
                "instruction": "Classify this compact ASL pose encoding.",
                "input": "sample_id=hearing_26986\nencoding=q64_full\npose_q64=abc",
                "output": "hearing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({"labels": ["drink", "hearing", "book"]}),
        encoding="utf-8",
    )
    return checkpoint_path, record_path, manifest_path


def test_constrained_top50_demo_returns_canonical_label_and_scoped_artifact(tmp_path: Path) -> None:
    checkpoint_path, record_path, manifest_path = _write_demo_inputs(tmp_path)
    out_dir = tmp_path / "demo_safe_constrained"
    scorer = MockConstrainedGlossScorer({"drink": 0.2, "hearing": 2.0, "book": 0.1})

    result = run_constrained_top50_demo(
        ConstrainedTop50DemoConfig(
            checkpoint_path=checkpoint_path,
            records_path=record_path,
            manifest_path=manifest_path,
            record_id="hearing_26986",
            out_dir=out_dir,
            top_k=2,
        ),
        scorer=scorer,
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.selected_label == "hearing"
    assert result.best_label == "hearing"
    assert result.expected_gloss == "hearing"
    assert result.correct is True
    assert result.output.display_text == "hearing"
    assert result.scope == "demo_safe_constrained_top50"
    assert result.claims == artifact["claims"]
    assert result.top_candidates[0].label == "hearing"
    assert result.top_candidates[0].score == 2.0
    assert artifact["scope"] == "demo_safe_constrained_top50"
    assert "not the primary free-generation proof metric" in artifact["claims"]
    assert artifact["selected_label"] == "hearing"
    assert artifact["top_candidates"] == [
        {"label": "hearing", "score": 2.0},
        {"label": "drink", "score": 0.2},
    ]
    assert artifact["confidence"] is None
    assert artifact["confidence_available"] is False
    assert artifact["constrained_metadata"]["constrained"] is True
    assert artifact["constrained_metadata"]["activation_evidence"]["issue_21"][
        "useful_constrained_signal"
    ] is True
    assert artifact["constrained_metadata"]["activation_evidence"]["issue_22"][
        "prompt_control_sufficient"
    ] is True


def test_constrained_top50_artifact_is_separate_from_free_generation_outputs(tmp_path: Path) -> None:
    checkpoint_path, record_path, manifest_path = _write_demo_inputs(tmp_path)
    out_dir = tmp_path / "demo_safe_constrained"

    result = run_constrained_top50_demo(
        ConstrainedTop50DemoConfig(
            checkpoint_path=checkpoint_path,
            records_path=record_path,
            manifest_path=manifest_path,
            record_id="hearing_26986",
            out_dir=out_dir,
        ),
        scorer=MockConstrainedGlossScorer({"drink": 0.2, "hearing": 2.0, "book": 0.1}),
    )

    assert result.artifact_path == out_dir / "demo_safe_constrained_top50_readiness.json"
    assert result.artifact_path.exists()
    assert not (out_dir / "predictions.csv").exists()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "comparison.json").exists()
    assert not (out_dir / "prerecorded_q64_demo_readiness.json").exists()


def test_constrained_top50_demo_cli_runs_with_mock_scores(tmp_path: Path, capsys) -> None:
    checkpoint_path, record_path, manifest_path = _write_demo_inputs(tmp_path)
    out_dir = tmp_path / "demo_artifacts"

    exit_code = run_constrained_top50_demo_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--records",
            str(record_path),
            "--manifest",
            str(manifest_path),
            "--record-id",
            "hearing_26986",
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
            "--mock",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    artifact = json.loads(
        (out_dir / "demo_safe_constrained_top50_readiness.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert summary["scope"] == "demo_safe_constrained_top50"
    assert summary["model_path"] == str(checkpoint_path)
    assert summary["input_record_id"] == "hearing_26986"
    assert summary["inference_mode"] == "mock"
    assert summary["selected_label"] == "hearing"
    assert summary["visible_gloss"] == "hearing"
    assert summary["correct"] is True
    assert summary["confidence_available"] is False
    assert summary["top_candidates"][0]["label"] == "hearing"
    assert summary["artifact_path"] == str(out_dir / "demo_safe_constrained_top50_readiness.json")
    assert artifact["constrained_metadata"]["metric_boundary"].startswith("Diagnostic/demo-safe")
