"""Behavior tests for Phase 2A validation decision reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.phase2a import (
    Phase2AConfig,
    build_phase2a_report,
    write_phase2a_artifacts,
)
from scripts.phase2a_run import run_label_prior_phase2a


def test_phase2a_report_identifies_weak_classes_and_blocks_signer_leakage() -> None:
    train_df = pd.DataFrame(
        {
            "sample_id": ["train-1", "train-2"],
            "gloss": ["hello", "thanks"],
            "signer_id": ["s1", "s2"],
        }
    )
    val_df = pd.DataFrame(
        {
            "sample_id": ["val-1"],
            "gloss": ["hello"],
            "signer_id": ["s3"],
        }
    )
    test_df = pd.DataFrame(
        {
            "sample_id": ["test-1", "test-2", "test-3", "test-4"],
            "gloss": ["hello", "hello", "thanks", "thanks"],
            "signer_id": ["s2", "s4", "s5", "s6"],
        }
    )

    report = build_phase2a_report(
        truth=["hello", "hello", "thanks", "thanks"],
        predictions=["hello", "hello", "hello", "hello"],
        train_split=train_df,
        val_split=val_df,
        test_split=test_df,
        training_history={"train_loss": [1.8, 1.1], "val_loss": [1.9, 1.3]},
        config=Phase2AConfig(min_macro_f1_for_phase2b=0.75, weak_class_f1_threshold=0.65),
    )

    assert report.metric_summary.macro_f1 == 0.333
    assert report.metric_summary.accuracy == 0.5
    assert report.per_class["hello"].f1 == 0.667
    assert report.per_class["thanks"].f1 == 0.0
    assert report.weak_classes == ("thanks",)
    assert report.split_summary.has_signer_leakage is True
    assert report.decision == "adjust_strategy"
    assert report.loss_curve.train_loss == (1.8, 1.1)
    assert report.loss_curve.val_loss == (1.9, 1.3)
    assert any("signer leakage" in blocker for blocker in report.blockers)


def test_phase2a_artifacts_write_json_markdown_and_loss_curve_csv(tmp_path: Path) -> None:
    split = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "gloss": ["hello", "thanks"],
            "signer_id": ["s1", "s2"],
        }
    )
    report = build_phase2a_report(
        truth=["hello", "thanks"],
        predictions=["hello", "thanks"],
        train_split=split,
        val_split=split.iloc[0:0],
        test_split=split.iloc[0:0],
        training_history={"train_loss": [0.8], "val_loss": [0.9]},
    )

    outputs = write_phase2a_artifacts(report, tmp_path)

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    assert outputs["loss_curve_csv"].exists()

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert payload["decision"] == "proceed_to_phase2b"
    assert "Proceed to Phase 2B" in markdown
    assert outputs["loss_curve_csv"].read_text(encoding="utf-8").splitlines()[0] == "epoch,train_loss,val_loss"


def test_phase2a_label_prior_runner_writes_deterministic_blocking_artifacts(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    output_dir = tmp_path / "phase2a"

    pd.DataFrame(
        {
            "sample_id": ["train-1", "train-2", "train-3"],
            "gloss": ["drink", "drink", "hello"],
            "signer_id": ["s1", "s2", "s3"],
        }
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        {
            "sample_id": ["val-1", "val-2"],
            "gloss": ["drink", "hello"],
            "signer_id": ["s4", "s5"],
        }
    ).to_csv(val_csv, index=False)
    pd.DataFrame(
        {
            "sample_id": ["test-1", "test-2"],
            "gloss": ["drink", "hello"],
            "signer_id": ["s6", "s7"],
        }
    ).to_csv(test_csv, index=False)

    report = run_label_prior_phase2a(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=output_dir,
    )

    payload = json.loads((output_dir / "phase2a_report.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "phase2a_decision.md").read_text(encoding="utf-8")
    loss_curve = (output_dir / "loss_curve.csv").read_text(encoding="utf-8").splitlines()

    assert report.decision == "adjust_strategy"
    assert report.metric_summary.accuracy == 0.5
    assert report.per_class["drink"].f1 == 0.667
    assert report.per_class["hello"].f1 == 0.0
    assert payload["metadata"]["backend"] == "label_prior"
    assert "Backend: label_prior" in markdown
    assert loss_curve[0] == "epoch,train_loss,val_loss"
    assert len(loss_curve) == 2
