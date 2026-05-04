"""Tests for the q64 Unsloth ASL evaluator contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.unsloth_asl import (
    CandidateLabelScore,
    MockASLGlossPredictor,
    MockConstrainedGlossScorer,
    MockPromptControlledASLGlossPredictor,
    build_constrained_comparison,
    build_metrics,
    build_prompt_control_comparison,
    build_prompt_control_prompt,
    build_sample_identity_validation,
    evaluate_q64_records,
    evaluate_q64_records_constrained,
    infer_q64_record,
    issue_22_activation_status,
    load_free_generation_artifacts,
    load_manifest_labels,
    normalize_model_output,
    score_q64_record_constrained,
    write_constrained_evaluation_artifacts,
    write_evaluation_artifacts,
    write_prompt_control_evaluation_artifacts,
)
from scripts.evaluation.evaluate_unsloth_asl_constrained import main as constrained_cli_main
from scripts.evaluation.evaluate_unsloth_asl_prompt_control import main as prompt_control_cli_main


def test_normalize_model_output_accepts_only_manifest_labels() -> None:
    labels = ("hello", "thank-you")

    predicted, valid = normalize_model_output("HELLO\nextra text", labels)
    invalid_predicted, invalid = normalize_model_output("not in manifest", labels)

    assert predicted == "hello"
    assert valid is True
    assert invalid_predicted == "not in manifest"
    assert invalid is False


def test_infer_q64_record_returns_shared_contract() -> None:
    record = {
        "instruction": "Classify.",
        "input": "sample_id=abc_1\nencoding=q64_full\npose_q64=WWW",
        "output": "hello",
    }
    predictor = MockASLGlossPredictor(("hello",))

    result = infer_q64_record(record, predictor, ("hello",))

    assert result.predicted_gloss == "hello"
    assert result.raw_model_output == "hello"
    assert result.valid_label is True
    assert result.expected_gloss == "hello"
    assert result.mode == "mock"


def test_evaluate_q64_records_builds_metrics_and_artifacts(tmp_path: Path) -> None:
    records = [
        {
            "instruction": "Classify.",
            "input": "sample_id=abc_1\nencoding=q64_full\npose_q64=WWW",
            "output": "hello",
        },
        {
            "instruction": "Classify.",
            "input": "sample_id=abc_2\nencoding=q64_full\npose_q64=XXX",
            "output": "thanks",
        },
    ]
    labels = ("hello", "thanks")

    rows, metrics = evaluate_q64_records(records, MockASLGlossPredictor(labels), labels)
    artifacts = write_evaluation_artifacts(rows, metrics, tmp_path)

    assert metrics["sample_count"] == 2
    assert set(metrics["per_class_accuracy"]) == {"hello", "thanks"}
    assert artifacts.predictions_csv.exists()
    assert artifacts.metrics_json.exists()

    with artifacts.predictions_csv.open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    written_metrics = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))

    assert len(csv_rows) == 2
    assert "raw_model_output" in csv_rows[0]
    assert written_metrics["confusion_matrix_counts"] == metrics["confusion_matrix_counts"]


def test_build_metrics_counts_invalid_outputs() -> None:
    rows = [
        {
            "expected_gloss": "hello",
            "prediction_for_metrics": "__invalid__",
            "valid_label": False,
            "correct": False,
        }
    ]

    metrics = build_metrics(rows, ("hello",))

    assert metrics["invalid_output_rate"] == 1.0
    assert metrics["strict_normalized_top1_accuracy"] == 0.0
    assert metrics["confusion_matrix_counts"]["hello"]["__invalid__"] == 1


def test_constrained_scoring_ranks_every_manifest_label() -> None:
    record = {
        "instruction": "Classify.",
        "input": "sample_id=abc_3\nencoding=q64_full\npose_q64=YYY",
        "output": "thanks",
    }
    labels = ("hello", "thanks", "yes")
    scorer = MockConstrainedGlossScorer({"hello": -3.0, "thanks": -0.2, "yes": -1.0})

    result = score_q64_record_constrained(record, scorer, labels)

    assert result.best_label == "thanks"
    assert result.expected_gloss == "thanks"
    assert result.correct is True
    assert result.mode == "mock"
    assert [score.label for score in result.ranked_scores] == ["thanks", "yes", "hello"]
    assert [score.score for score in result.ranked_scores] == [-0.2, -1.0, -3.0]
    assert set(score.label for score in result.ranked_scores) == set(labels)


def test_constrained_scoring_top1_is_deterministic_for_ties() -> None:
    record = {
        "instruction": "Classify.",
        "input": "sample_id=abc_4\nencoding=q64_full\npose_q64=ZZZ",
        "output": "yes",
    }
    labels = ("hello", "thanks", "yes")
    scorer = MockConstrainedGlossScorer({"hello": 1.0, "thanks": 1.0, "yes": 0.5})

    result = score_q64_record_constrained(record, scorer, labels)

    assert result.best_label == "hello"
    assert result.correct is False
    assert [score.label for score in result.ranked_scores] == ["hello", "thanks", "yes"]


def test_constrained_scoring_requires_scores_for_all_candidates() -> None:
    class IncompleteScorer:
        mode = "mock"

        def score_candidate_labels(self, record, labels):
            del record, labels
            return (CandidateLabelScore(label="hello", score=1.0),)

    record = {
        "instruction": "Classify.",
        "input": "sample_id=abc_5\nencoding=q64_full\npose_q64=AAA",
        "output": "hello",
    }

    try:
        score_q64_record_constrained(record, IncompleteScorer(), ("hello", "thanks"))
    except ValueError as exc:
        assert "Missing constrained scores" in str(exc)
    else:
        raise AssertionError("Expected missing constrained scores to fail.")


def test_constrained_scoring_is_separate_from_free_generation_metrics() -> None:
    record = {
        "instruction": "Classify.",
        "input": "sample_id=abc_6\nencoding=q64_full\npose_q64=BBB",
        "output": "thanks",
    }
    labels = ("hello", "thanks")

    rows, metrics = evaluate_q64_records([record], MockASLGlossPredictor(("hello",)), labels)
    constrained = score_q64_record_constrained(
        record,
        MockConstrainedGlossScorer({"hello": -5.0, "thanks": -0.1}),
        labels,
    )

    assert rows[0]["predicted_gloss"] == "hello"
    assert metrics["strict_normalized_top1_accuracy"] == 0.0
    assert constrained.best_label == "thanks"
    assert constrained.correct is True
    assert "constrained" not in metrics


def test_constrained_evaluation_writes_required_artifacts(tmp_path: Path) -> None:
    records = [
        {
            "instruction": "Classify.",
            "input": "sample_id=abc_7\nencoding=q64_full\npose_q64=CCC",
            "output": "thanks",
        }
    ]
    labels = ("hello", "thanks", "yes")
    scorer = MockConstrainedGlossScorer({"hello": -3.0, "thanks": -0.1, "yes": -2.0})

    rows, metrics = evaluate_q64_records_constrained(
        records,
        scorer,
        labels,
        free_generation_predictions={"abc_7": "hello"},
        top_score_count=2,
    )
    comparison = build_constrained_comparison(
        metrics,
        {
            "sample_count": 1,
            "strict_normalized_top1_accuracy": 0.0,
            "invalid_output_rate": 0.25,
            "correct": 0,
        },
    )
    artifacts = write_constrained_evaluation_artifacts(rows, metrics, comparison, tmp_path)

    with artifacts.constrained_predictions_csv.open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    written_metrics = json.loads(artifacts.constrained_metrics_json.read_text(encoding="utf-8"))
    written_comparison = json.loads(artifacts.comparison_json.read_text(encoding="utf-8"))
    top_scores = json.loads(csv_rows[0]["top_scores"])

    assert artifacts.constrained_predictions_csv.name == "constrained_predictions.csv"
    assert artifacts.constrained_metrics_json.name == "constrained_metrics.json"
    assert artifacts.comparison_json.name == "comparison.json"
    assert csv_rows[0]["sample_id"] == "abc_7"
    assert csv_rows[0]["expected_gloss"] == "thanks"
    assert csv_rows[0]["free_generation_prediction"] == "hello"
    assert csv_rows[0]["constrained_prediction"] == "thanks"
    assert csv_rows[0]["constrained_correct"] == "True"
    assert top_scores == [{"label": "thanks", "score": -0.1}, {"label": "yes", "score": -2.0}]
    assert written_metrics["sample_count"] == 1
    assert written_metrics["constrained_top1_accuracy"] == 1.0
    assert written_metrics["top_score_count"] == 2
    assert written_comparison["deltas"]["top1_accuracy"] == 1.0
    assert written_comparison["deltas"]["invalid_output_rate"] == -0.25


def test_load_free_generation_artifacts_requires_metrics_and_predictions(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")

    try:
        load_free_generation_artifacts(tmp_path)
    except FileNotFoundError as exc:
        assert "Free-generation predictions not found" in str(exc)
    else:
        raise AssertionError("Expected missing free-generation predictions to fail.")


def test_load_free_generation_artifacts_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "sample_count": 2,
                "strict_normalized_top1_accuracy": 0.0,
                "invalid_output_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "predictions.csv").write_text(
        "index,sample_id,expected_gloss,predicted_gloss,raw_model_output,valid_label,correct,mode\n"
        "0,dup_1,hello,hello,hello,True,True,real\n"
        "1,dup_1,thanks,thanks,thanks,True,True,real\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_id.*dup_1"):
        load_free_generation_artifacts(tmp_path)


def test_load_manifest_labels_rejects_normalized_label_collisions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"labels": ["Thank You!", "thank you"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized-label collisions.*Thank You!.*thank you"):
        load_manifest_labels(manifest)


def test_constrained_cli_mock_smoke_and_missing_baseline_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    test_file = tmp_path / "test.jsonl"
    baseline_dir = tmp_path / "baseline"
    out_dir = tmp_path / "constrained"
    baseline_dir.mkdir()
    manifest.write_text(json.dumps({"labels": ["hello", "thanks"]}), encoding="utf-8")
    test_file.write_text(
        json.dumps(
            {
                "instruction": "Classify.",
                "input": "sample_id=abc_8\nencoding=q64_full\npose_q64=DDD",
                "output": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (baseline_dir / "metrics.json").write_text(
        json.dumps(
            {
                "sample_count": 1,
                "strict_normalized_top1_accuracy": 1.0,
                "invalid_output_rate": 0.0,
                "correct": 1,
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "predictions.csv").write_text(
        "index,sample_id,expected_gloss,predicted_gloss,raw_model_output,valid_label,correct,mode\n"
        "0,abc_8,hello,hello,hello,True,True,real\n",
        encoding="utf-8",
    )

    status = constrained_cli_main(
        [
            "--mock",
            "--manifest",
            str(manifest),
            "--test-file",
            str(test_file),
            "--free-generation-results-dir",
            str(baseline_dir),
            "--out-dir",
            str(out_dir),
            "--top-scores",
            "1",
        ]
    )
    missing_status = constrained_cli_main(
        [
            "--mock",
            "--manifest",
            str(manifest),
            "--test-file",
            str(test_file),
            "--free-generation-results-dir",
            str(tmp_path / "missing-baseline"),
            "--out-dir",
            str(tmp_path / "missing-output"),
        ]
    )

    assert status == 0
    assert (out_dir / "constrained_predictions.csv").exists()
    assert (out_dir / "constrained_metrics.json").exists()
    assert (out_dir / "comparison.json").exists()
    assert missing_status == 2


def test_prompt_control_prompt_includes_canonical_label_contract() -> None:
    record = {
        "instruction": "Old instruction should be replaced.",
        "input": "sample_id=abc_9\nencoding=q64_full\npose_q64=EEE",
        "output": "hello",
    }

    prompt = build_prompt_control_prompt(record, ("hello", "thank you"))

    assert "choose exactly one label" in prompt
    assert "Reply with only the chosen label" in prompt
    assert "- hello" in prompt
    assert "- thank you" in prompt
    assert "sample_id=abc_9" in prompt


def test_prompt_control_mock_predictor_records_strict_prompt() -> None:
    record = {
        "instruction": "Classify.",
        "input": "sample_id=abc_10\nencoding=q64_full\npose_q64=FFF",
        "output": "hello",
    }
    predictor = MockPromptControlledASLGlossPredictor(("hello", "thanks"))

    rows, metrics = evaluate_q64_records([record], predictor, ("hello", "thanks"))

    assert len(predictor.rendered_prompts) == 1
    assert "Canonical Top-50 labels" in predictor.rendered_prompts[0]
    assert "- thanks" in predictor.rendered_prompts[0]
    assert rows[0]["mode"] == "mock"
    assert metrics["sample_count"] == 1


def test_prompt_control_comparison_and_report_artifacts(tmp_path: Path) -> None:
    rows = [
        {
            "index": 0,
            "sample_id": "abc_11",
            "expected_gloss": "hello",
            "predicted_gloss": "hello",
            "raw_model_output": "hello",
            "valid_label": True,
            "correct": True,
            "mode": "mock",
            "prediction_for_metrics": "hello",
        },
        {
            "index": 1,
            "sample_id": "abc_12",
            "expected_gloss": "thanks",
            "predicted_gloss": "hello",
            "raw_model_output": "hello",
            "valid_label": True,
            "correct": False,
            "mode": "mock",
            "prediction_for_metrics": "hello",
        },
    ]
    prompt_metrics = build_metrics(rows, ("hello", "thanks"))
    baseline_metrics = {
        "sample_count": 2,
        "strict_normalized_top1_accuracy": 0.0,
        "invalid_output_rate": 0.5,
        "correct": 0,
        "invalid": 1,
    }
    constrained_comparison = {
        "constrained": {
            "constrained_top1_accuracy": 1.0,
            "sample_count": 2,
            "correct": 2,
        }
    }

    comparison = build_prompt_control_comparison(
        prompt_metrics,
        baseline_metrics,
        constrained_comparison,
    )
    artifacts = write_prompt_control_evaluation_artifacts(
        rows,
        prompt_metrics,
        comparison,
        tmp_path,
    )

    written_comparison = json.loads(artifacts.comparison_json.read_text(encoding="utf-8"))
    report = artifacts.report_md.read_text(encoding="utf-8")

    assert artifacts.predictions_csv.name == "predictions.csv"
    assert artifacts.metrics_json.name == "metrics.json"
    assert artifacts.comparison_json.name == "comparison.json"
    assert artifacts.report_md.name == "report.md"
    assert written_comparison["activation"]["active"] is True
    assert written_comparison["deltas"]["prompt_control_vs_baseline_accuracy"] == 0.5
    assert (
        written_comparison["deltas"]["prompt_control_vs_baseline_invalid_output_rate"]
        == -0.5
    )
    assert (
        written_comparison["recommendation"]["prompt_output_control_enough_before_retraining"]
        is True
    )
    assert "Enough before retraining: True" in report


def test_prompt_control_comparison_requires_full_sample_count_for_recommendation(
    tmp_path: Path,
) -> None:
    prompt_metrics = {
        "sample_count": 1,
        "strict_normalized_top1_accuracy": 1.0,
        "invalid_output_rate": 0.0,
        "correct": 1,
        "invalid": 0,
    }
    baseline_metrics = {
        "sample_count": 50,
        "strict_normalized_top1_accuracy": 0.4,
        "invalid_output_rate": 0.6,
        "correct": 20,
        "invalid": 30,
    }

    comparison = build_prompt_control_comparison(
        prompt_metrics,
        baseline_metrics,
        {"constrained": {"constrained_top1_accuracy": 0.7}},
    )
    artifacts = write_prompt_control_evaluation_artifacts(
        [
            {
                "index": 0,
                "sample_id": "abc_14",
                "expected_gloss": "hello",
                "predicted_gloss": "hello",
                "raw_model_output": "hello",
                "valid_label": True,
                "correct": True,
                "mode": "mock",
            }
        ],
        prompt_metrics,
        comparison,
        tmp_path,
    )
    report = artifacts.report_md.read_text(encoding="utf-8")

    assert comparison["comparison_scope"]["sample_count_matches_baseline"] is False
    assert (
        comparison["recommendation"]["prompt_output_control_enough_before_retraining"]
        is False
    )
    assert "partial/smoke comparison" in report


def test_prompt_control_comparison_missing_sample_count_is_no_go() -> None:
    prompt_metrics = {
        "strict_normalized_top1_accuracy": 1.0,
        "invalid_output_rate": 0.0,
        "correct": 1,
        "invalid": 0,
    }
    baseline_metrics = {
        "sample_count": 1,
        "strict_normalized_top1_accuracy": 0.0,
        "invalid_output_rate": 1.0,
        "correct": 0,
        "invalid": 1,
    }

    comparison = build_prompt_control_comparison(prompt_metrics, baseline_metrics)

    assert comparison["comparison_scope"]["sample_count_matches_baseline"] is False
    assert (
        comparison["recommendation"]["prompt_output_control_enough_before_retraining"]
        is False
    )


def test_prompt_control_comparison_mismatched_sample_ids_are_no_go(
    tmp_path: Path,
) -> None:
    baseline_rows = [
        {"sample_id": "abc_15", "expected_gloss": "hello"},
        {"sample_id": "abc_16", "expected_gloss": "thanks"},
    ]
    prompt_rows = [
        {"sample_id": "abc_15", "expected_gloss": "hello"},
        {"sample_id": "abc_17", "expected_gloss": "thanks"},
    ]
    sample_identity = build_sample_identity_validation(baseline_rows, prompt_rows)
    comparison = build_prompt_control_comparison(
        {
            "sample_count": 2,
            "strict_normalized_top1_accuracy": 1.0,
            "invalid_output_rate": 0.0,
            "correct": 2,
            "invalid": 0,
        },
        {
            "sample_count": 2,
            "strict_normalized_top1_accuracy": 0.0,
            "invalid_output_rate": 1.0,
            "correct": 0,
            "invalid": 2,
        },
        sample_identity_validation=sample_identity,
    )
    artifacts = write_prompt_control_evaluation_artifacts(
        [
            {
                "index": 0,
                "sample_id": "abc_15",
                "expected_gloss": "hello",
                "predicted_gloss": "hello",
                "raw_model_output": "hello",
                "valid_label": True,
                "correct": True,
                "mode": "mock",
            },
            {
                "index": 1,
                "sample_id": "abc_17",
                "expected_gloss": "thanks",
                "predicted_gloss": "thanks",
                "raw_model_output": "thanks",
                "valid_label": True,
                "correct": True,
                "mode": "mock",
            },
        ],
        {
            "sample_count": 2,
            "strict_normalized_top1_accuracy": 1.0,
            "invalid_output_rate": 0.0,
            "correct": 2,
            "invalid": 0,
        },
        comparison,
        tmp_path,
    )
    report = artifacts.report_md.read_text(encoding="utf-8")

    assert sample_identity["matches_baseline"] is False
    assert sample_identity["sample_id_order_matches_baseline"] is False
    assert comparison["comparison_scope"]["sample_count_matches_baseline"] is True
    assert comparison["comparison_scope"]["sample_identity_matches_baseline"] is False
    assert (
        comparison["recommendation"]["prompt_output_control_enough_before_retraining"]
        is False
    )
    assert "prediction identities do not align" in report


def test_sample_identity_validation_rejects_duplicate_prompt_sample_ids() -> None:
    baseline_rows = [{"sample_id": "abc_18", "expected_gloss": "hello"}]
    prompt_rows = [
        {"sample_id": "abc_18", "expected_gloss": "hello"},
        {"sample_id": "abc_18", "expected_gloss": "hello"},
    ]

    with pytest.raises(ValueError, match="duplicate sample_id.*abc_18"):
        build_sample_identity_validation(baseline_rows, prompt_rows)


def test_issue_22_activation_requires_invalid_outputs_and_constrained_gain() -> None:
    active = issue_22_activation_status(
        {"strict_normalized_top1_accuracy": 0.4, "invalid_output_rate": 0.6},
        {"constrained": {"constrained_top1_accuracy": 0.7}},
    )
    inactive = issue_22_activation_status(
        {"strict_normalized_top1_accuracy": 0.4, "invalid_output_rate": 0.0},
        {"constrained": {"constrained_top1_accuracy": 0.7}},
    )

    assert active["active"] is True
    assert inactive["active"] is False


def test_prompt_control_cli_mock_smoke_and_overwrite_safety(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    test_file = tmp_path / "test.jsonl"
    baseline_dir = tmp_path / "baseline"
    constrained_dir = tmp_path / "constrained"
    out_dir = tmp_path / "prompt-control"
    baseline_dir.mkdir()
    constrained_dir.mkdir()
    manifest.write_text(json.dumps({"labels": ["hello", "thanks"]}), encoding="utf-8")
    test_file.write_text(
        json.dumps(
            {
                "instruction": "Classify.",
                "input": "sample_id=abc_13\nencoding=q64_full\npose_q64=GGG",
                "output": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (baseline_dir / "metrics.json").write_text(
        json.dumps(
            {
                "sample_count": 1,
                "strict_normalized_top1_accuracy": 0.0,
                "invalid_output_rate": 1.0,
                "correct": 0,
                "invalid": 1,
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "predictions.csv").write_text(
        "index,sample_id,expected_gloss,predicted_gloss,raw_model_output,valid_label,correct,mode\n"
        "0,abc_13,hello,thank,thank,False,False,real\n",
        encoding="utf-8",
    )
    (constrained_dir / "comparison.json").write_text(
        json.dumps(
            {
                "constrained": {
                    "constrained_top1_accuracy": 1.0,
                    "sample_count": 1,
                    "correct": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    status = prompt_control_cli_main(
        [
            "--mock",
            "--manifest",
            str(manifest),
            "--test-file",
            str(test_file),
            "--free-generation-results-dir",
            str(baseline_dir),
            "--constrained-results-dir",
            str(constrained_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    overwrite_status = prompt_control_cli_main(
        [
            "--mock",
            "--manifest",
            str(manifest),
            "--test-file",
            str(test_file),
            "--free-generation-results-dir",
            str(baseline_dir),
            "--constrained-results-dir",
            str(constrained_dir),
            "--out-dir",
            str(baseline_dir),
        ]
    )

    assert status == 0
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "report.md").exists()
    assert overwrite_status == 2
