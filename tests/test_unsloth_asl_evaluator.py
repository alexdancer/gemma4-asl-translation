"""Tests for the q64 Unsloth ASL evaluator contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.unsloth_asl import (
    CandidateLabelScore,
    MockASLGlossPredictor,
    MockConstrainedGlossScorer,
    build_constrained_comparison,
    build_metrics,
    evaluate_q64_records_constrained,
    evaluate_q64_records,
    infer_q64_record,
    load_free_generation_artifacts,
    normalize_model_output,
    score_q64_record_constrained,
    write_constrained_evaluation_artifacts,
    write_evaluation_artifacts,
)
from scripts.evaluate_unsloth_asl_constrained import main as constrained_cli_main


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
