"""Tests for the q64 Unsloth ASL evaluator contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.unsloth_asl import (
    CandidateLabelScore,
    MockASLGlossPredictor,
    MockConstrainedGlossScorer,
    build_metrics,
    evaluate_q64_records,
    infer_q64_record,
    normalize_model_output,
    score_q64_record_constrained,
    write_evaluation_artifacts,
)


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
