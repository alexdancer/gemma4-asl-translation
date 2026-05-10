"""Tests for multi-word ASL evaluation harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.multiword_asl import (
    evaluate_multiword_rows,
    load_multiword_predictions_jsonl,
    normalize_word,
    write_multiword_evaluation_artifacts,
)
from scripts.evaluation.evaluate_multiword_asl import main as multiword_cli_main


def test_normalize_word_basic() -> None:
    assert normalize_word("  HELLO! ") == "hello"
    assert normalize_word("Thank-you") == "thankyou"


def test_evaluate_multiword_rows_builds_dataset_level_metrics_and_artifacts(tmp_path: Path) -> None:
    records = [
        {
            "sample_id": "mw_1",
            "expected_words": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
                {"word": "world", "start_ms": 100, "end_ms": 250},
            ],
            "predicted_words": [
                {"word": "hello", "start_ms": 10, "end_ms": 90},
                {"word": "word", "start_ms": 100, "end_ms": 250},
            ],
        },
        {
            "sample_id": "mw_2",
            "expected_words": [
                {"word": "thank", "start_ms": 0, "end_ms": 60},
                {"word": "you", "start_ms": 60, "end_ms": 130},
            ],
            "predicted_words": [
                {"word": "thank", "start_ms": 5, "end_ms": 55},
                {"word": "you", "start_ms": 70, "end_ms": 150},
                {"word": "now", "start_ms": 170, "end_ms": 210},
            ],
        },
    ]

    rows, metrics = evaluate_multiword_rows(records)
    artifacts = write_multiword_evaluation_artifacts(rows, metrics, tmp_path)

    assert metrics["sample_count"] == 2
    assert metrics["total_expected_words"] == 4
    # dataset-level S/I/D: sample1(1S), sample2(1I) => WER=(1+1)/4=0.5
    assert metrics["word_error_rate"] == 0.5
    assert metrics["substitutions"] == 1
    assert metrics["insertions"] == 1
    assert metrics["deletions"] == 0
    assert metrics["exact_sequence_accuracy"] == 0.0

    # timestamp boundary MAE is weighted by matched boundaries across dataset.
    # matches with timestamps:
    # mw_1 hello: |0-10|+|100-90| = 20
    # mw_2 thank: |0-5|+|60-55| = 10
    # mw_2 you:   |60-70|+|130-150| = 30
    # total abs error=60, boundary_count=6 => 10.0
    assert metrics["timestamp_boundary_count"] == 6
    assert metrics["timestamp_boundary_mae_ms"] == 10.0

    with artifacts.predictions_csv.open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    written_metrics = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))

    assert len(csv_rows) == 2
    assert csv_rows[0]["sample_id"] == "mw_1"
    assert written_metrics["word_error_rate"] == 0.5


def test_load_multiword_predictions_jsonl_requires_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": "x", "expected_words": ["hello"]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires expected_words and predicted_words"):
        load_multiword_predictions_jsonl(path)


def test_load_multiword_predictions_jsonl_parses_string_or_object_word_items(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "mw_3",
                        "expected_words": ["Hello", "World"],
                        "predicted_words": [
                            {"word": "hello", "start_ms": 0, "end_ms": 100},
                            {"word": "world", "start_ms": 120, "end_ms": 230},
                        ],
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_multiword_predictions_jsonl(path)

    assert rows[0]["expected_words"] == [{"word": "hello"}, {"word": "world"}]
    assert rows[0]["predicted_words"][0]["start_ms"] == 0


def test_evaluate_multiword_rows_requires_non_empty_expected_words() -> None:
    with pytest.raises(ValueError, match="expected_words must be a non-empty list"):
        evaluate_multiword_rows([
            {"sample_id": "mw_bad", "expected_words": [], "predicted_words": ["hello"]}
        ])


def test_load_multiword_predictions_jsonl_rejects_non_integer_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "bad_timestamp.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "mw_4",
                "expected_words": [{"word": "hello", "start_ms": "10", "end_ms": 20}],
                "predicted_words": [{"word": "hello", "start_ms": 10, "end_ms": 20}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start_ms must be an integer timestamp"):
        load_multiword_predictions_jsonl(path)


def test_multiword_cli_main_writes_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    out_dir = tmp_path / "out"
    input_path.write_text(
        json.dumps(
            {
                "sample_id": "mw_cli_1",
                "expected_words": ["hello", "world"],
                "predicted_words": ["hello", "word"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = multiword_cli_main(["--input-jsonl", str(input_path), "--out-dir", str(out_dir)])

    assert status == 0
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "metrics.json").exists()
