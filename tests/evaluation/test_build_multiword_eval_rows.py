from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.build_multiword_eval_rows import build_multiword_eval_rows, main


def test_build_multiword_eval_rows_merges_expected_and_predicted_aliases() -> None:
    expected = [
        {"sample_id": "a", "expected_words": ["hello", "world"]},
        {"sample_id": "b", "expected_words": ["thank", "you"]},
    ]
    predicted = [
        {"sample_id": "a", "transcript_words": ["hello", "word"]},
        {"sample_id": "b", "predicted_words": ["thank", "you"]},
    ]

    rows = build_multiword_eval_rows(expected, predicted, allow_missing_predictions=False)

    assert rows == [
        {"sample_id": "a", "expected_words": ["hello", "world"], "predicted_words": ["hello", "word"]},
        {"sample_id": "b", "expected_words": ["thank", "you"], "predicted_words": ["thank", "you"]},
    ]


def test_build_multiword_eval_rows_rejects_missing_prediction_by_default() -> None:
    with pytest.raises(ValueError, match="missing prediction"):
        build_multiword_eval_rows(
            [{"sample_id": "a", "expected_words": ["hello"]}],
            [],
            allow_missing_predictions=False,
        )


def test_build_multiword_eval_rows_allows_missing_prediction_with_sentinel() -> None:
    rows = build_multiword_eval_rows(
        [{"sample_id": "a", "expected_words": ["hello"]}],
        [],
        allow_missing_predictions=True,
    )
    assert rows[0]["predicted_words"] == ["__missing__"]


def test_cli_main_writes_merged_jsonl(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    out_path = tmp_path / "merged.jsonl"

    expected_path.write_text(
        json.dumps({"sample_id": "clip_1", "expected_words": ["hello", "world"]}) + "\n",
        encoding="utf-8",
    )
    predicted_path.write_text(
        json.dumps({"sample_id": "clip_1", "words": ["hello", "word"]}) + "\n",
        encoding="utf-8",
    )

    status = main([
        "--expected-jsonl",
        str(expected_path),
        "--predicted-jsonl",
        str(predicted_path),
        "--out-jsonl",
        str(out_path),
    ])

    assert status == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["sample_id"] == "clip_1"
    assert payload["predicted_words"] == ["hello", "word"]
