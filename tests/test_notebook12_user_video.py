from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.notebook12_user_video import (
    DEFAULT_NOTEBOOK12_CONFIG,
    DurationLimitError,
    append_result_jsonl,
    build_batch_summary,
    build_result_row,
    build_top50_prompt,
    deterministic_translation,
    even_sample_indices,
    list_video_files,
    load_batch_expected_labels,
    load_top50_labels,
    strict_top50_prediction,
    validate_clip_duration,
)


def test_strict_top50_prediction_accepts_only_canonical_first_line() -> None:
    labels = ["book", "thank you", "drink"]

    result = strict_top50_prediction(" Thank you.\nextra text", labels)

    assert result.status == "ok"
    assert result.predicted_gloss == "thank you"
    assert result.raw_output == " Thank you.\nextra text"


def test_build_top50_prompt_includes_the_allowed_labels() -> None:
    prompt = build_top50_prompt(["book", "thank you", "drink"])

    assert "Return exactly one gloss label" in prompt
    assert "Approved labels:" in prompt
    assert "book, thank you, drink" in prompt


def test_strict_top50_prediction_returns_uncertain_for_non_allowlisted_text() -> None:
    labels = ["book", "thank you", "drink"]

    result = strict_top50_prediction("hello", labels)

    assert result.status == "uncertain"
    assert result.predicted_gloss is None
    assert result.raw_output == "hello"


def test_validate_clip_duration_hard_blocks_over_five_seconds() -> None:
    validate_clip_duration(5.0, max_seconds=5.0)

    with pytest.raises(DurationLimitError) as excinfo:
        validate_clip_duration(5.01, max_seconds=5.0)

    assert excinfo.value.duration_seconds == 5.01
    assert excinfo.value.max_seconds == 5.0


def test_even_sample_indices_require_exact_target_count() -> None:
    assert even_sample_indices(source_count=10, target_count=5) == (0, 2, 4, 7, 9)

    with pytest.raises(ValueError, match="Cannot sample 30 frames"):
        even_sample_indices(source_count=29, target_count=30)


def test_build_result_row_uses_deterministic_translation_and_correctness() -> None:
    prediction = strict_top50_prediction("book", ["book", "drink"])

    row = build_result_row(
        video_filename="sample.mov",
        duration_seconds=3.2,
        expected_label="book",
        prediction=prediction,
        config=DEFAULT_NOTEBOOK12_CONFIG,
    )

    assert row["status"] == "ok"
    assert row["predicted_gloss"] == "book"
    assert row["translation"] == "Book."
    assert row["correct"] is True
    assert row["model"] == "AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora"
    assert row["base_model"] == "unsloth/gemma-4-26B-A4B-it"
    assert row["num_frames"] == 30
    assert row["frame_size"] == 448
    assert "timestamp" in row


def test_deterministic_translation_handles_multiword_glosses() -> None:
    assert deterministic_translation("thank you") == "Thank you."
    assert deterministic_translation(None) is None


def test_append_result_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_result_jsonl(path, {"video_filename": "a.mov", "status": "ok"})
    append_result_jsonl(path, {"video_filename": "b.mov", "status": "uncertain"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"video_filename": "a.mov", "status": "ok"},
        {"video_filename": "b.mov", "status": "uncertain"},
    ]


def test_load_top50_labels_requires_exact_unique_normalized_labels(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("\n".join(f"label_{i}" for i in range(50)), encoding="utf-8")

    assert load_top50_labels(labels_path) == tuple(f"label_{i}" for i in range(50))

    labels_path.write_text("\n".join(["book", "Book"] + [f"label_{i}" for i in range(48)]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_top50_labels(labels_path)

    labels_path.write_text("\n".join(f"label_{i}" for i in range(49)), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected exactly 50"):
        load_top50_labels(labels_path)


def test_load_batch_expected_labels_from_csv_normalizes_by_filename(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,expected_label\n"
        "videos/book_001.mp4,Book\n"
        "dance.mov, dance \n",
        encoding="utf-8",
    )

    labels = load_batch_expected_labels(csv_path)

    assert labels == {"book_001.mp4": "book", "dance.mov": "dance"}


def test_list_video_files_recurses_and_sorts_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "videos" / "nested").mkdir(parents=True)
    keep_b = tmp_path / "videos" / "b.mov"
    keep_a = tmp_path / "videos" / "nested" / "a.mp4"
    keep_c = tmp_path / "videos" / "c.m4v"
    skip = tmp_path / "videos" / "notes.txt"
    for path in [keep_b, keep_a, keep_c, skip]:
        path.write_text("x", encoding="utf-8")

    assert list_video_files(tmp_path / "videos") == (keep_a, keep_b, keep_c)


def test_build_batch_summary_counts_labeled_accuracy_uncertain_and_failures() -> None:
    rows = [
        {"status": "ok", "expected_label": "book", "predicted_gloss": "book", "correct": True},
        {"status": "ok", "expected_label": "dance", "predicted_gloss": "book", "correct": False},
        {"status": "uncertain", "expected_label": "drink", "predicted_gloss": None, "correct": False},
        {"status": "error", "expected_label": "hello", "predicted_gloss": None, "correct": None},
        {"status": "ok", "expected_label": None, "predicted_gloss": "book", "correct": None},
    ]

    summary = build_batch_summary(rows)

    assert summary == {
        "total": 5,
        "labeled": 4,
        "ok": 3,
        "correct": 1,
        "incorrect": 2,
        "uncertain": 1,
        "failed": 1,
        "accuracy_on_labeled": 0.25,
    }
