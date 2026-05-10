"""Multi-word ASL evaluation utilities.

Evaluates sequence quality (WER/exact match) plus optional timing quality
(timestamp boundary MAE) from expected/predicted multi-word transcripts.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MultiwordEvaluationArtifacts:
    predictions_csv: Path
    metrics_json: Path


def normalize_word(word: str) -> str:
    normalized = word.strip().lower()
    normalized = re.sub(r"[^a-z0-9']+", "", normalized)
    return normalized


def _parse_timestamp_ms(value: Any, *, field_name: str, index: int, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name}[{index}].{key} must be an integer timestamp in milliseconds.")
    return value


def _words_from_field(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")

    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if isinstance(item, str):
            word = normalize_word(item)
            if not word:
                raise ValueError(f"{field_name}[{idx}] has empty word.")
            parsed.append({"word": word})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{idx}] must be string or object.")
        raw_word = item.get("word")
        if not isinstance(raw_word, str) or not normalize_word(raw_word):
            raise ValueError(f"{field_name}[{idx}].word must be non-empty string.")
        entry = {"word": normalize_word(raw_word)}
        if item.get("start_ms") is not None:
            entry["start_ms"] = _parse_timestamp_ms(
                item["start_ms"],
                field_name=field_name,
                index=idx,
                key="start_ms",
            )
        if item.get("end_ms") is not None:
            entry["end_ms"] = _parse_timestamp_ms(
                item["end_ms"],
                field_name=field_name,
                index=idx,
                key="end_ms",
            )
        parsed.append(entry)
    return parsed


def load_multiword_predictions_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain JSON objects.")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{path}:{line_no} missing required sample_id.")
            if "expected_words" not in row or "predicted_words" not in row:
                raise ValueError(
                    f"{path}:{line_no} requires expected_words and predicted_words fields."
                )
            expected_words = _words_from_field(row["expected_words"], field_name="expected_words")
            predicted_words = _words_from_field(row["predicted_words"], field_name="predicted_words")
            records.append(
                {
                    "sample_id": sample_id,
                    "expected_words": expected_words,
                    "predicted_words": predicted_words,
                }
            )
    return records


def _levenshtein_alignment(expected: Sequence[str], predicted: Sequence[str]) -> tuple[int, int, int, list[tuple[int, int]]]:
    m, n = len(expected), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0)
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if expected[i - 1] == predicted[j - 1] else 1
            choices = [
                (dp[i - 1][j] + 1, (i - 1, j)),
                (dp[i][j - 1] + 1, (i, j - 1)),
                (dp[i - 1][j - 1] + cost, (i - 1, j - 1)),
            ]
            best_cost, best_prev = min(choices, key=lambda x: x[0])
            dp[i][j] = best_cost
            back[i][j] = best_prev

    i, j = m, n
    substitutions = insertions = deletions = 0
    matches: list[tuple[int, int]] = []

    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj = prev
        if pi == i - 1 and pj == j - 1:
            if expected[i - 1] == predicted[j - 1]:
                matches.append((i - 1, j - 1))
            else:
                substitutions += 1
        elif pi == i - 1 and pj == j:
            deletions += 1
        else:
            insertions += 1
        i, j = pi, pj

    matches.reverse()
    return substitutions, insertions, deletions, matches


def evaluate_multiword_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    sample_count = 0
    exact_count = 0
    total_expected_words = 0
    total_sub = total_ins = total_del = 0
    matched_word_count = 0
    total_boundary_abs_error_ms = 0.0
    total_boundary_count = 0

    for index, record in enumerate(records):
        sample_count += 1
        expected_words = _words_from_field(record.get("expected_words"), field_name="expected_words")
        predicted_words = _words_from_field(record.get("predicted_words"), field_name="predicted_words")
        expected_tokens = [word["word"] for word in expected_words]
        predicted_tokens = [word["word"] for word in predicted_words]

        sub, ins, dele, matches = _levenshtein_alignment(expected_tokens, predicted_tokens)
        total_sub += sub
        total_ins += ins
        total_del += dele
        total_expected_words += len(expected_tokens)

        is_exact = expected_tokens == predicted_tokens
        if is_exact:
            exact_count += 1

        sample_boundary_abs_error = 0.0
        sample_boundary_count = 0
        for expected_idx, predicted_idx in matches:
            expected = expected_words[expected_idx]
            predicted = predicted_words[predicted_idx]
            if (
                expected.get("start_ms") is not None
                and expected.get("end_ms") is not None
                and predicted.get("start_ms") is not None
                and predicted.get("end_ms") is not None
            ):
                sample_boundary_abs_error += abs(float(expected["start_ms"]) - float(predicted["start_ms"]))
                sample_boundary_abs_error += abs(float(expected["end_ms"]) - float(predicted["end_ms"]))
                sample_boundary_count += 2

        matched_word_count += len(matches)
        total_boundary_abs_error_ms += sample_boundary_abs_error
        total_boundary_count += sample_boundary_count

        rows.append(
            {
                "index": index,
                "sample_id": str(record.get("sample_id", f"sample_{index}")),
                "expected_text": " ".join(expected_tokens),
                "predicted_text": " ".join(predicted_tokens),
                "expected_word_count": len(expected_tokens),
                "predicted_word_count": len(predicted_tokens),
                "substitutions": sub,
                "insertions": ins,
                "deletions": dele,
                "exact_match": is_exact,
                "matched_word_count": len(matches),
                "timestamp_boundary_mae_ms": (
                    sample_boundary_abs_error / sample_boundary_count if sample_boundary_count else None
                ),
            }
        )

    if total_expected_words == 0:
        raise ValueError("Expected words are required to compute WER.")

    word_error_rate = (total_sub + total_ins + total_del) / total_expected_words
    timestamp_boundary_mae_ms = (
        total_boundary_abs_error_ms / total_boundary_count if total_boundary_count > 0 else None
    )

    metrics = {
        "sample_count": sample_count,
        "exact_sequence_accuracy": exact_count / sample_count if sample_count else 0.0,
        "word_error_rate": word_error_rate,
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
        "total_expected_words": total_expected_words,
        "matched_word_recall": matched_word_count / total_expected_words,
        "timestamp_boundary_mae_ms": timestamp_boundary_mae_ms,
        "timestamp_boundary_count": total_boundary_count,
    }

    return rows, metrics


def write_multiword_evaluation_artifacts(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    out_dir: Path | str,
) -> MultiwordEvaluationArtifacts:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    fieldnames = [
        "index",
        "sample_id",
        "expected_text",
        "predicted_text",
        "expected_word_count",
        "predicted_word_count",
        "substitutions",
        "insertions",
        "deletions",
        "exact_match",
        "matched_word_count",
        "timestamp_boundary_mae_ms",
    ]

    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return MultiwordEvaluationArtifacts(predictions_csv=predictions_path, metrics_json=metrics_path)
