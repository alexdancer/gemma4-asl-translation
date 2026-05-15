"""Colab Top-50 gatekeeper metrics for issue #90.

Computes:
- per-anchor pass/fail (>=70% accuracy)
- collapse hard-fail (>40% single predicted gloss share)
- first-pass vs final (one-retry) decision deltas
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from src.evaluation.unsloth_asl import normalize_gloss


def _alias_lookup(alias_entries: Sequence[Mapping[str, str]] | None) -> dict[str, str]:
    table: dict[str, str] = {}
    for entry in alias_entries or ():
        alias = normalize_gloss(str(entry.get("alias", "")))
        canonical = normalize_gloss(str(entry.get("canonical", "")))
        if alias and canonical:
            table[alias] = canonical
    return table


def _canonical_from_raw(
    raw: str,
    *,
    allowlist: set[str],
    aliases: Mapping[str, str],
) -> str | None:
    norm = normalize_gloss(raw)
    if not norm:
        return None
    canonical = aliases.get(norm, norm)
    return canonical if canonical in allowlist else None


def _row_expected(row: Mapping[str, Any], aliases: Mapping[str, str], allowlist: set[str]) -> str:
    expected = _canonical_from_raw(str(row.get("expected_gloss", "")), allowlist=allowlist, aliases=aliases)
    if expected is None:
        raise ValueError(f"Row expected_gloss is not in allowlist after normalization: {row.get('expected_gloss')!r}")
    return expected


def _final_prediction(row: Mapping[str, Any], aliases: Mapping[str, str], allowlist: set[str]) -> str | None:
    if not bool(row.get("final_valid", False)):
        return None
    return _canonical_from_raw(str(row.get("final_gloss", "")), allowlist=allowlist, aliases=aliases)


def _first_prediction(row: Mapping[str, Any], aliases: Mapping[str, str], allowlist: set[str]) -> str | None:
    if not bool(row.get("first_pass_valid", False)):
        return None
    return _canonical_from_raw(str(row.get("first_pass_raw", "")), allowlist=allowlist, aliases=aliases)


def _build_anchor_subset(
    rows: Sequence[Mapping[str, Any]],
    anchors: Sequence[str],
    aliases: Mapping[str, str],
    allowlist: set[str],
) -> dict[str, list[Mapping[str, Any]]]:
    by_anchor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    anchor_set = set(anchors)
    for row in rows:
        expected = _row_expected(row, aliases, allowlist)
        if expected in anchor_set:
            by_anchor[expected].append(row)
    return by_anchor


def _anchor_accuracy(rows: Sequence[Mapping[str, Any]], *, use_final: bool, aliases: Mapping[str, str], allowlist: set[str]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        expected = _row_expected(row, aliases, allowlist)
        predicted = _final_prediction(row, aliases, allowlist) if use_final else _first_prediction(row, aliases, allowlist)
        if predicted == expected:
            correct += 1
    return correct / len(rows)


def _collapse_ratio(rows: Sequence[Mapping[str, Any]], *, use_final: bool, aliases: Mapping[str, str], allowlist: set[str]) -> tuple[str, float]:
    bucket = Counter()
    total = 0
    for row in rows:
        pred = _final_prediction(row, aliases, allowlist) if use_final else _first_prediction(row, aliases, allowlist)
        if pred is not None:
            bucket[pred] += 1
        total += 1
    if total == 0 or not bucket:
        return "__none__", 0.0
    label, count = max(bucket.items(), key=lambda item: (item[1], item[0]))
    return label, count / total


def evaluate_colab_gatekeeper(
    *,
    contract: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    per_anchor_threshold: float = 0.70,
    collapse_threshold: float = 0.40,
    anchor_min_samples: int = 3,
    anchor_max_samples: int = 5,
) -> dict[str, Any]:
    allowlist = {normalize_gloss(item) for item in contract.get("allowlist", [])}
    if not allowlist:
        raise ValueError("contract.allowlist is required")

    aliases = _alias_lookup(contract.get("alias_map"))
    anchors = [normalize_gloss(item["gloss"]) for item in contract.get("anchors", [])]
    if len(anchors) != 10:
        raise ValueError(f"expected 10 anchors, got {len(anchors)}")

    anchor_rows = _build_anchor_subset(prediction_rows, anchors, aliases, allowlist)

    per_anchor_final: list[dict[str, Any]] = []
    per_anchor_first: list[dict[str, Any]] = []
    sample_range_violations: list[str] = []

    for gloss in anchors:
        rows = anchor_rows.get(gloss, [])
        support = len(rows)
        if support < anchor_min_samples or support > anchor_max_samples:
            sample_range_violations.append(gloss)

        final_acc = _anchor_accuracy(rows, use_final=True, aliases=aliases, allowlist=allowlist)
        first_acc = _anchor_accuracy(rows, use_final=False, aliases=aliases, allowlist=allowlist)

        per_anchor_final.append(
            {
                "gloss": gloss,
                "support": support,
                "accuracy": round(final_acc, 6),
                "pass": final_acc >= per_anchor_threshold,
            }
        )
        per_anchor_first.append(
            {
                "gloss": gloss,
                "support": support,
                "accuracy": round(first_acc, 6),
                "pass": first_acc >= per_anchor_threshold,
            }
        )

    all_anchor_rows = [row for gloss in anchors for row in anchor_rows.get(gloss, [])]
    final_mode, final_ratio = _collapse_ratio(all_anchor_rows, use_final=True, aliases=aliases, allowlist=allowlist)
    first_mode, first_ratio = _collapse_ratio(all_anchor_rows, use_final=False, aliases=aliases, allowlist=allowlist)

    final_anchor_failures = [entry["gloss"] for entry in per_anchor_final if not entry["pass"]]
    first_anchor_failures = [entry["gloss"] for entry in per_anchor_first if not entry["pass"]]

    final_reasons: list[str] = []
    first_reasons: list[str] = []

    if sample_range_violations:
        joined = ", ".join(sample_range_violations)
        final_reasons.append(f"anchor_support_out_of_range: {joined}")
        first_reasons.append(f"anchor_support_out_of_range: {joined}")

    if final_anchor_failures:
        final_reasons.append("per_anchor_below_threshold: " + ", ".join(final_anchor_failures))
    if first_anchor_failures:
        first_reasons.append("per_anchor_below_threshold: " + ", ".join(first_anchor_failures))

    if final_ratio > collapse_threshold:
        final_reasons.append(
            f"collapse_detected: gloss={final_mode} ratio={round(final_ratio, 6)} threshold={collapse_threshold}"
        )
    if first_ratio > collapse_threshold:
        first_reasons.append(
            f"collapse_detected: gloss={first_mode} ratio={round(first_ratio, 6)} threshold={collapse_threshold}"
        )

    final_pass = not final_reasons
    first_pass = not first_reasons

    retry_helped_rows = sum(
        1
        for row in prediction_rows
        if bool(row.get("retry_used", False)) and bool(row.get("first_pass_valid", False)) is False and bool(row.get("correct", False))
    )

    return {
        "gate_version": "colab_gatekeeper_v1",
        "thresholds": {
            "per_anchor_accuracy_min": per_anchor_threshold,
            "collapse_ratio_max_exclusive": collapse_threshold,
            "anchor_support_min": anchor_min_samples,
            "anchor_support_max": anchor_max_samples,
        },
        "first_pass": {
            "pass": first_pass,
            "reasons": first_reasons,
            "collapse": {
                "mode_gloss": first_mode,
                "mode_ratio": round(first_ratio, 6),
            },
            "per_anchor": per_anchor_first,
        },
        "final_after_retry": {
            "pass": final_pass,
            "reasons": final_reasons,
            "collapse": {
                "mode_gloss": final_mode,
                "mode_ratio": round(final_ratio, 6),
            },
            "per_anchor": per_anchor_final,
        },
        "retry_effect": {
            "decision_changed": first_pass != final_pass,
            "from": "pass" if first_pass else "fail",
            "to": "pass" if final_pass else "fail",
            "retry_helped_rows": retry_helped_rows,
        },
    }
