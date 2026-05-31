from __future__ import annotations

from src.evaluation.colab_gatekeeper import evaluate_colab_gatekeeper


def _contract() -> dict[str, object]:
    anchors = [f"g{i}" for i in range(10)]
    return {
        "allowlist": anchors + ["extra"],
        "anchors": [{"gloss": gloss, "count": 10} for gloss in anchors],
        "alias_map": [{"alias": "g0_alias", "canonical": "g0"}],
    }


def _rows_all_pass() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(10):
        gloss = f"g{i}"
        for j in range(3):
            rows.append(
                {
                    "sample_id": f"{gloss}-{j}",
                    "expected_gloss": gloss,
                    "first_pass_raw": gloss,
                    "first_pass_valid": True,
                    "retry_used": False,
                    "retry_raw": "",
                    "retry_valid": False,
                    "final_gloss": gloss,
                    "final_valid": True,
                    "correct": True,
                }
            )
    return rows


def test_gatekeeper_passes_when_every_anchor_meets_threshold_and_no_collapse() -> None:
    result = evaluate_colab_gatekeeper(contract=_contract(), prediction_rows=_rows_all_pass())

    assert result["final_after_retry"]["pass"] is True
    assert result["final_after_retry"]["reasons"] == []
    assert all(item["pass"] for item in result["final_after_retry"]["per_anchor"])
    assert result["final_after_retry"]["collapse"]["mode_ratio"] <= 0.4


def test_gatekeeper_hard_fails_on_collapse_ratio_above_threshold() -> None:
    rows = _rows_all_pass()
    # Force collapse: 13/30 rows predict g0 (> 40%)
    for idx in range(13):
        rows[idx]["final_gloss"] = "g0"
        rows[idx]["final_valid"] = True
        rows[idx]["correct"] = rows[idx]["expected_gloss"] == "g0"

    result = evaluate_colab_gatekeeper(contract=_contract(), prediction_rows=rows)

    assert result["final_after_retry"]["pass"] is False
    assert any("collapse_detected" in reason for reason in result["final_after_retry"]["reasons"])


def test_gatekeeper_fails_anchor_below_70_percent() -> None:
    rows = _rows_all_pass()
    # Anchor g3 has 1/3 correct => 33%
    for row in rows:
        if row["expected_gloss"] == "g3":
            row["final_gloss"] = "g0"
            row["final_valid"] = True
            row["correct"] = False
    rows[-1]["expected_gloss"] = "g3"
    rows[-1]["final_gloss"] = "g3"
    rows[-1]["correct"] = True

    result = evaluate_colab_gatekeeper(contract=_contract(), prediction_rows=rows)

    assert result["final_after_retry"]["pass"] is False
    assert any("per_anchor_below_threshold" in reason for reason in result["final_after_retry"]["reasons"])


def test_retry_effect_reports_decision_change_when_retry_flips_fail_to_pass() -> None:
    rows = _rows_all_pass()
    # Make first-pass invalid for every sample (first pass fails all anchors),
    # but final-after-retry remains fully correct.
    for row in rows:
        row["first_pass_raw"] = "oov"
        row["first_pass_valid"] = False
        row["retry_used"] = True

    result = evaluate_colab_gatekeeper(contract=_contract(), prediction_rows=rows)

    assert result["first_pass"]["pass"] is False
    assert result["final_after_retry"]["pass"] is True
    assert result["retry_effect"]["decision_changed"] is True
    assert result["retry_effect"]["from"] == "fail"
    assert result["retry_effect"]["to"] == "pass"


def test_gatekeeper_validates_anchor_support_range_3_to_5() -> None:
    rows = _rows_all_pass()
    # Push g0 support to 6 rows (out of range)
    rows.extend(
        [
            {
                "sample_id": "g0-extra-1",
                "expected_gloss": "g0",
                "first_pass_raw": "g0",
                "first_pass_valid": True,
                "retry_used": False,
                "retry_raw": "",
                "retry_valid": False,
                "final_gloss": "g0",
                "final_valid": True,
                "correct": True,
            },
            {
                "sample_id": "g0-extra-2",
                "expected_gloss": "g0",
                "first_pass_raw": "g0",
                "first_pass_valid": True,
                "retry_used": False,
                "retry_raw": "",
                "retry_valid": False,
                "final_gloss": "g0",
                "final_valid": True,
                "correct": True,
            },
            {
                "sample_id": "g0-extra-3",
                "expected_gloss": "g0",
                "first_pass_raw": "g0",
                "first_pass_valid": True,
                "retry_used": False,
                "retry_raw": "",
                "retry_valid": False,
                "final_gloss": "g0",
                "final_valid": True,
                "correct": True,
            },
        ]
    )

    result = evaluate_colab_gatekeeper(contract=_contract(), prediction_rows=rows)

    assert result["final_after_retry"]["pass"] is False
    assert any("anchor_support_out_of_range" in reason for reason in result["final_after_retry"]["reasons"])


def test_collapse_check_ignores_invalid_predictions_as_mode_gloss() -> None:
    rows = _rows_all_pass()
    for row in rows:
        row["final_valid"] = False
        row["final_gloss"] = ""
        row["correct"] = False

    result = evaluate_colab_gatekeeper(contract=_contract(), prediction_rows=rows)

    assert result["final_after_retry"]["pass"] is False
    assert not any("collapse_detected" in reason for reason in result["final_after_retry"]["reasons"])
