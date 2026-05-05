from datetime import date

from scripts.release.validate_submission_lock import build_readiness


def _base_checklist() -> dict:
    return {
        "issue": 44,
        "as_of_date": "2026-05-05",
        "package_inputs": {
            "demo_video": {"status": "ready", "value": "https://example.com/demo.mp4", "notes": "ok"},
            "project_repository": {"status": "ready", "value": "https://github.com/alexdancer/sign-language-asl", "notes": "ok"},
            "reproducibility_bundle": {"status": "blocked", "value": "https://github.com/alexdancer/sign-language-asl/issues/41", "notes": "blocked"},
            "promotion_gate_report": {"status": "blocked", "value": "https://github.com/alexdancer/sign-language-asl/issues/43", "notes": "blocked"},
            "demo_failover_artifact": {"status": "ready", "value": "evaluation/results/prompt_control_reference/reference.json", "notes": "ok"},
            "metrics_dashboard_artifact": {"status": "ready", "value": "evaluation/results/unsloth_top50_q64_full_dashboard_baseline_prompt_control/report.md", "notes": "ok"},
            "writeup_problem_constraints": {"status": "ready", "value": "docs/PRD_CACTUS_IOS_PHASE1_AND_100_GLOSS_EXPANSION.md", "notes": "ok"},
            "writeup_routing_fallback": {"status": "ready", "value": "docs/PRD_CACTUS_IOS_PHASE1_AND_100_GLOSS_EXPANSION.md", "notes": "ok"},
        },
        "freeze_gates": {
            "feature_freeze": {"date": "2026-05-12", "status": "scheduled"},
            "demo_writeup_freeze": {"date": "2026-05-15", "status": "scheduled"},
        },
    }


def test_build_readiness_allows_scheduled_future_freezes() -> None:
    readiness, errors = build_readiness(_base_checklist(), today=date(2026, 5, 5))
    assert errors == []
    assert readiness["freeze_gates_satisfied"] is True
    assert readiness["package"]["blocked"] == 2


def test_build_readiness_flags_due_freeze_without_pass() -> None:
    checklist = _base_checklist()
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 16))
    assert errors == []
    assert readiness["freeze_gates"]["feature_freeze"]["due"] is True
    assert readiness["freeze_gates"]["feature_freeze"]["satisfied"] is False
    assert readiness["freeze_gates_satisfied"] is False
    assert any("feature_freeze due" in risk for risk in readiness["open_risks"])


def test_build_readiness_fails_missing_required_inputs() -> None:
    checklist = _base_checklist()
    del checklist["package_inputs"]["demo_video"]
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 5))
    assert errors
    assert any("missing package input keys" in error for error in errors)
    assert readiness["package"]["complete"] is False
