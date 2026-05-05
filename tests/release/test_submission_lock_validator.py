import json
import subprocess
import sys
from pathlib import Path

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
    assert readiness["package"]["complete"] is False


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


def test_build_readiness_fails_when_required_freeze_gate_missing() -> None:
    checklist = _base_checklist()
    del checklist["freeze_gates"]["demo_writeup_freeze"]
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 5))
    assert any("missing freeze gate: demo_writeup_freeze" in error for error in errors)
    assert readiness["freeze_gates_satisfied"] is False


def test_build_readiness_fails_when_required_freeze_gate_date_invalid() -> None:
    checklist = _base_checklist()
    checklist["freeze_gates"]["feature_freeze"]["date"] = "2026-13-99"
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 5))
    assert any("invalid gate date for feature_freeze" in error for error in errors)
    assert readiness["freeze_gates_satisfied"] is False


def test_build_readiness_rejects_placeholder_demo_video_when_marked_ready() -> None:
    checklist = _base_checklist()
    checklist["package_inputs"]["demo_video"]["value"] = (
        "https://github.com/alexdancer/sign-language-asl/issues/44#issuecomment-placeholder-demo-video"
    )
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 5))
    assert any("demo_video is marked ready but value appears to be a placeholder" in error for error in errors)
    assert readiness["package"]["complete"] is False


def test_build_readiness_rejects_non_dict_package_inputs() -> None:
    checklist = _base_checklist()
    checklist["package_inputs"] = []
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 5))
    assert any("package_inputs must be an object/dict" in error for error in errors)
    assert readiness["package"]["complete"] is False


def test_build_readiness_rejects_non_dict_freeze_gates() -> None:
    checklist = _base_checklist()
    checklist["freeze_gates"] = []
    readiness, errors = build_readiness(checklist, today=date(2026, 5, 5))
    assert any("freeze_gates must be an object/dict" in error for error in errors)
    assert readiness["freeze_gates_satisfied"] is False


def test_cli_invalid_as_of_date_returns_validation_error(tmp_path) -> None:
    checklist_path = tmp_path / "bad_as_of_date_checklist.json"
    out_path = tmp_path / "artifact.json"

    checklist = _base_checklist()
    checklist["as_of_date"] = "2026-13-99"
    checklist_path.write_text(json.dumps(checklist), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/release/validate_submission_lock.py",
            "--checklist",
            str(checklist_path),
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "invalid as_of_date" in result.stdout
    assert "Traceback" not in result.stderr
