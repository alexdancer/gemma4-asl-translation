#!/usr/bin/env python3
"""Validate submission-lock checklist and emit readiness artifact."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

REQUIRED_INPUT_KEYS = (
    "demo_video",
    "project_repository",
    "reproducibility_bundle",
    "promotion_gate_report",
    "demo_failover_artifact",
    "metrics_dashboard_artifact",
    "writeup_problem_constraints",
    "writeup_routing_fallback",
)

REQUIRED_FREEZE_GATES = (
    "feature_freeze",
    "demo_writeup_freeze",
)

ALLOWED_INPUT_STATUSES = {"ready", "blocked", "pending"}
ALLOWED_GATE_STATUSES = {"scheduled", "pass", "fail"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _is_placeholder_demo_video(value: str) -> bool:
    normalized = value.strip().lower()
    return "placeholder" in normalized


def build_readiness(checklist: dict, *, today: date) -> tuple[dict, list[str]]:
    errors: list[str] = []
    risks: list[str] = []
    gate_errors: list[str] = []

    package_inputs = checklist.get("package_inputs", {})
    missing_keys = [key for key in REQUIRED_INPUT_KEYS if key not in package_inputs]
    if missing_keys:
        errors.append(f"missing package input keys: {', '.join(missing_keys)}")

    ready_count = 0
    blocked_count = 0
    pending_count = 0

    for key in REQUIRED_INPUT_KEYS:
        entry = package_inputs.get(key, {})
        status = entry.get("status")
        value = str(entry.get("value", "")).strip()
        if status not in ALLOWED_INPUT_STATUSES:
            errors.append(f"invalid status for {key}: {status!r}")
            continue
        if not value:
            errors.append(f"empty value for {key}")
        if key == "demo_video" and status == "ready" and _is_placeholder_demo_video(value):
            errors.append("demo_video is marked ready but value appears to be a placeholder")
        if status == "ready":
            ready_count += 1
        elif status == "blocked":
            blocked_count += 1
            risks.append(f"{key} is blocked: {entry.get('notes', 'no note provided')}")
        elif status == "pending":
            pending_count += 1
            risks.append(f"{key} is pending: {entry.get('notes', 'no note provided')}")

    gates = checklist.get("freeze_gates", {})
    gate_summary: dict[str, dict] = {}

    for gate_name in REQUIRED_FREEZE_GATES:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict):
            message = f"missing freeze gate: {gate_name}"
            errors.append(message)
            gate_errors.append(message)
            continue

        status = gate.get("status")
        gate_date_raw = gate.get("date")
        if status not in ALLOWED_GATE_STATUSES:
            message = f"invalid gate status for {gate_name}: {status!r}"
            errors.append(message)
            gate_errors.append(message)
            continue
        try:
            gate_date = _parse_date(gate_date_raw)
        except Exception:  # noqa: BLE001
            message = f"invalid gate date for {gate_name}: {gate_date_raw!r}"
            errors.append(message)
            gate_errors.append(message)
            continue

        due = today >= gate_date
        satisfied = status == "pass" if due else status in {"scheduled", "pass"}

        if due and status != "pass":
            risks.append(f"{gate_name} due on {gate_date.isoformat()} but status is {status}")
        elif not due and status == "fail":
            risks.append(f"{gate_name} marked fail before due date {gate_date.isoformat()}")

        gate_summary[gate_name] = {
            "date": gate_date.isoformat(),
            "status": status,
            "due": due,
            "satisfied": satisfied,
        }

    package_complete = len(missing_keys) == 0 and blocked_count == 0 and pending_count == 0 and not errors
    all_required_gates_present = set(gate_summary.keys()) == set(REQUIRED_FREEZE_GATES)
    freeze_gates_satisfied = (
        all_required_gates_present
        and not gate_errors
        and all(g.get("satisfied") for g in gate_summary.values())
    )

    readiness = {
        "schema_version": "1.0",
        "issue": checklist.get("issue"),
        "as_of_date": today.isoformat(),
        "package": {
            "required_inputs": len(REQUIRED_INPUT_KEYS),
            "ready": ready_count,
            "blocked": blocked_count,
            "pending": pending_count,
            "complete": package_complete,
        },
        "freeze_gates": gate_summary,
        "freeze_gates_satisfied": freeze_gates_satisfied,
        "open_risks": risks,
        "validation_errors": errors,
    }
    return readiness, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checklist",
        type=Path,
        default=Path("docs/submission/submission_lock_checklist.json"),
        help="Path to submission lock checklist JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/results/submission_lock/final_readiness_artifact.json"),
        help="Where to write readiness artifact JSON.",
    )
    args = parser.parse_args()

    checklist = _load_json(args.checklist)
    today = _parse_date(str(checklist.get("as_of_date", date.today().isoformat())))
    readiness, errors = build_readiness(checklist, today=today)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote readiness artifact: {args.out}")

    if errors:
        print("validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
