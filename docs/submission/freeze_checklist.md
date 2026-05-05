# Submission Freeze Checklist (Issue #44)

This checklist enforces the locked freeze milestones from the PRD.

## Gate Status (as of 2026-05-05, America/Chicago)

| Gate | Date | Status | Satisfied |
| --- | --- | --- | --- |
| Feature freeze | 2026-05-12 | scheduled | yes (not due yet) |
| Demo/write-up freeze | 2026-05-15 | scheduled | yes (not due yet) |

## Enforcement Rules

- If current date is before a freeze date, `scheduled` is acceptable.
- If current date is on/after a freeze date, the gate must be `pass`.
- Failing or overdue gates are surfaced as open risks in the final readiness artifact.

## Validator Command

```bash
./venv/bin/python scripts/release/validate_submission_lock.py
```

Outputs:
`evaluation/results/submission_lock/final_readiness_artifact.json`
