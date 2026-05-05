# ASL Top-50 Proof + Demo Workspace

This repo is the project workspace for the ASL Top-50 q64 proof-of-learning and demo-safe integration flow.

## Primary workflow

1. Prepare/freeze Top-50 data contract + splits.
2. Evaluate held-out q64 records with strict normalized exact-match metrics.
3. Run diagnostics (constrained / prompt-control) only when needed.
4. Run demo-scoped prerecorded known-good paths using the shared q64 prediction contract.

## Key seams

- `src/evaluation/unsloth_asl.py` — evaluation contract, metrics, artifacts
- `src/demo/output_contract.py` — demo output contract
- `src/demo/prerecorded_q64.py` — known-good prerecorded q64 demo path
- `src/demo/constrained_top50.py` — optional diagnostic/demo-safe constrained Top-50 fallback
- `src/demo/fallback_a.py` / `src/demo/fallback_b.py` — fallback demo modes

## Script adapters

- `scripts/data/verify_cached_pose_q64.py`
- `scripts/evaluation/evaluate_unsloth_asl.py`
- `scripts/evaluation/evaluate_unsloth_asl_constrained.py`
- `scripts/evaluation/evaluate_unsloth_asl_prompt_control.py`
- `scripts/demo/run_prerecorded_q64_demo.py`
- `scripts/demo/run_constrained_top50_demo.py`
- `scripts/demo/run_prerecorded_fallback.py`
- `scripts/demo/run_precomputed_replay.py`

## Quick commands

```bash
# tests + compile checks
npm run test
npm run typecheck

# evaluator smoke path
python scripts/evaluation/evaluate_unsloth_asl.py --mock \
  --test-file data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --out-dir evaluation/results/unsloth_asl_mock_smoke --max-samples 50

# cached pose -> q64 verification path
python scripts/data/verify_cached_pose_q64.py \
  --pose-path path/to/hearing_26986.npz \
  --sample-id hearing_26986 \
  --expected-gloss hearing

# submission lock checklist validator
./venv/bin/python scripts/release/validate_submission_lock.py
```

The verifier uses the matching record in `--records` as the q64 shape contract;
cached archives are resampled to that frame count and feature-count mismatches
fail before writing evaluation metrics or readiness artifacts.

## Docs

- `docs/IMPLEMENTATION_REFERENCE.md` — current module map
- `docs/PRERECORDED_Q64_DEMO.md` — demo-scope and known-good path
- `docs/ISSUE_30_CACTUS_PROMPT_CONTROL_PARITY_PLAN.md` — active Cactus parity implementation plan
- `docs/VALIDATION_PIPELINE_CODE_GUIDE.md` — guide to the current staged validation pipeline
- `docs/submission/submission_package_inputs.md` — required demo video, links, and write-up input tracker
- `docs/submission/freeze_checklist.md` — feature/demo-writeup freeze enforcement checklist
- `evaluation/results/submission_lock/final_readiness_artifact.json` — machine-readable package completeness + open risks
- `docs/archive/training-proof-era/` — older training/proof-era PRDs and runbooks kept for reference

## Scope guardrails

- Free-generation strict normalized exact-match remains the primary proof metric.
- Constrained and prompt-control paths are diagnostic modules.
- Demo-safe constrained Top-50 inference is an optional fallback for scoped demos; it always chooses from canonical labels and is not a primary proof metric.
- Demo paths are explicitly scoped to supported Top-50 signs and are not production-grade ASL recognition.
