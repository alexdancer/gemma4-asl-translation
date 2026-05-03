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
- `src/demo/fallback_a.py` / `src/demo/fallback_b.py` — fallback demo modes

## Script adapters

- `scripts/evaluate_unsloth_asl.py`
- `scripts/evaluate_unsloth_asl_constrained.py`
- `scripts/evaluate_unsloth_asl_prompt_control.py`
- `scripts/run_prerecorded_q64_demo.py`
- `scripts/run_prerecorded_fallback.py`
- `scripts/run_precomputed_replay.py`

## Quick commands

```bash
# tests + compile checks
npm run test
npm run typecheck

# evaluator smoke path
python scripts/evaluate_unsloth_asl.py --mock \
  --test-file data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --out-dir evaluation/results/unsloth_asl_mock_smoke --max-samples 50
```

## Docs

- `docs/PRD_UNSLOTH_TOP50_PROOF_AND_DEMO.md` — product/decision context
- `docs/QUICK_START_FINETUNING.md` — current runbook commands
- `docs/IMPLEMENTATION_REFERENCE.md` — current module map
- `docs/PRERECORDED_Q64_DEMO.md` — demo-scope and known-good path

## Scope guardrails

- Free-generation strict normalized exact-match remains the primary proof metric.
- Constrained and prompt-control paths are diagnostic modules.
- Demo paths are explicitly scoped to supported Top-50 signs and are not production-grade ASL recognition.
