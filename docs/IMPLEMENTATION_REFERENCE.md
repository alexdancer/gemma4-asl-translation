# Implementation Reference (Current)

This document is the current code map for the Top-50 proof/evaluation/demo flow.

## Core Modules

- `src/evaluation/unsloth_asl.py`
  - q64 JSONL loading
  - manifest label normalization/validation
  - free-generation inference contract (mock + real checkpoint)
  - metrics + artifact writing
  - constrained/prompt-control comparison helpers
- `src/demo/output_contract.py`
  - shared demo output Interface (`ok` / `uncertain` / `error`)
- `src/demo/fallback_a.py`
  - prerecorded media fallback through live-style inference seam
- `src/demo/fallback_b.py`
  - precomputed replay fallback seam
- `src/demo/prerecorded_q64.py`
  - demo-scoped known-good q64 record path with readiness artifact output
- `src/demo/prompt_control_reference.py`
  - prompt-control free-generation fixture selector for stable smoke/demo samples
- `src/mobile/cactus_prompt_control_parity.py`
  - Cactus Engine prompt-control parity harness/report contract with a mockable runner seam and honest real-run failure reporting
- `src/data/cached_pose_q64.py`
  - cached/precomputed pose archive to q64 JSONL compatibility verifier
  - writes dedicated verification artifacts outside evaluation metrics and demo readiness outputs

## Main Script Adapters

- `scripts/data/verify_cached_pose_q64.py`
  - verifies one known Top-50 cached pose archive against manifest labels and q64 sample records
- `scripts/evaluation/evaluate_unsloth_asl.py`
  - held-out free-generation evaluation (mock or real)
- `scripts/evaluation/evaluate_unsloth_asl_constrained.py`
  - constrained diagnostic and root-cause report path
- `scripts/evaluation/evaluate_unsloth_asl_prompt_control.py`
  - prompt/output-control experiment path
- `scripts/evaluation/build_prompt_control_reference.py`
  - selects one smoke sample plus demo samples that are correct under prompt-control free generation and writes `evaluation/results/prompt_control_reference/reference.json`
- `scripts/demo/run_prerecorded_q64_demo.py`
  - known-good demo path runner
- `scripts/mobile/run_cactus_prompt_control_parity.py`
  - compares Cactus Engine completions against the prompt-control reference fixture and writes `evaluation/results/cactus_prompt_control_parity/parity_report.json`
- `scripts/demo/run_prerecorded_fallback.py`
  - fallback A runner
- `scripts/demo/run_precomputed_replay.py`
  - fallback B runner

## Test Surface

- `tests/evaluation/test_unsloth_asl_evaluator.py`
- `tests/demo/test_prerecorded_q64_demo.py`
- `tests/demo/test_prerecorded_fallback.py`
- `tests/demo/test_precomputed_replay_fallback.py`
- `tests/demo/test_demo_output_contract.py`

## Notes

- This project treats free-generation strict normalized exact-match as the primary proof metric.
- Constrained scoring and prompt-control are diagnostic modules, not replacements for the primary metric.
- Build the prompt-control reference fixture with:
  `python scripts/project_python.py scripts.evaluation.build_prompt_control_reference`
  Use `--predictions-csv evaluation/results/unsloth_top50_q64_full_dashboard_baseline_prompt_control/predictions.csv` to select from an existing prompt-control run without loading the checkpoint.
- Run the CI-safe Cactus parity harness with:
  `PYTHONPATH=. ./venv/bin/python scripts/mobile/run_cactus_prompt_control_parity.py --reference evaluation/results/prompt_control_reference/reference.json --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json --cactus-weights mock-cactus-weights --out-dir evaluation/results/cactus_prompt_control_parity --max-samples 1 --mock-cactus-output hearing`
  Omit `--mock-cactus-output` only when a real Cactus-converted weights directory exists; missing weights are reported as `runtime_mode: cactus_engine` with `real_cactus_parity_proven: false`.
- Verify one cached pose archive can emit the shared q64 contract with:
  `python scripts/data/verify_cached_pose_q64.py --pose-path path/to/sample.npz --sample-id hearing_26986 --expected-gloss hearing`
  The default artifacts are written under `data/processed/verification/cached_pose_q64`.
  The verifier uses the matching record from `--records` as the q64 shape contract, resamples cached pose frames to that frame count, and rejects encoding, feature-count, or payload-shape mismatches.
- Demo paths are explicitly scoped to supported Top-50 signs and are not production-grade ASL recognition.
