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
- `scripts/demo/run_prerecorded_fallback.py`
  - fallback A runner
- `scripts/demo/run_precomputed_replay.py`
  - fallback B runner

## Test Surface

- `tests/test_unsloth_asl_evaluator.py`
- `tests/test_prerecorded_q64_demo.py`
- `tests/test_prerecorded_fallback.py`
- `tests/test_precomputed_replay_fallback.py`
- `tests/test_demo_output_contract.py`

## Notes

- This project treats free-generation strict normalized exact-match as the primary proof metric.
- Constrained scoring and prompt-control are diagnostic modules, not replacements for the primary metric.
- Build the prompt-control reference fixture with:
  `python scripts/project_python.py scripts.evaluation.build_prompt_control_reference`
  Use `--predictions-csv evaluation/results/unsloth_top50_q64_full_dashboard_baseline_prompt_control/predictions.csv` to select from an existing prompt-control run without loading the checkpoint.
- Verify one cached pose archive can emit the shared q64 contract with:
  `python scripts/data/verify_cached_pose_q64.py --pose-path path/to/sample.npz --sample-id hearing_26986 --expected-gloss hearing`
  The default artifacts are written under `data/processed/verification/cached_pose_q64`.
  The verifier also compares generated q64 encoding, frame count, feature count, and payload shape with the matching record from `--records`.
- Demo paths are explicitly scoped to supported Top-50 signs and are not production-grade ASL recognition.
