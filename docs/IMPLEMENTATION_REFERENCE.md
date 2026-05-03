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

## Main Script Adapters

- `scripts/evaluate_unsloth_asl.py`
  - held-out free-generation evaluation (mock or real)
- `scripts/evaluate_unsloth_asl_constrained.py`
  - constrained diagnostic and root-cause report path
- `scripts/evaluate_unsloth_asl_prompt_control.py`
  - prompt/output-control experiment path
- `scripts/run_prerecorded_q64_demo.py`
  - known-good demo path runner
- `scripts/run_prerecorded_fallback.py`
  - fallback A runner
- `scripts/run_precomputed_replay.py`
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
- Demo paths are explicitly scoped to supported Top-50 signs and are not production-grade ASL recognition.
