# src/models Deprecation/Removal Plan (Safe)

## Current reality (verified)

`src/models/*` is still active:

- `gemma_finetune.py` used by:
  - `scripts/test_finetuning.py`
  - `scripts/phase2a_run.py`
  - `tests/test_gemma_finetune.py`
- `gemma_loader.py` used by:
  - `scripts/test_finetuning.py`
  - `src/mobile/cactus_export.py`
- `tcn_baseline.py` used by:
  - `scripts/run_prerecorded_fallback.py`
  - `tests/test_tcn_baseline.py`
- `utils.py` used by:
  - `src/models/gemma_finetune.py`
  - `src/data/pose_to_text_dataset.py`

So deleting `src/models` now would break active paths.

## Goal

Remove training-era model modules safely, while preserving current Top-50 evaluation + demo flows.

## Phase 1 (decouple, no removals yet)

1. ✅ Move checkpoint-loading helper used by `src/mobile/cactus_export.py` into a mobile-local module.
2. ✅ Update `cactus_export.py` import to new seam (`src/mobile/checkpoint_loader.py`).
3. Keep backward-compatible wrappers in `src/models/gemma_loader.py` temporarily.

Remaining `gemma_loader` dependency after this step:
- `scripts/test_finetuning.py`

Gate:
- `npm run test` ✅
- `npm run typecheck` ✅

## Phase 2 (retire finetuning path if intentionally out-of-scope)

If we are no longer running local fine-tuning in this repo:

1. Mark these scripts as deprecated and remove from runbook/docs:
   - `scripts/test_finetuning.py`
   - `scripts/phase2a_run.py` (only if fully retired)
2. Remove tests tied only to retired training path:
   - `tests/test_gemma_finetune.py`
3. Remove `src/models/gemma_finetune.py` once no imports remain.

Gate:
- dependency grep shows no imports of `src.models.gemma_finetune`
- full test/typecheck pass

## Phase 3 (remove `gemma_loader.py`)

After Phase 1 + Phase 2:

1. Ensure no imports of `src.models.gemma_loader` remain.
2. Delete `src/models/gemma_loader.py`.

Gate:
- full test/typecheck pass

## Phase 4 (evaluate `tcn_baseline.py` and `utils.py`)

- Keep `tcn_baseline.py` as long as prerecorded fallback A is part of demo path.
- For `utils.py`, split and keep only what is still needed (`normalize_pose_embeddings` for dataset path).
- Remove dead helpers only after import graph is clean.

## Deletion policy

Only delete when:
- zero live imports
- docs updated
- tests pass without that module
- one commit per phase for clean rollback
