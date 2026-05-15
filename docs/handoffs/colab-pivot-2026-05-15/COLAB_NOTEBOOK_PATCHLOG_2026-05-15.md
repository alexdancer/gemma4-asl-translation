# Colab Notebook Patchlog — 2026-05-15

## Scope
This patchlog documents fixes applied to `notebooks/09_colab_demo_self_contained.ipynb` to align Colab demo behavior with full Top-50 evaluation intent.

## Problem observed
- Notebook run produced only 6 demo samples (`no, need, yes, again, drink, mother`) despite `TOP_K = 50`.
- Gatekeeper previously failed support constraints when using 1-sample-per-label test split.

## Root cause
1. **Anchor selection was filtered by `ALLOWSET` that came from a placeholder conversational allowlist**, not the canonical Top-50 manifest labels.
   - Result: only overlapping labels from placeholder list were retained.
2. Gatekeeper defaults were originally tuned for 3–5 samples per anchor.

## Changes applied
1. **Top-K expansion for demo row selection**
   - `TOP_K = 50`
   - `PER_ANCHOR = 1` (aligned to this test split’s one-sample-per-label structure)

2. **Canonical allowlist override from manifest in row-builder cell**
   - Added logic to load labels from:
     - `/content/asl_unsloth_pose_train_q64_full_top50_manifest.json`
   - Overrides placeholder allowlist at runtime:
     - `ALLOWLIST = [normalize_gloss(x) for x in manifest['labels']]`
     - `ALLOWSET = set(ALLOWLIST)`
   - Prints loaded label count for visibility.

3. **Gatekeeper alignment for full test split run**
   - `min_n=1, max_n=1`
   - Removed 10-anchor cap:
     - from `anchors = ...[:10]`
     - to `anchors = ...`

## Expected behavior after patch
- DEMO row builder should target all manifest labels and build 50 rows (assuming test artifact availability and integrity).
- Gatekeeper should no longer fail due to support-range mismatch for this split mode.

## Run evidence captured by user (post-fix scoring cell)
- Samples: 50
- Correct: 27
- Accuracy: 54.00%
- Valid predictions: 27/50 (54.00%)
- Unknown predictions: 23/50 (46.00%)

## Notes
- Inference run used Unsloth + Gemma4 LoRA path (`AlexD281/asl-gemma4-e2b-q64-top50-lora`) with warning about missing adapter keys; run still completed.
- This commit intentionally scopes to notebook + documentation updates only.
