# Colab Execution Checklist — ASL E2B Top-50 (Deadline Mode)

## Goal
Fast direct model validation in Colab. Ignore endpoint/app integration until model behavior stable.

## Inputs
- Repo: `AlexD281/sign-language-asl` (or local upload)
- Model: `AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit`
- Top-50 manifest: `data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json`
- Test records: `data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl`

## Colab setup
1. Runtime: GPU (T4/L4/A100 available).
2. Install deps from repo runtime/eval scripts.
3. Set HF auth token (`huggingface_hub.login(...)` or env).
4. Set `UNSLOTH_DISABLE_STATISTICS=1` (prevents known timeout issue).

## Execution plan
1. **Sanity load**
   - Load tokenizer/model once.
   - Run 1 known prompt sample end-to-end.

2. **Focused sign checks**
   - Build mini set including `yes` and `thank you` clips/signals.
   - Run direct inference, capture raw output + normalized gloss.

3. **Top-50 mini regression**
   - Run 10–20 representative samples.
   - Record:
     - expected gloss
     - predicted gloss
     - raw output
     - confidence (if available)

4. **Metrics + failure modes**
   - Compute top1 accuracy on mini set.
   - Compute repeated-output rate (collapse detector).
   - List common confusions.

5. **Decision gate**
   - If collapse persists (single gloss dominates), adjust prompt/control and rerun.
   - If stable, freeze prompt/config for endpoint reintegration.

## Required artifacts to save
- `colab_predictions.csv`
- `colab_metrics.json`
- `colab_run_notes.md`
- Optional: confusion matrix image/table

## Success criteria (deadline practical)
- `yes` and `thank you` no longer both collapse to same wrong token
- Mini regression accuracy acceptable for demo scope
- Output format deterministic enough to wrap back into endpoint later

## Next step after success
Return to Hermes with artifacts + frozen config. Then re-enable endpoint path using same proven prompt/control settings.
