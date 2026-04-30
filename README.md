# ASL Transcription System

ASL transcription pipeline for a Kaggle hackathon project focused on real-time
ASL transcription on mobile devices. Phase 1 and 2 cover the data pipeline,
pose processing, Gemma loading, and fine-tuning infrastructure.

## Architecture

`MediaPipe -> Pose Encoder -> Gemma 4 2B-E2B -> Text`

## Project Layout

```text
src/
  data/
  models/
  mobile/
data/
  raw/
  processed/
notebooks/
tests/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Fine-Tuning Smoke Test

Run the fast fine-tuning test before starting a full training job:

```bash
python scripts/test_finetuning.py --max-samples 8 --batch-size 2
```

The default command loads `google/gemma-4-2b-e2b-instruct` through Unsloth,
loads up to 50 pose samples, runs one short epoch, checks that the forward pass
and optimization step work, verifies the loss trend, and saves a checkpoint to
`checkpoints/gemma_asl_smoke/smoke-checkpoint`.

For CPU-only machines, CI, or environments that do not yet have Hugging Face
access configured, run the same pipeline contract with a tiny local model:

```bash
python scripts/test_finetuning.py --mock-model --max-samples 8 --batch-size 2
```

If extracted WLASL pose data is not available, the script creates a tiny
synthetic pose dataset under the smoke output directory. To test a specific
manifest, pass:

```bash
python scripts/test_finetuning.py \
  --manifest data/processed/training_pairs/train.csv \
  --pose-root data/processed/poses \
  --max-samples 50
```

Expected output:

```text
Fine-tuning smoke test summary
====================================
[PASS] model_loads: model and tokenizer loaded
[PASS] data_pipeline: batch pose shape=(2, ..., ...)
[PASS] forward_and_train: ran ... optimization steps
[PASS] loss_decreases: same-batch before=... after=...
[PASS] checkpoint_saved: checkpoint=.../smoke-checkpoint
Runtime: ...s
```

The notebook walkthrough is in `notebooks/05_test_pipeline.ipynb`. It shows the
same flow step by step: model loading, dataset loading, one-epoch training,
loss plotting, and checkpoint inspection.

## Unit Tests

Fine-tuning utility tests are in `tests/test_gemma_finetune.py`:

```bash
pytest tests/test_gemma_finetune.py
```

The tests cover `FineTuneConfig`, gradient clipping, the cosine warmup learning
rate scheduler, and checkpoint save/load state.

## Top-50 Split Artifacts

Generate the fixed Top-50 ASL gloss contract plus both random and
signer-independent train/validation/test splits with:

```bash
python scripts/prepare_training_data.py --top50-only
```

The contract is versioned at `data/contracts/asl_top50_glosses_v1.json`. Split
artifacts are written under `data/processed/splits/top50/`.

## Prerecorded Fallback Demo

Fallback A runs a prerecorded media file through the same live feature stream
and TCN prediction path used by live mode:

```bash
python scripts/run_prerecorded_fallback.py \
  --media-path data/demo/prerecorded_clip.mp4 \
  --frame-count 8
```

The script prints a JSON payload with `mode: prerecorded`, confidence-aware
display text, and the observable media path so the mode switch is visible in
logs or UI plumbing. Lightweight `.npy` clips with shape
`(frames, height, width, channels)` are also supported for deterministic tests.

## Troubleshooting

- `unsloth is required`: install dependencies with `pip install -r requirements.txt`.
- `Gemma load failed`: verify Hugging Face authentication, model access, and
  available VRAM. Use `--mock-model` for a local pipeline check.
- CUDA out of memory: reduce `--batch-size`, reduce `--max-samples`, or run on a
  GPU with more VRAM. The script already uses a low LoRA rank for smoke tests.
- `Dataset manifest not found`: pass the correct `--manifest`, or omit it to let
  the smoke test create synthetic pose data.
- `Pose archive missing required body/hand components`: regenerate pose archives
  with `scripts/extract_poses_batch.py`; each `.npz` needs `body`, `left_hand`,
  and `right_hand` arrays.
- Loss sanity check fails: rerun once to rule out small-sample noise. If it keeps
  failing on real data, inspect the labels, tokenization, learning rate, and
  whether model parameters are trainable.

## Phase Scope

- Project scaffolding
- WLASL data loading
- Pose extraction with MediaPipe Holistic
- Train/validation/test split generation
- Basic exploratory notebooks and tests
- Gemma loading, LoRA fine-tuning utilities, checkpointing, and smoke tests

## Notes

- WLASL metadata is downloaded from the public GitHub repository.
- Video files should be stored under `data/raw/wlasl/videos/`.
- Extracted pose sequences are written to `data/processed/`.
