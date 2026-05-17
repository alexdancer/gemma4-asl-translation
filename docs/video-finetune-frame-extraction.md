# Video Fine-Tune Frame Extraction

This branch trains Gemma-4 E4B multimodal/Unsloth on extracted image frames, not native video files.

## Local inputs

Expected local WLASL layout:

```text
data/WLASL/
  start_kit/
    WLASL_v0.3.json
    videos/
      {gloss}/
        {video_id}.mp4
```

`data/WLASL/` is local-only and ignored by git.

## Output layout

Generated artifacts are written under `data/video_finetune/`, also ignored by git:

```text
data/video_finetune/
  full/
    frames/
    manifest.jsonl
    train.jsonl
    val.jsonl
    test.jsonl
    failures.jsonl
    summary.json

  top50/
    frames/
    manifest.jsonl
    train.jsonl
    val.jsonl
    test.jsonl
    failures.jsonl
    summary.json
    labels.txt
```

Each valid video sample produces exactly 30 ordered JPG frames at 448x448 by default.

## Recommended first run: Top-50 only

Use this for initial fine-tuning:

```bash
.venv-py312/bin/python scripts/data/extract_wlasl_video_frames.py \
  --datasets top50 \
  --output-root data/video_finetune \
  --num-frames 30 \
  --image-size 448 \
  --overwrite
```

This creates only:

```text
data/video_finetune/top50/
```

## Later run: full WLASL plus Top-50

Use this when you want both the full dataset and the dedicated Top-50 subset:

```bash
.venv-py312/bin/python scripts/data/extract_wlasl_video_frames.py \
  --datasets both \
  --output-root data/video_finetune \
  --num-frames 30 \
  --image-size 448 \
  --overwrite
```

This creates:

```text
data/video_finetune/full/
data/video_finetune/top50/
```

## Important note

You do **not** need to run both commands for initial fine-tuning. Start with the Top-50 command only.

## Fine-tuning notebook

Use this Colab notebook after generating `data/video_finetune/top50/`:

```text
notebooks/10_colab_gemma4_e4b_video_top50_finetune.ipynb
```

It loads `train.jsonl`, `val.jsonl`, `test.jsonl`, `labels.txt`, and the copied `frames/` folder, then builds Unsloth `FastVisionModel` multimodal messages with 30 images plus one Top-50 gloss-classification prompt.

## Smoke test command

For a quick small run without generating the full 448x448 dataset:

```bash
.venv-py312/bin/python scripts/data/extract_wlasl_video_frames.py \
  --datasets top50 \
  --max-samples 2 \
  --output-root /tmp/asl_video_finetune_smoke \
  --num-frames 4 \
  --image-size 32 \
  --overwrite
```

## Failure handling

Missing or unreadable preprocessed videos are skipped and recorded in `failures.jsonl`; they do not stop the full extraction run.

Top-50 is computed from locally available preprocessed clips, not theoretical WLASL metadata counts.
