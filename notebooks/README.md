# Notebook Upload and Run Guide

This file explains what each notebook expects you to upload, where files should live in Google Drive, and the recommended run order.

## Common runtime setup

Use Google Colab with a high-memory GPU runtime.

Recommended for the 26B/A4B Gemma-4 notebooks:

- H100/H200 if available.
- A100-class runtime may work for some inference paths.
- A10G/T4-class runtimes are usually too small for full 30-frame 26B/A4B inference unless you reduce settings or use a smaller model.

Before running model cells:

1. Set your Hugging Face token when prompted, or set `HF_TOKEN` in the notebook environment.
2. Mount Google Drive when the notebook asks.
3. Keep the frame contract consistent unless you intentionally change the model/data contract:
   - `NUM_FRAMES = 30`
   - `FRAME_SIZE = 448`
   - RGB images
   - deterministic/even video sampling

## Expected Drive layout

The notebooks use these default Drive paths:

```text
/content/drive/MyDrive/asl/
├── phase1_zahid_top50_bundle.zip
├── video_finetune/
│   └── top50/
│       └── labels.txt
├── video_finetune/top50_bundle.zip
├── wlasl_top50_30x448_bundle.zip              # fallback for Notebook 13
├── notebook12_user_video_results.jsonl         # created by Notebook 12
└── notebook12_wlasl_top50_batch_results.jsonl  # created by Notebook 12 batch mode
```

The exact paths can be edited in notebook config cells if your Drive layout differs.

## Notebook 11 — Zahid Top-50 pretraining

File:

```text
notebooks/11_colab_gemma4_26b_zahid_phase1_pretrain.ipynb
```

Purpose:

- Train or reuse the Zahid Top-50 adapter.
- Uses `unsloth/gemma-4-26B-A4B-it`.
- Publishes/saves an adapter such as `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`.

Upload/prepare:

1. Put this ZIP in Drive:

   ```text
   /content/drive/MyDrive/asl/phase1_zahid_top50_bundle.zip
   ```

2. Put the Top-50 label file in Drive:

   ```text
   /content/drive/MyDrive/asl/video_finetune/top50/labels.txt
   ```

Expected ZIP contents:

```text
phase1_zahid_top50_bundle.zip
├── train.jsonl / val.jsonl / test.jsonl or equivalent split manifests
└── frames_30x448/ or frames/
    ├── train/<sample_id>/frame_000.jpg ... frame_029.jpg
    ├── val/<sample_id>/frame_000.jpg ... frame_029.jpg
    └── test/<sample_id>/frame_000.jpg ... frame_029.jpg
```

Run order:

1. Runtime setup/install cells.
2. Hugging Face login cell.
3. Drive mount cell.
4. Config cells.
5. Bundle unzip/manifest resolution cells.
6. Dataset construction checks.
7. Model load/training cells.
8. Adapter save/push cells.
9. Evaluation/metrics upload cells.

Notes:

- Do not switch Notebook 11 to the Google base model unless you intentionally update the full load path and verification cells.
- Keep `UNSLOTH_DISABLE_STATISTICS=1` if the notebook sets it; this avoids the Unsloth statistics timeout while preserving Hugging Face model downloads.

## Notebook 12 — user-video and batch inference

File:

```text
notebooks/12_colab_user_video_inference_top50.ipynb
```

Purpose:

- Main user-facing inference notebook.
- Runs one uploaded video or a batch ZIP.
- Logs JSONL output to Drive.

Default model:

```text
AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora
```

### Mode A: single video

Upload exactly one video when prompted.

Supported extensions:

```text
.mp4
.mov
.m4v
```

The notebook will:

1. Check duration.
2. Extract exactly 30 evenly sampled frames at 448x448.
3. Load the model adapter from Hugging Face.
4. Run deterministic generation with a small `max_new_tokens` value.
5. Normalize the raw model output.
6. Accept the prediction only if it is in the approved Top-50 list.
7. Append a row to:

   ```text
   /content/drive/MyDrive/asl/notebook12_user_video_results.jsonl
   ```

8. Delete the raw uploaded video from the Colab runtime after logging.

### Mode B: batch ZIP

Upload exactly one `.zip` when prompted.

Minimum ZIP contents:

```text
my_batch.zip
└── any_folder/
    ├── video_001.mp4
    ├── video_002.mp4
    └── ...
```

Optional scoring file:

```text
my_batch.zip
└── labels.csv
```

`labels.csv` must have these columns:

```csv
filename,expected_label
video_001.mp4,all
video_002.mp4,before
```

Rules:

- `filename` must match the video filename inside the ZIP.
- `expected_label` must be in the same Top-50 label space as the active model.
- If `labels.csv` is omitted, the notebook still predicts but cannot score accuracy.

The notebook writes batch results to:

```text
/content/drive/MyDrive/asl/notebook12_wlasl_top50_batch_results.jsonl
```

Current caveat: the output filename says `top50`; for the known validated WLASL run the labels were specifically aligned to the Zahid/model label space. Rename the path in the notebook if you want clearer run names.

## Notebook 13 — Hugging Face model evaluation

File:

```text
notebooks/13_colab_hf_model_eval_top50.ipynb
```

Purpose:

- Compare Hugging Face adapters under the same evaluation contract.
- Scores Zahid and WLASL Top-50 prepared bundles.

Default adapters compared:

```text
AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora
AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora
```

Upload/prepare these Drive files:

```text
/content/drive/MyDrive/asl/phase1_zahid_top50_bundle.zip
/content/drive/MyDrive/asl/video_finetune/top50_bundle.zip
/content/drive/MyDrive/asl/video_finetune/top50/labels.txt
```

Fallback WLASL path if `top50_bundle.zip` is absent:

```text
/content/drive/MyDrive/asl/wlasl_top50_30x448_bundle.zip
```

Expected bundle contents:

```text
bundle.zip
├── train.jsonl / val.jsonl / test.jsonl or relevant split manifests
└── frames_30x448/ or frames/
    └── <split>/<sample_id>/frame_000.jpg ... frame_029.jpg
```

Run order:

1. Install/setup cells.
2. Config cell.
3. Drive mount and label-load cell.
4. ZIP extraction/resolution cells.
5. Split-building cells.
6. Model evaluation cells.
7. Summary/export/visualization cells.

Outputs are written under the notebook work directory, for example:

```text
/content/asl_gemma4_26b_model_comparison/model_comparison_11_vs_14.json
```

## Notebook 14 — Colab Gradio demo

File:

```text
notebooks/14_colab_gradio_asl_demo.ipynb
```

Purpose:

- Launch a temporary Gradio UI from Colab.
- Accept a raw uploaded ASL video.
- Sample frames in memory and return a strict Top-50 prediction.

Default adapter:

```text
AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora
```

Upload/prepare:

- No dataset ZIP is required.
- You only need one short `.mp4`, `.mov`, `.webm`, or `.m4v` ASL video for the UI.
- Set `HF_TOKEN` or log in if the model download requires it.

Run order:

1. Runtime setup.
2. Config.
3. Shared helper definitions.
4. Model load.
5. Prediction function.
6. Launch Gradio.
7. Open the Gradio link, upload one video, and click Translate.

The Gradio demo does not save extracted frame folders. It samples the uploaded video in memory.

## How to create a Notebook 12 batch ZIP

A simple valid batch ZIP can look like this:

```text
batch_eval.zip
├── labels.csv
├── 001_all_01912.mp4
├── 002_before_05724.mp4
└── 003_better_06062.mp4
```

Example `labels.csv`:

```csv
filename,expected_label
001_all_01912.mp4,all
002_before_05724.mp4,before
003_better_06062.mp4,better
```

Then in Notebook 12 Mode B:

1. Run all setup/model cells through the shared model load.
2. Run the batch upload cell.
3. Upload `batch_eval.zip`.
4. Let the batch loop finish.
5. Check the JSONL output in Drive.

## Troubleshooting

### Missing Drive file

If a notebook raises `FileNotFoundError`, check the Drive path printed in the config cell. Either move the file to the expected path or edit the config variable.

### Bad or low accuracy in batch mode

First check label-space alignment. The expected labels in `labels.csv` must match the model's allowed labels exactly after normalization.

### Out-of-allowlist prediction

This means the model generated text, but the normalized candidate was not in the approved Top-50 list. The notebooks should show the candidate and rejection reason instead of hiding it behind a vague `unknown` result.

### CUDA or VRAM errors

Use a larger Colab GPU runtime, reduce the number of frames only for experiments, or switch to a smaller model intentionally. Do not silently change frame count for reported runs.

### Hugging Face download timeout

Keep these environment defaults where present:

```python
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
```
