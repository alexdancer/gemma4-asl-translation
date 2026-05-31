# ASL Top-50 Colab + Gradio Project

This repository contains a Colab-first ASL recognition prototype for a fixed **Top-50 ASL gloss vocabulary**. It is designed around Gemma-4 vision-language inference/fine-tuning, notebook-based evaluation, and a lightweight Gradio demo.

The current project is **not** a mobile app. The legacy React Native, Cactus, iOS, and mobile tracer code has been removed from this branch.

## What the project does

The project takes a short ASL video, samples a fixed set of frames, asks a Gemma-4 vision model to identify the sign, and accepts the answer only if it is one of the approved Top-50 labels.

At a high level:

1. A video is uploaded in Colab or Gradio.
2. The runtime samples **30 evenly spaced RGB frames** from the video.
3. Frames are resized to **448x448**.
4. The frames are passed to a Gemma-4 vision model through the Unsloth `FastVisionModel` path.
5. The model is prompted to return exactly one gloss from the approved list.
6. The raw model output is normalized and checked against the strict Top-50 allowlist.
7. The notebook/app returns the accepted prediction, or shows the rejected candidate and reason.

This strict allowlist behavior is intentional. It prevents vague, free-form, or out-of-vocabulary answers from being treated as valid ASL predictions.

## Main ways to use it

### 1. Single-video inference

Use:

```text
notebooks/12_colab_user_video_inference_top50.ipynb
```

Upload one `.mp4`, `.mov`, or `.m4v` clip. The notebook extracts 30 frames in the Colab runtime, runs the model, logs the result to Drive, and deletes the raw uploaded video from the runtime after logging.

### 2. Batch ZIP inference

Use the same notebook:

```text
notebooks/12_colab_user_video_inference_top50.ipynb
```

Upload one `.zip` containing video files. Optionally include a `labels.csv` file with expected labels for scoring.

Required `labels.csv` columns if labels are included:

```csv
filename,expected_label
001_all_01912.mp4,all
002_before_05724.mp4,before
```

The expected labels must match the active model label space. A previous WLASL batch looked bad until the labels were aligned to the Zahid/model Top-50 vocabulary.

### 3. Model comparison/evaluation

Use:

```text
notebooks/13_colab_hf_model_eval_top50.ipynb
```

This compares Hugging Face adapters on Zahid and WLASL Top-50 bundles using the same frame/prompt/normalization contract.

### 4. Gradio demo

Use either:

```text
notebooks/14_colab_gradio_asl_demo.ipynb
```

or the Hugging Face Space source:

```text
spaces/asl-gradio-cloud-demo/
```

The Gradio app accepts a raw uploaded video, samples frames in memory, runs the model, and returns a strict Top-50 prediction plus debug JSON.

## Current notebooks

| Notebook | Purpose | Main input |
|---|---|---|
| `notebooks/11_colab_gemma4_26b_zahid_phase1_pretrain.ipynb` | Training/pretraining notebook for the Zahid Top-50 adapter. | `phase1_zahid_top50_bundle.zip` in Drive. |
| `notebooks/12_colab_user_video_inference_top50.ipynb` | Main user-video inference notebook. | One raw video, or one batch ZIP with optional `labels.csv`. |
| `notebooks/13_colab_hf_model_eval_top50.ipynb` | Hugging Face adapter comparison/evaluation. | Zahid/WLASL prepared frame bundles in Drive. |
| `notebooks/14_colab_gradio_asl_demo.ipynb` | Colab-hosted Gradio demo. | One uploaded raw video. |

See `notebooks/README.md` for exact upload files, Drive paths, and run order.

## Setup: prepare the ZIP files used by the notebooks

The notebooks do not expect raw local datasets to be committed to git. They expect uploadable ZIPs in Google Drive. Use the scripts below to create those ZIP inputs before running the Colab notebooks.

### Zahid Phase-1 training ZIP for Notebook 11

Notebook 11 expects this file in Drive:

```text
/content/drive/MyDrive/asl/phase1_zahid_top50_bundle.zip
```

Build it locally in three steps:

```bash
# 1) Download videos referenced by the Zahid train/val/test manifests.
python scripts/download_phase1_videos.py \
  --manifest-dir data/phase1_zahid_top50 \
  --out-dir data/phase1_zahid_top50/videos_cache \
  --repo ZahidYasinMittha/American-Sign-Language-Dataset \
  --revision 3a6226f9c8de394a07b6c2e01158f6291897f97b

# 2) Extract 30x448 frames from the downloaded videos.
python scripts/extract_phase1_frames_and_zip.py \
  --manifest-dir data/phase1_zahid_top50 \
  --videos-dir data/phase1_zahid_top50/videos_cache \
  --frames-dir data/phase1_zahid_top50/frames_30x448 \
  --num-frames 30 \
  --size 448

# 3) Create the Colab bundle ZIP with manifests + frames.
python - <<'PY'
from pathlib import Path
import zipfile
root = Path('data/phase1_zahid_top50')
out = Path('data/phase1_zahid_top50_bundle.zip')
include = [root / 'train.jsonl', root / 'val.jsonl', root / 'test.jsonl', root / 'frames_30x448']
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for item in include:
        if item.is_file():
            zf.write(item, item.relative_to(root))
        else:
            for path in sorted(item.rglob('*')):
                if path.is_file():
                    zf.write(path, path.relative_to(root))
print(out)
PY
```

Upload the resulting ZIP to:

```text
/content/drive/MyDrive/asl/phase1_zahid_top50_bundle.zip
```

The ZIP must contain `train.jsonl`, `val.jsonl`, `test.jsonl`, and deterministic frame folders with `frame_000.jpg` through `frame_029.jpg` for each sample.

### WLASL Top-50 prepared bundle for Notebook 13

Notebook 13 expects a prepared WLASL frame bundle at:

```text
/content/drive/MyDrive/asl/video_finetune/top50_bundle.zip
```

Prepare WLASL frames locally from the local WLASL checkout/data directory:

```bash
python scripts/data/extract_wlasl_video_frames.py \
  --wlasl-root data/WLASL \
  --output-root data/video_finetune \
  --datasets both \
  --num-frames 30 \
  --image-size 448 \
  --top-k 50
```

Then ZIP the generated Top-50 folder:

```bash
python - <<'PY'
from pathlib import Path
import zipfile
src = Path('data/video_finetune/top50')
out = Path('data/video_finetune/top50_bundle.zip')
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(src.rglob('*')):
        if path.is_file():
            zf.write(path, path.relative_to(src.parent))
print(out)
PY
```

Upload it to:

```text
/content/drive/MyDrive/asl/video_finetune/top50_bundle.zip
```

Also upload/copy the generated label file to:

```text
/content/drive/MyDrive/asl/video_finetune/top50/labels.txt
```

### Batch ZIP for Notebook 12

Notebook 12 batch mode expects one user-created ZIP of raw videos. This ZIP is different from the prepared training/eval frame bundles.

Example:

```text
batch_eval.zip
├── labels.csv
├── 001_all_01912.mp4
├── 002_before_05724.mp4
└── 003_better_06062.mp4
```

`labels.csv` is optional, but if provided it must have:

```csv
filename,expected_label
001_all_01912.mp4,all
002_before_05724.mp4,before
003_better_06062.mp4,better
```

Upload this ZIP directly in Notebook 12 Mode B when prompted. The notebook extracts frames itself at runtime.

## Current model defaults

Notebook 11 / Notebook 12 use:

- Base model: `unsloth/gemma-4-26B-A4B-it`
- Notebook 11/12 adapter: `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`

Notebook 13 compares:

- `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`
- `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`

Notebook 14 and the Gradio Space default to:

- `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`

Do not substitute the Google base model for Notebook 11/12 without deliberately updating the notebook load path and verification cells.

## Repository layout

| Path | Role |
|---|---|
| `notebooks/` | Colab notebooks and notebook run guide. |
| `spaces/asl-gradio-cloud-demo/` | Hugging Face Gradio Space app. |
| `src/data/` | Data loading, frame/pose contracts, and dataset utilities. |
| `src/evaluation/` | Notebook/evaluation parsing, normalization, scoring, and gatekeeping helpers. |
| `src/demo/` | Demo/reference-output helpers. |
| `src/shared/` | Small shared contracts used by runtime tests. |
| `tests/` | Unit/contract tests for data, notebooks, runtime helpers, Gradio app behavior, and evaluation utilities. |
| `docs/` | Human-readable project notes and technical documentation. |

## Data and artifact policy

Large/generated files are intentionally not tracked:

- `data/WLASL/`
- `data/video_finetune/`
- `data/*.zip`
- `checkpoints/`
- `notebooks/output/`
- root `results/`
- generated runtime proof artifacts under `artifacts/`

Store large datasets, upload ZIPs, checkpoints, and run logs in Google Drive, Hugging Face, or ignored local paths. Keep only reusable code, notebooks, documentation, configs, and tests in git.

## Useful local commands

```bash
# Run the repository test suite
npm run test

# Compile-check Python source/tests/scripts
npm run typecheck
```

Data preparation helpers:

```bash
python scripts/data/extract_wlasl_video_frames.py --help
python scripts/extract_phase1_frames_and_zip.py --help
python scripts/download_phase1_videos.py --help
```

Evaluation helpers:

```bash
python scripts/evaluation/evaluate_colab_gatekeeper.py --help
python scripts/evaluation/run_multiword_eval_pipeline.py --help
```

## Documentation

- `notebooks/README.md` — what to upload for each notebook and how to run it.
- `docs/technical-project-overview.md` — technical architecture, data contracts, inference path, and testing strategy.
- `docs/notebook12-wlasl-zahid-labelset-batch-results.md` — Notebook 12 WLASL batch result and label-space caveat.
- `docs/wlasl-i3d-reference.md` — WLASL/I3D reference notes.
- `spaces/asl-gradio-cloud-demo/README.md` — Gradio Space runtime notes.

## Scope guardrails

- Notebook 12 is the main inference surface.
- Notebook 13 is the main lightweight evaluation surface.
- Notebook 14 / the Gradio Space are demo surfaces, not production apps.
- Keep the Top-50 allowlist and label normalization consistent across notebooks, Gradio, and tests.
- Treat WLASL batch scores as out-of-domain evidence unless the model has been trained on the corresponding WLASL split.
- Do not commit generated checkpoints, ZIPs, run logs, notebook output artifacts, or raw uploaded videos.
