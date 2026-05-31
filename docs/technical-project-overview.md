# Technical Project Overview

This document describes the current ASL Top-50 project architecture after the repository cleanup. It focuses on the active Colab, evaluation, and Gradio workflow.

## Project scope

The project is a **notebook-first ASL isolated-sign recognition prototype**. It uses Gemma-4 vision-language inference with strict output validation against a canonical Top-50 gloss list.

Current active surfaces:

- Colab training/pretraining notebook.
- Colab single-video inference notebook.
- Colab batch evaluation notebook.
- Colab/Hugging Face Gradio demo.
- Local Python tests that enforce the notebook and runtime contracts.

Removed/out of scope on this branch:

- React Native app.
- Cactus runtime/hybrid service.
- iOS/mobile tracer code.
- Mobile export artifacts and mobile-specific tests.

## Core inference contract

The central contract is the same across Notebook 12, Notebook 13, Notebook 14, and the Gradio Space:

| Field | Value |
|---|---|
| Input media | Short ASL video, or prepared frame bundle for eval/training. |
| Frame count | 30 frames. |
| Frame size | 448x448. |
| Color mode | RGB. |
| Sampling | Even/deterministic across the clip or precomputed manifest. |
| Prompt | Ask for exactly one ASL gloss from the approved labels. |
| Generation | Deterministic, low `max_new_tokens`. |
| Output validation | Normalize first candidate and require membership in Top-50 allowlist. |

The model may produce arbitrary text. The project does **not** trust raw generation directly. Every prediction is normalized and checked against the allowlist before it is accepted.

## Why videos become frames

Gemma-4/Unsloth FastVision inference in this project consumes image inputs through a chat template. The notebooks and Gradio app therefore convert a video into a fixed list of image frames before calling the model.

This conversion happens in two ways:

1. **Prepared datasets**: training/evaluation bundles contain deterministic frame folders such as `frames_30x448/<split>/<sample_id>/frame_000.jpg` through `frame_029.jpg`.
2. **User uploads**: Notebook 12 and Gradio accept raw videos, sample frames in the runtime, and pass those images directly to the model.

The Gradio app accepts a raw video from the user, but internally it still samples frames because the active model path is image-frame based rather than native-video based.

## Label-space contract

The project uses a fixed Top-50 gloss vocabulary. The exact active list appears in the notebooks and Gradio app. The important technical rules are:

- Train, eval, batch `labels.csv`, notebook prompts, and Gradio validation must use the same label space.
- Labels are normalized before comparison.
- Aliases may be mapped before validation when explicitly configured.
- Out-of-list generations are rejected and surfaced with a reason.

This matters because a previous WLASL batch result looked bad when expected labels came from the wrong Top-50 vocabulary. After aligning expected labels to the model label space, Notebook 12 reached 33/50 correct on the validated out-of-domain batch.

## Main model artifacts

### Base model

```text
unsloth/gemma-4-26B-A4B-it
```

### Zahid-pretrained adapter

```text
AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora
```

Used by:

- Notebook 11 training/pretraining.
- Notebook 12 default inference.
- Notebook 13 as one comparison target.

### Zahid + WLASL combined Top-50 adapter

```text
AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora
```

Used by:

- Notebook 13 as one comparison target.
- Notebook 14 Gradio demo.
- `spaces/asl-gradio-cloud-demo/app.py` by default.

## Notebook architecture

### Notebook 11

Training/pretraining path:

1. Mount Drive.
2. Load `phase1_zahid_top50_bundle.zip`.
3. Resolve manifests and `frames_30x448` folder.
4. Build multimodal chat samples from frame paths.
5. Load `unsloth/gemma-4-26B-A4B-it` with Unsloth FastVision.
6. Train LoRA adapter.
7. Save adapter locally/Drive and optionally push to Hugging Face.
8. Run lightweight eval/metrics upload.

### Notebook 12

User/batch inference path:

1. Define shared config and helper functions.
2. Mount Drive.
3. Mode A: upload one raw video, or Mode B: upload one batch ZIP.
4. Extract 30 evenly sampled 448x448 frames from each video.
5. Load adapter snapshot from Hugging Face.
6. Run deterministic generation.
7. Normalize and validate the model output.
8. Append JSONL result rows to Drive.
9. Delete raw uploaded video/ZIP from the runtime after logging.

### Notebook 13

Evaluation path:

1. Mount Drive.
2. Load Top-50 labels.
3. Extract Zahid/WLASL prepared bundles.
4. Resolve frame roots and split manifests.
5. Convert rows to model chat samples.
6. Evaluate each configured adapter.
7. Emit metrics and comparison artifacts under the Colab work directory.

### Notebook 14

Colab Gradio path:

1. Install/runtime setup.
2. Configure adapter and frame contract.
3. Define video sampling, prompt, normalization, and validation helpers.
4. Load model once.
5. Launch a Gradio UI.
6. On each upload, sample frames in memory and return prediction/debug JSON.

## Gradio Space architecture

Path:

```text
spaces/asl-gradio-cloud-demo/app.py
```

Startup behavior:

1. Set Hugging Face/Unsloth environment defaults.
2. Resolve `ASL_ADAPTER_MODEL_ID`, defaulting to the combined Top-50 adapter.
3. Optionally eager-load on startup via `ASL_EAGER_LOAD_ON_STARTUP`.
4. `snapshot_download` the adapter.
5. Load with `FastVisionModel.from_pretrained(..., load_in_4bit=True, max_seq_length=8192)`.
6. Store model/tokenizer/diagnostics in a module-global bundle.

Request behavior:

1. Validate uploaded video path and extension.
2. Sample `ASL_FRAME_COUNT` frames at `ASL_FRAME_SIZE`.
3. Build the approved-label prompt.
4. Apply tokenizer chat template with image inputs.
5. Generate with deterministic settings.
6. Normalize first-line output.
7. Return accepted prediction or out-of-allowlist diagnostics.

Failure behavior:

- Missing video: fail closed with an explicit UI error.
- Unsupported extension: fail closed.
- Model load failure: show diagnostics instead of fake predictions.
- Out-of-list candidate: return the candidate and rejection reason.

## Local source modules

| Path | Responsibility |
|---|---|
| `src/notebook12_user_video.py` | Notebook 12 behavior contract helpers. |
| `src/cloud_translate_api.py` | Cloud-style translation API contract helpers and response shaping. |
| `src/data/` | Data loading, pose/frame cache contracts, synthetic phrase utilities. |
| `src/evaluation/colab_unsloth_inference.py` | Colab inference output parsing and normalization helpers. |
| `src/evaluation/colab_gatekeeper.py` | Gatekeeper checks for notebook/eval outputs. |
| `src/evaluation/colab_anchor_contract.py` | Anchor/sample contract checks. |
| `src/evaluation/phase2a.py` | Phase 2A report and deterministic label-prior baseline. |
| `src/demo/` | Demo and reference-output utilities. |
| `src/shared/` | Shared runtime contracts. |

## Tests and quality gates

Run:

```bash
npm run typecheck
npm run test
```

Current test categories:

| Test area | Purpose |
|---|---|
| `tests/data/` | Data loading, frame extraction, pose/q64 contracts, synthetic phrase generation. |
| `tests/demo/` | Demo output contracts, replay/fallback behavior, prompt-control smoke checks. |
| `tests/evaluation/` | Colab gatekeeping, model evaluation, multiword evaluation, report generation. |
| `tests/runtime/` | Runtime API contracts, frame extraction, live capture, telemetry, video ingest. |
| `tests/spaces/` | Gradio Space app contract tests. |
| `tests/training/` | Training utility contracts. |

The tests do not run full Gemma inference locally. They enforce data shapes, prompt/output contracts, normalization behavior, and application-level fail-closed semantics.

## Data formats

### Prepared frame bundle

Expected shape for Notebook 11/13 style data:

```text
bundle.zip
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── frames_30x448/
    ├── train/<sample_id>/frame_000.jpg ... frame_029.jpg
    ├── val/<sample_id>/frame_000.jpg ... frame_029.jpg
    └── test/<sample_id>/frame_000.jpg ... frame_029.jpg
```

Manifests should include enough information to map each row to:

- sample ID
- label/gloss
- split
- frame paths or a resolvable frame folder

### Notebook 12 batch ZIP

Expected shape:

```text
batch.zip
├── labels.csv              # optional
├── video_001.mp4
├── video_002.mp4
└── nested/video_003.mov    # nesting is allowed
```

`labels.csv` shape:

```csv
filename,expected_label
video_001.mp4,all
video_002.mp4,before
```

## Generated outputs

Important outputs are generally written outside git-tracked paths:

- Notebook 12 single-video JSONL: `/content/drive/MyDrive/asl/notebook12_user_video_results.jsonl`
- Notebook 12 batch JSONL: `/content/drive/MyDrive/asl/notebook12_wlasl_top50_batch_results.jsonl`
- Notebook 13 comparison JSON: `/content/asl_gemma4_26b_model_comparison/model_comparison_11_vs_14.json`
- Gradio Space debug JSON: returned in the UI, not committed.

Do not commit raw videos, checkpoints, extracted frames, JSONL run logs, or notebook output artifacts.

## Operational assumptions

- Hugging Face is the model registry.
- Google Drive is the notebook data/output staging area.
- Colab provides the GPU runtime.
- Local repository tests are for contracts and regressions, not full model execution.
- The project prioritizes reproducible label-space and frame-shape contracts over free-form chatbot behavior.

## Extension points

Reasonable future extensions:

- Add a confusion-matrix helper for Notebook 12 batch JSONL logs.
- Add a smaller-model notebook variant for cheaper GPUs.
- Convert Notebook 12 output filenames from generic `top50` names to explicit model-labelset names.
- Add a native-video model path only if the model/processor truly supports raw video and the training/eval contracts are updated accordingly.

When extending the project, keep the following synchronized:

1. The Top-50 allowlist.
2. Label normalization and aliases.
3. Frame count and resolution.
4. Prompt format.
5. Notebook docs and tests.
