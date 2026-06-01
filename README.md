# Gemma4 ASL Translation Model

Colab-first ASL recognition prototype for a fixed Top-50 ASL gloss set. It samples video frames, runs Gemma-4 vision inference through Unsloth, and accepts only labels in the Top-50 allowlist.

**Gradio demo notebook:** [`notebooks/14_colab_gradio_asl_demo.ipynb`](https://github.com/alexdancer/gemma4-asl-translation/blob/master/notebooks/14_colab_gradio_asl_demo.ipynb)

## Links

- Model adapter: [AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora](https://huggingface.co/AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora)
- WLASL dataset: [dxli94/WLASL](https://github.com/dxli94/WLASL)
- Zahid ASL dataset: [ZahidYasinMittha/American-Sign-Language-Dataset](https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset)

## How it works

1. Upload a video in Colab.
2. Sample 30 evenly spaced RGB frames.
3. Resize frames to 448x448.
4. Run Gemma-4 vision inference with `FastVisionModel`.
5. Normalize the model output.
6. Accept the result only if it matches the Top-50 allowlist.

Out-of-vocabulary or vague outputs are rejected with the candidate and rejection reason shown.

## Notebooks


| Notebook                                                    | Purpose                                               | Input                                                            |
| ----------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| `notebooks/11_colab_gemma4_26b_zahid_phase1_pretrain.ipynb` | Train/pretrain the Zahid Top-50 adapter.              | `phase1_zahid_top50_bundle.zip`                                  |
| `notebooks/12_colab_user_video_inference_top50.ipynb`       | Main single-video and batch inference notebook.       | One raw video, or a ZIP of raw videos with optional `labels.csv` |
| `notebooks/13_colab_hf_model_eval_top50.ipynb`              | Compare Hugging Face adapters on Zahid/WLASL bundles. | Prepared Zahid/WLASL frame bundles                               |
| `notebooks/14_colab_gradio_asl_demo.ipynb`                  | Temporary Colab Gradio demo.                          | One raw uploaded video                                           |


See `notebooks/README.md` for exact Drive paths and run order.

Model evaluation metrics and comparison results are documented in `docs/`. Start with `docs/notebook12-wlasl-zahid-labelset-batch-results.md` for the Notebook 12 batch results.

## Model defaults

Notebook 11 and Notebook 12 use:

- Base model: `unsloth/gemma-4-26B-A4B-it`
- Adapter: `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`

Notebook 13 compares:

- `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`
- `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`

Notebook 14 defaults to:

- `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`

Do not swap in the Google base model for Notebook 11/12 without updating the notebook load path and verification cells.

## Batch ZIP format for Notebook 12

Notebook 12 batch mode expects raw videos, not prepared frame folders.

```text
batch_eval.zip
├── labels.csv
├── 001_all_01912.mp4
├── 002_before_05724.mp4
└── 003_better_06062.mp4
```

Optional `labels.csv`:

```csv
filename,expected_label
001_all_01912.mp4,all
002_before_05724.mp4,before
003_better_06062.mp4,better
```

Expected labels must match the active model label space.

## Prepare data bundles

### Zahid Phase-1 bundle for Notebook 11

```bash
python scripts/download_phase1_videos.py \
  --manifest-dir data/phase1_zahid_top50 \
  --out-dir data/phase1_zahid_top50/videos_cache \
  --repo ZahidYasinMittha/American-Sign-Language-Dataset \
  --revision 3a6226f9c8de394a07b6c2e01158f6291897f97b

python scripts/extract_phase1_frames_and_zip.py \
  --manifest-dir data/phase1_zahid_top50 \
  --videos-dir data/phase1_zahid_top50/videos_cache \
  --frames-dir data/phase1_zahid_top50/frames_30x448 \
  --num-frames 30 \
  --size 448
```

Create the bundle:

```bash
cd data/phase1_zahid_top50
zip -r ../phase1_zahid_top50_bundle.zip train.jsonl val.jsonl test.jsonl frames_30x448
```

Upload it to:

```text
/content/drive/MyDrive/asl/phase1_zahid_top50_bundle.zip
```

The ZIP must contain `train.jsonl`, `val.jsonl`, `test.jsonl`, and frame folders with `frame_000.jpg` through `frame_029.jpg`.

### WLASL Top-50 bundle for Notebook 13

```bash
python scripts/data/extract_wlasl_video_frames.py \
  --wlasl-root data/WLASL \
  --output-root data/video_finetune \
  --datasets both \
  --num-frames 30 \
  --image-size 448 \
  --top-k 50
```

Create the bundle:

```bash
cd data/video_finetune
zip -r top50_bundle.zip top50
```

Upload these files:

```text
/content/drive/MyDrive/asl/video_finetune/top50_bundle.zip
/content/drive/MyDrive/asl/video_finetune/top50/labels.txt
```

## Repository layout


| Path              | Role                                                      |
| ----------------- | --------------------------------------------------------- |
| `notebooks/`      | Colab notebooks and notebook run guide.                   |
| `src/data/`       | Data loading and frame/pose utilities.                    |
| `src/evaluation/` | Parsing, normalization, scoring, and gatekeeping helpers. |
| `src/demo/`       | Demo/reference-output helpers.                            |
| `tests/`          | Unit and contract tests.                                  |
| `docs/`           | Project notes, technical docs, and model evaluation metrics/results. |


## Local commands

```bash
npm run test
npm run typecheck

python scripts/data/extract_wlasl_video_frames.py --help
python scripts/extract_phase1_frames_and_zip.py --help
python scripts/download_phase1_videos.py --help
```

## Artifact policy

Do not commit generated data, checkpoints, run logs, or ZIP bundles. Keep these in Google Drive, Hugging Face, or ignored local paths.

Ignored examples:

- `data/WLASL/`
- `data/video_finetune/`
- `data/*.zip`
- `checkpoints/`
- `notebooks/output/`
- `results/`
- `artifacts/`

## Notes

- Notebook 12 is the main user-video path.
- Notebook 13 is the main evaluation path.
- Notebook 14 is a Colab demo with Gradio
- Keep the Top-50 allowlist and label normalization consistent across notebooks and tests.

