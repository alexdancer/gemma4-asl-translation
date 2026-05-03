# Unsloth Top-50 Dashboard Run Checklist

Purpose: run the first proof-of-learning Top-50 ASL fine-tuning experiment in Unsloth Dashboard using the compact `q64_full` encoding.

## Goal

Validate that Gemma4 E4B IT can learn ASL pose-to-gloss mapping on a clean Top-50 held-out split.

Success ladder:

- **Go:** test accuracy >= 70%
- **Strong go:** test accuracy >= 80%
- **Yellow:** 40-70%, tune training/encoding/prompt
- **No-go/debug:** < 40%, investigate before scaling or demo integration

## Files

### Training dataset

Upload this as the dashboard training dataset:

```text
/home/alex-server/Documents/ASL-Hackathon/sign-language-asl/data/processed/exports/asl_unsloth_pose_train_q64_full_top50_train.jsonl
```

### Eval dataset

Upload this as the dashboard eval/validation dataset if the dashboard supports eval upload:

```text
/home/alex-server/Documents/ASL-Hackathon/sign-language-asl/data/processed/exports/asl_unsloth_pose_train_q64_full_top50_val.jsonl
```

### Held-out test dataset — do not train on this

Keep this file untouched for final local evaluation:

```text
/home/alex-server/Documents/ASL-Hackathon/sign-language-asl/data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl
```

### Manifest

```text
/home/alex-server/Documents/ASL-Hackathon/sign-language-asl/data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json
```

## Dashboard settings

| Field | Value |
|---|---|
| Base model | `unsloth/gemma-4-E4B-it` |
| Method | `QLoRA / 4-bit` |
| Context length | `2048` |
| Epochs | `3` |
| Learning rate | `0.0002` |
| Batch size | `1` |
| Optimizer | `adamw_8bit` |
| Gradient accumulation | `4` if available |
| Save strategy | every epoch or every ~100 steps |

If dashboard asks for max steps instead of epochs:

- Train file has 200 rows.
- Batch size 1 for 3 epochs is roughly 600 sample steps.
- If gradient accumulation changes displayed optimizer steps, fewer optimizer steps is expected.

## What to watch

Good signs:

- train loss trends down
- eval/validation loss trends down or stays stable

Overfitting signs:

- train loss keeps dropping
- eval/validation loss rises repeatedly

If overfitting is obvious, stop and keep the best checkpoint.

## After training

Send/record:

- screenshot of final train/eval loss
- output/checkpoint folder name
- final train loss
- final eval loss, if shown

Then freeze the checkpoint into the project before evaluation.

Expected project checkpoint destination pattern:

```text
/home/alex-server/Documents/ASL-Hackathon/sign-language-asl/checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline
```

## Local evaluation command after checkpoint is frozen

```bash
python scripts/evaluate_unsloth_asl.py \
  --checkpoint checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline \
  --test-file data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --out-dir evaluation/results/unsloth_top50_q64_full_dashboard_baseline
```

Evaluation outputs:

- `predictions.csv`
- `metrics.json`

Primary metric:

```text
strict_normalized_top1_accuracy
```

Secondary metrics:

- invalid output rate
- per-class accuracy
- confusion matrix counts
