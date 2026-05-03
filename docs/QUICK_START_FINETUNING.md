# Quick Start (Current Top-50 Workflow)

## 1) Activate environment

```bash
cd /home/alex-server/Documents/ASL-Hackathon/sign-language-asl
source venv/bin/activate
```

## 2) Prepare Top-50 split artifacts

```bash
python scripts/prepare_training_data.py --top50-only
```

## 3) Run evaluator contract smoke test (no checkpoint required)

```bash
python scripts/evaluate_unsloth_asl.py \
  --mock \
  --test-file data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --out-dir evaluation/results/unsloth_asl_mock_smoke \
  --max-samples 50
```

## 4) Run real held-out evaluation (checkpoint required)

```bash
python scripts/evaluate_unsloth_asl.py \
  --checkpoint checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline \
  --test-file data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --out-dir evaluation/results/unsloth_top50_q64_full_dashboard_baseline
```

## 5) Optional diagnostics

### Constrained diagnostic

```bash
python scripts/evaluate_unsloth_asl_constrained.py
```

### Prompt-control diagnostic

```bash
python scripts/evaluate_unsloth_asl_prompt_control.py
```

## 6) Prerecorded known-good q64 demo path

```bash
python scripts/run_prerecorded_q64_demo.py \
  --checkpoint checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline \
  --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --record-id hearing_26986 \
  --out-dir evaluation/results/prerecorded_q64_demo
```

For dependency-light smoke checks:

```bash
python scripts/run_prerecorded_q64_demo.py \
  --mock \
  --checkpoint checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline \
  --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --record-id hearing_26986
```

## Verification loops

```bash
npm run test
npm run typecheck
```
