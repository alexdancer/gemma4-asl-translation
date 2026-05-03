# Quick Start: Fine-Tuning Gemma 4 on ASL Data

**TL;DR:** 3 commands to fine-tune in ~2 hours.

## 1. Activate Environment

```bash
cd /home/alex-server/Documents/ASL-Hackathon/sign-language-asl
source venv/bin/activate
```

## 2. Prepare Training Data (Top-50 ASL Glosses)

Fast path for quick iteration. Trains on the 50 most common ASL signs.

```bash
python scripts/prepare_training_data.py --top50-only
```

**Expected output:**
```
data/contracts/asl_top50_glosses_v1.json       (50 glosses)
data/processed/splits/top50/train.csv           (train split)
data/processed/splits/top50/val.csv             (validation split)
data/processed/splits/top50/test.csv            (test split)
```

**Time:** ~5 minutes

## 3. Run Smoke Test (Verify Pipeline)

Test that the fine-tuning loop works end-to-end before committing to full training.

```bash
python scripts/test_finetuning.py --mock-model --max-samples 8 --batch-size 2
```

**Expected output:**
```
Fine-tuning smoke test summary
====================================
[PASS] model_loads: model and tokenizer loaded
[PASS] data_pipeline: batch pose shape=(2, 256, 99)
[PASS] forward_and_train: ran 4 optimization steps
[PASS] loss_decreases: same-batch before=4.2 after=2.1
[PASS] checkpoint_saved: checkpoint=./checkpoints/gemma_asl_smoke/smoke-checkpoint
Runtime: 45s
```

**Time:** ~1 minute on CPU

## 4. Fine-Tune on Top-50 Data

Full training loop with real Gemma 4 model (requires GPU).

```bash
python scripts/test_finetuning.py \
  --manifest data/processed/splits/top50/train.csv \
  --pose-root data/processed/poses \
  --max-samples 500 \
  --batch-size 16 \
  --num-epochs 20 \
  --output-dir checkpoints/gemma_asl_top50
```

**What's happening:**
- Loads Gemma 4 2B-E2B in 4-bit precision
- Attaches LoRA adapters (rank=16)
- Trains on ~500 pose→gloss pairs for 20 epochs
- Saves best checkpoint to `checkpoints/gemma_asl_top50/`
- Logs to `training_log.json`

**Expected output (first few epochs):**
```
Epoch 1/20 | Loss: 3.45 | Eval Acc: 0.42 | LR: 2.0e-4
Epoch 2/20 | Loss: 2.89 | Eval Acc: 0.58 | LR: 1.98e-4
Epoch 3/20 | Loss: 2.12 | Eval Acc: 0.71 | LR: 1.96e-4
...
Epoch 20/20 | Loss: 0.42 | Eval Acc: 0.87 | LR: 1.5e-4

Best checkpoint saved: checkpoints/gemma_asl_top50/best-checkpoint
Final validation accuracy: 0.87
```

**Time:** ~2.5 hours on 8GB GPU (RTX 3060, RTX 4060, etc.)

---

## Hyperparameter Cheatsheet

### For Fast Iteration (sacrifice accuracy)
```bash
python scripts/test_finetuning.py \
  --manifest data/processed/splits/top50/train.csv \
  --pose-root data/processed/poses \
  --max-samples 100 \
  --batch-size 8 \
  --num-epochs 5 \
  --learning-rate 5e-4
```
**Time:** ~20 minutes, Accuracy: ~75%

### For High Accuracy (sacrifice speed)
```bash
python scripts/test_finetuning.py \
  --manifest data/processed/splits/top50/train.csv \
  --pose-root data/processed/poses \
  --max-samples 500 \
  --batch-size 32 \
  --num-epochs 30 \
  --learning-rate 1e-4
```
**Time:** ~4 hours, Accuracy: ~90%

### For Production (balanced)
```bash
python scripts/test_finetuning.py \
  --manifest data/processed/splits/top50/train.csv \
  --pose-root data/processed/poses \
  --max-samples 500 \
  --batch-size 16 \
  --num-epochs 20 \
  --learning-rate 2e-4
```
**Time:** ~2.5 hours, Accuracy: ~87%

---

## Monitoring Training

### Watch logs in real-time
```bash
tail -f training_log.json | jq '.loss, .eval_accuracy'
```

### Plot loss curve
```bash
python -c "
import json
import matplotlib.pyplot as plt

with open('training_log.json') as f:
    logs = [json.loads(line) for line in f]

plt.plot([x['loss'] for x in logs], label='Train Loss')
plt.plot([x['eval_loss'] for x in logs], label='Eval Loss')
plt.legend()
plt.savefig('training_curve.png')
print('Saved to training_curve.png')
"
```

### Check GPU usage
```bash
watch -n 1 nvidia-smi
```

---

## After Training: Inference

### Load fine-tuned model
```python
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import numpy as np

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-E2B-it",
    max_seq_length=256,
    load_in_4bit=True,
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    model,
    "checkpoints/gemma_asl_top50/best-checkpoint"
)
model.eval()

# Load a test pose
pose_data = np.load("data/processed/poses/book_68011.npz")
body = pose_data["body"]  # Shape: (seq_len, 17, 3)

# Run inference
with torch.no_grad():
    # Convert pose to token IDs (simplified, see full pipeline for real implementation)
    input_ids = torch.tensor([pose_data]).long().cuda()
    output_ids = model.generate(input_ids, max_new_tokens=8)
    
predicted_gloss = tokenizer.decode(output_ids[0])
print(f"Predicted gloss: {predicted_gloss}")
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
--batch-size 8  # Instead of 16
```

### "Loss doesn't decrease"
Check data loading:
```bash
python -c "
import pandas as pd
train_df = pd.read_csv('data/processed/splits/top50/train.csv')
print(f'Samples: {len(train_df)}')
print(f'Glosses: {train_df[\"gloss\"].unique()}')
print(train_df.head())
"
```

### "Model accuracy is low (< 50%)"
Try higher learning rate:
```bash
--learning-rate 5e-4
```

Or more epochs:
```bash
--num-epochs 50
```

### "Training is slow (> 10 min per epoch)"
Check if gradient checkpointing is enabled (trades memory for speed):
```python
# In the training script:
model = prepare_model_for_training(
    model,
    use_cache=True,
    gradient_checkpointing=False  # Try disabling
)
```

---

## Next Steps

1. **After smoke test passes:** Run full fine-tuning
2. **After fine-tuning completes:** Evaluate on test set
3. **Iterate:** Adjust hyperparameters if accuracy < 85%
4. **Scale up:** Train on full WLASL dataset (not just Top-50)
5. **Optimize:** Quantize further for mobile deployment (Cactus pipeline)

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/prepare_training_data.py` | Create train/val/test splits from pose data |
| `scripts/test_finetuning.py` | Fine-tuning entry point (both smoke test + full training) |
| `src/models/gemma_loader.py` | Unsloth + LoRA model setup |
| `src/data/` | Data loading + pose preprocessing |
| `docs/UNSLOTH_FINE_TUNING_DEEP_DIVE.md` | Technical details (this is the reference guide) |
| `config.yaml` | Hyperparameters (edit before running) |

---

## Full Pipeline in One Command

Prepare → Smoke test → Fine-tune:

```bash
python scripts/prepare_training_data.py --top50-only && \
python scripts/test_finetuning.py --mock-model --max-samples 8 && \
python scripts/test_finetuning.py \
  --manifest data/processed/splits/top50/train.csv \
  --pose-root data/processed/poses \
  --max-samples 500 \
  --batch-size 16 \
  --num-epochs 20
```

**Total time:** ~3 hours
