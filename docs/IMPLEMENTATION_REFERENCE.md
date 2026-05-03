# Implementation Reference: Gemma 4 Fine-Tuning with Unsloth

This guide walks through the actual code we use to fine-tune Gemma 4 on ASL poses. Cross-reference with `UNSLOTH_FINE_TUNING_DEEP_DIVE.md` for the technical theory.

---

## File Structure

```
src/
  data/
    create_splits.py         # Top-50 contract + split generation
    wlasl_loader.py          # Load WLASL metadata
    pose_loader.py           # Load NPZ pose archives
    
  models/
    gemma_loader.py          # Unsloth model loading + LoRA setup
    
scripts/
  prepare_training_data.py   # Create train/val/test splits
  test_finetuning.py         # Fine-tuning + smoke test entry point
  extract_poses_batch.py     # Pose extraction from videos (reference)

docs/
  UNSLOTH_FINE_TUNING_DEEP_DIVE.md     # Theory
  IMPLEMENTATION_REFERENCE.md           # You are here
  QUICK_START_FINETUNING.md            # Commands
```

---

## Part 1: Model Loading (Gemma + Unsloth + LoRA)

**File:** `src/models/gemma_loader.py`

### 1.1 Load Gemma in 4-Bit Precision

```python
from unsloth import FastLanguageModel
import torch

def load_gemma_unsloth(
    model_name: str = "google/gemma-4-E2B-it",
    max_seq_length: int = 256,
    load_in_4bit: bool = True,
) -> tuple:
    """Load Gemma 4 2B-E2B with Unsloth optimizations."""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,  # Mixed precision
        load_in_4bit=load_in_4bit,
    )
    
    return model, tokenizer
```

**What happens inside Unsloth:**

1. **Quantization Setup:**
   ```python
   # Unsloth configures bitsandbytes for 4-bit quantization
   bitsandbytes_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_quant_type="nf4",  # Normalized Float 4
       bnb_4bit_use_double_quant=True,
   )
   ```

2. **Model Loading:**
   ```python
   # Load Gemma from HuggingFace Hub
   model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-4-E2B-it",
       quantization_config=bitsandbytes_config,
       device_map="auto",  # Load to GPU if available
   )
   ```

3. **Kernel Optimization:**
   - Unsloth patches attention kernels (Flash Attention or Xformers)
   - Replaces default PyTorch kernels with faster versions
   - Patches forward/backward passes

4. **KV Cache Pre-allocation:**
   ```python
   # For inference optimization
   model.config.use_cache = True
   model._prepare_decoder_attention_mask(...)
   ```

**Result:** Model loaded in 4-bit, optimized kernels ready, ~2GB VRAM.

### 1.2 Attach LoRA Adapters

```python
from peft import LoraConfig, get_peft_model

def attach_lora(model, rank: int = 16, lora_alpha: int = 32):
    """Attach LoRA adapters to Gemma."""
    
    lora_config = LoraConfig(
        r=rank,                           # LoRA rank
        lora_alpha=lora_alpha,            # Scaling factor
        lora_dropout=0.05,                # Dropout in LoRA
        bias="none",                      # Don't train biases
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameter count
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model
```

**LoRA math in practice:**

For each attention layer's query projection:
```
Original weight: W_q ∈ ℝ^{2048 × 2048}
LoRA update:     ΔW_q = B_q @ A_q
                 where B_q ∈ ℝ^{2048 × 16}, A_q ∈ ℝ^{16 × 2048}

Forward pass:    output = (W_q + ΔW_q) × input
                 = W_q × input + B_q @ A_q × input

Backward:        Only compute gradients for B_q and A_q
                 W_q remains frozen (no gradient)
```

**Parameters:**
- `q_proj` (Query projection): 2048 × 2048 → 2048 × 16 + 16 × 2048 = 65K trainable
- `v_proj` (Value projection): Same, 65K trainable
- Total per layer: ~130K trainable
- 30 layers × 130K = ~3.9M trainable parameters
- Original model: 2B parameters
- Percentage: 3.9M / 2B = **0.195% trainable** ✓

**Expected output:**
```
Trainable: 3,941,376 / 2,000,000,000 (0.20%)
```

### 1.3 Prepare for Training

```python
def prepare_model_for_training(model, use_cache: bool = True):
    """Enable gradient checkpointing and training mode."""
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Keep KV cache for inference optimization
    model.config.use_cache = use_cache
    
    model.train()  # Set to training mode
    
    return model
```

**What this does:**

- **Gradient checkpointing:** Recompute activations during backward (saves VRAM)
- **Use cache:** Pre-allocate KV cache (speeds up inference)
- **Training mode:** Enable dropout, disable batch norm momentum updates, etc.

---

## Part 2: Data Loading & Preprocessing

**File:** `src/data/pose_loader.py`

### 2.1 Load Pose Archives (NPZ files)

```python
import numpy as np
from pathlib import Path
from typing import Tuple

def load_pose(pose_path: Path) -> np.ndarray:
    """Load a single pose archive (NPZ)."""
    
    with np.load(pose_path) as data:
        body = data["body"]          # Shape: (seq_len, 17, 3)
        left_hand = data["left_hand"]    # Shape: (seq_len, 21, 3)
        right_hand = data["right_hand"]  # Shape: (seq_len, 21, 3)
    
    # Concatenate landmarks: 17 + 21 + 21 = 59 joints × 3 = 177 values per frame
    pose_sequence = np.concatenate([body, left_hand, right_hand], axis=1)
    # Shape: (seq_len, 59, 3)
    
    return pose_sequence


def normalize_pose(pose: np.ndarray) -> np.ndarray:
    """Normalize pose to [-1, 1] range."""
    
    # Flatten to (seq_len, 177)
    seq_len = pose.shape[0]
    pose_flat = pose.reshape(seq_len, -1)
    
    # Normalize per frame (center at origin, scale to [-1, 1])
    pose_normalized = pose_flat.copy()
    for t in range(seq_len):
        frame = pose_flat[t]
        frame_min, frame_max = frame.min(), frame.max()
        if frame_max > frame_min:  # Avoid division by zero
            pose_normalized[t] = 2 * (frame - frame_min) / (frame_max - frame_min) - 1
    
    return pose_normalized
```

**Shape transformations:**

```
NPZ archive:
  body shape:       (T, 17, 3)  — 17 body keypoints × 3 coords
  left_hand shape:  (T, 21, 3)  — 21 left hand keypoints × 3 coords
  right_hand shape: (T, 21, 3)  — 21 right hand keypoints × 3 coords
  
After concatenation:
  (T, 59, 3)  — 59 total joints × 3 coordinates
  
After flattening:
  (T, 177)    — 177-dimensional feature vector per frame
```

### 2.2 Dataset Class for PyTorch

```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class PoseGlossDataset(Dataset):
    """Load pose-gloss pairs for training."""
    
    def __init__(
        self,
        manifest_csv: Path,
        pose_root: Path,
        tokenizer,
        max_seq_length: int = 256,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.pose_root = pose_root
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        
        # Load pose
        pose_path = self.pose_root / row["sample_id"] + ".npz"
        pose = load_pose(pose_path)  # Shape: (T, 177)
        pose = normalize_pose(pose)
        
        # Truncate or pad to max_seq_length
        if pose.shape[0] > self.max_seq_length:
            pose = pose[:self.max_seq_length]
        elif pose.shape[0] < self.max_seq_length:
            pad_len = self.max_seq_length - pose.shape[0]
            pose = np.pad(pose, ((0, pad_len), (0, 0)), mode='constant')
        
        # Convert to tensor
        pose_tensor = torch.tensor(pose, dtype=torch.float32)  # (256, 177)
        
        # Tokenize gloss (target label)
        gloss = row["gloss"]
        tokens = self.tokenizer(
            gloss,
            truncation=True,
            padding="max_length",
            max_length=8,
            return_tensors="pt",
        )
        
        return {
            "input_ids": pose_tensor.flatten(),  # (256*177,) = (45312,)
            "attention_mask": torch.ones(self.max_seq_length),
            "labels": tokens["input_ids"].squeeze(),
        }


# Create data loader
def create_data_loader(
    manifest_csv: Path,
    pose_root: Path,
    tokenizer,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    
    dataset = PoseGlossDataset(manifest_csv, pose_root, tokenizer)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Speed up GPU transfer
    )
    
    return loader
```

**Batch shape after collation:**

```
batch["input_ids"]:    (batch_size=16, 256*177) = (16, 45312)
batch["attention_mask"]: (16, 256)
batch["labels"]:       (16, 8)  — Target gloss tokens
```

---

## Part 3: Training Loop (Unsloth + HuggingFace Trainer)

**File:** `scripts/test_finetuning.py` (simplified excerpt)

### 3.1 Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output directories
    output_dir="checkpoints/gemma_asl_top50",
    overwrite_output_dir=True,
    
    # Learning rate schedule
    learning_rate=2e-4,         # LoRA-specific (high LR)
    warmup_steps=100,           # Gradual warmup
    lr_scheduler_type="cosine", # Cosine annealing
    
    # Batch size & gradient accumulation
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    
    # Gradient clipping
    max_grad_norm=1.0,
    
    # Training duration
    num_train_epochs=20,
    max_steps=-1,  # Use num_train_epochs instead
    
    # Checkpointing & logging
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    
    # Precision
    fp16=True,  # Mixed precision (float16)
    half_precision_backend="auto",
    
    # Optimization
    optim="adamw_8bit",  # 8-bit Adam (memory efficient)
    weight_decay=0.01,
    
    # Early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,  # Keep only 3 checkpoints
    
    # Hardware
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Reporting
    report_to=["tensorboard"],
    logging_dir="logs",
)
```

**Why these values?**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `learning_rate` | 2e-4 | LoRA can use 5-10x higher LR than full FT. We use middle ground. |
| `warmup_steps` | 100 | Ramp up learning rate smoothly for stability. |
| `lr_scheduler_type` | cosine | Decay LR smoothly, helps convergence at end. |
| `batch_size` | 16 | Fits in 8GB VRAM with 4-bit quantization. Larger = smoother gradients. |
| `max_grad_norm` | 1.0 | Prevent gradient explosion (LoRA updates can spike). |
| `num_epochs` | 20 | Sufficient iterations for small dataset (551 samples). |
| `fp16` | True | Mixed precision: compute in float16, accumulate in float32 for stability. |
| `optim` | adamw_8bit | 8-bit Adam reduces optimizer VRAM from 2GB to 0.5GB. |
| `weight_decay` | 0.01 | L2 regularization (prevents overfitting). |
| `save_steps` | 100 | Save checkpoint every 100 steps (~1 per epoch). |

### 3.2 Training Loop with SFTTrainer

```python
from transformers import SFTTrainer
from peft import prepare_model_for_kbit_training

def train_gemma(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
):
    """Fine-tune Gemma with SFTTrainer."""
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="gloss",  # Column to use as text
        packing=False,  # Don't pack sequences (poses variable length)
        max_seq_length=256,
    )
    
    # Start training
    result = trainer.train()
    
    return result, trainer


# Example usage:
train_loader = create_data_loader(
    "data/processed/splits/top50/random/train.csv",
    "data/processed/poses",
    tokenizer,
    batch_size=16,
)
eval_loader = create_data_loader(
    "data/processed/splits/top50/random/val.csv",
    "data/processed/poses",
    tokenizer,
    batch_size=16,
)

result, trainer = train_gemma(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_loader.dataset,
    eval_dataset=eval_loader.dataset,
    training_args=training_args,
)
```

### 3.3 What Happens During Training

```
Epoch 1/20:
  Step 1/35:
    Forward pass:
      1. Load batch of 16 pose sequences (16, 256, 177)
      2. Pass through Gemma (compute logits for gloss)
      3. Compute cross-entropy loss with target glosses
      Loss = 3.45
    
    Backward pass:
      1. Compute gradients for LoRA adapters only (3.9M params)
      2. Gradient checkpointing: recompute activations on-the-fly
      3. Skip gradient computation for base model (2B params frozen)
    
    Optimization step:
      1. Update momentum: m_t = 0.9 * m_{t-1} + 0.1 * grad
      2. Update variance: v_t = 0.999 * v_{t-1} + 0.001 * grad^2
      3. Adaptive update: param -= lr * m_t / (sqrt(v_t) + eps)
      4. Weight decay: param -= 0.01 * lr * param
      
      LR = 2e-4 × (warmup schedule) = 2e-4 × sin(π * step / total_steps)
  
  Step 2/35: Loss = 3.23 (decreasing ✓)
  ...
  Step 35/35: Loss = 2.45
  
  Validation:
    - Run eval on 121 validation samples
    - Compute eval_loss = 2.89, eval_accuracy = 0.62
    - Save checkpoint if best eval_loss

Epoch 2/20: (similar loop)
  ...

Epoch 20/20:
  Final validation: eval_loss = 0.42, eval_accuracy = 0.87
  Save best_checkpoint (from epoch 18)
```

---

## Part 4: Loading & Inference

### 4.1 Load Fine-Tuned Checkpoint

```python
def load_finetuned_model(checkpoint_path: Path):
    """Load Gemma + LoRA from checkpoint."""
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E2B-it",
        load_in_4bit=True,
    )
    
    # Load LoRA weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
    )
    
    return model, tokenizer
```

### 4.2 Inference on Test Poses

```python
def predict_gloss(model, tokenizer, pose_path: Path) -> str:
    """Predict ASL gloss for a single pose sequence."""
    
    # Load & normalize pose
    pose = load_pose(pose_path)
    pose = normalize_pose(pose)
    
    # Pad to 256 frames
    if pose.shape[0] < 256:
        pad_len = 256 - pose.shape[0]
        pose = np.pad(pose, ((0, pad_len), (0, 0)), mode='constant')
    
    # Convert to tensor
    pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)  # (1, 256, 177)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=pose_tensor.cuda(),
            max_new_tokens=8,
            num_beams=5,
            temperature=0.7,
            do_sample=False,
        )
    
    # Decode
    predicted_gloss = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return predicted_gloss
```

---

## Part 5: Monitoring & Debugging

### 5.1 Loss Curves

```python
import json
import matplotlib.pyplot as plt

def plot_training_curves(log_file: Path):
    """Plot loss and accuracy curves from training logs."""
    
    with open(log_file) as f:
        logs = [json.loads(line) for line in f if "loss" in line]
    
    steps = [l["step"] for l in logs]
    train_loss = [l.get("loss", None) for l in logs]
    eval_loss = [l.get("eval_loss", None) for l in logs if "eval_loss" in l]
    eval_acc = [l.get("eval_accuracy", None) for l in logs if "eval_accuracy" in l]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(steps, train_loss, label="Train Loss", marker='o')
    axes[0].plot(eval_loss, label="Eval Loss", marker='s')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()
    
    # Accuracy plot
    axes[1].plot(eval_acc, marker='o')
    axes[1].set_xlabel("Eval Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Saved to training_curves.png")
```

**Expected plot:**
```
Loss (epoch 1):     3.45 → 2.45  (training loss decreasing ✓)
Accuracy (epoch 1): 0.20 → 0.45  (random baseline 2% for 50 classes)

Loss (epoch 10):    0.89 → 0.62
Accuracy (epoch 10): 0.72 → 0.78

Loss (epoch 20):    0.42 → 0.35  (well-converged)
Accuracy (epoch 20): 0.87 → 0.86 (plateau = good sign)
```

### 5.2 Checkpointing

```python
def load_best_checkpoint(output_dir: Path):
    """Load the best checkpoint saved during training."""
    
    # Trainer saves best checkpoint with "best-" prefix
    best_checkpoint = output_dir / "best-checkpoint"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        "google/gemma-4-E2B-it",
        load_in_4bit=True,
    )
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, best_checkpoint)
    
    return model, tokenizer
```

---

## Appendix: Common Issues & Solutions

### Issue: "Loss doesn't decrease"

**Diagnosis:**
```python
# Check data loading
dataset = PoseGlossDataset(...)
sample = dataset[0]
print(f"Pose shape: {sample['input_ids'].shape}")
print(f"Label: {sample['labels']}")

# Check if labels are in vocab
print(f"Max token ID: {sample['labels'].max().item()}")
print(f"Vocab size: {tokenizer.vocab_size}")
```

**Solution:**
- Verify pose data is normalized (-1 to 1 range)
- Check gloss tokenization (should be short, e.g., "BOOK" = 1-2 tokens)
- Increase learning rate to 5e-4
- Check gradient flow: Print gradients before/after backward

### Issue: "CUDA out of memory"

**Diagnosis:**
```bash
nvidia-smi  # Check memory usage
```

**Solution:**
- Reduce batch_size: 16 → 8
- Reduce sequence length: 256 → 128
- Disable gradient checkpointing (trades speed for memory)

### Issue: "Overfitting (train loss ↓, eval loss ↑)"

**Solution:**
- Increase lora_dropout: 0.05 → 0.1
- Add more weight decay: 0.01 → 0.05
- Reduce num_epochs: 20 → 10
- Use early stopping: stop if eval_loss doesn't improve for 3 evals

---

## References

- SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- LoRA Config: https://huggingface.co/docs/peft/conceptual_guides/lora
- Unsloth: https://github.com/unslothai/unsloth
- Gemma: https://huggingface.co/google/gemma-4-E2B-it
