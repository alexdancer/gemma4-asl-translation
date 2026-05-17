# Notebook 10 Fine-Tuning Hyperparameters Guide

This document explains the main hyperparameters in `notebooks/10_colab_gemma4_e4b_video_top50_finetune.ipynb`, what each one does, how it affects training, and what I recommend changing for the ASL Top-50 Gemma-4 E4B video-frame fine-tuning run.

The notebook fine-tunes:

```python
BASE_MODEL = "unsloth/gemma-4-E4B-it"
```

on WLASL Top-50 sign-language videos represented as **30 extracted image frames per video**. The model uses Unsloth Vision, LoRA adapters, and a 4-bit loaded base model so it can run more realistically in Colab.

---

## Important Fixes Before Training

Before changing tuning settings, there are two notebook issues worth fixing.

### 1. Define `MAX_SEQ_LENGTH`

The notebook currently uses:

```python
max_seq_length=MAX_SEQ_LENGTH
```

inside the model-loading cell, but the earlier config cell defines only:

```python
MAX_LENGTH = 100_000
```

This can fail unless `MAX_SEQ_LENGTH` already exists from an old Colab run. Use this instead:

```python
MAX_SEQ_LENGTH = 100_000
MAX_LENGTH = MAX_SEQ_LENGTH
```

This keeps the model, tokenizer, and trainer length settings consistent.

### 2. Fix the `SFTConfig` sequence length

The notebook currently includes:

```python
max_seq_length=8196
```

inside `SFTConfig`. This is risky because 30 image frames can require roughly:

```text
30 frames * 256 image tokens = 7,680 image tokens
```

before adding the text prompt and answer tokens. That leaves very little room and can cause truncation or image-token mismatch errors.

Recommended fix:

```python
max_length=MAX_LENGTH
```

If `max_seq_length` is kept, make it consistent:

```python
max_seq_length=MAX_LENGTH
```

Do not leave it at `8196`.

---

## Dataset and Input Settings

### `NUM_FRAMES`

```python
NUM_FRAMES = 30
```

This controls how many frames each video sample contains.

A higher frame count gives the model more motion information, which matters for sign language because many signs are defined by movement rather than a single hand pose. The downside is that more frames create more image tokens, use more VRAM, and increase the chance of Colab running out of memory.

**Recommendation:** keep `30` if training runs successfully. If Colab runs out of memory, reduce frames in this order:

```text
30 frames @ 448
20 frames @ 448
16 frames @ 448
12 frames @ 448
8-12 frames @ 336/384
```

### `IMAGE_SIZE`

```python
IMAGE_SIZE = 448
```

This is the expected resolution for each extracted frame.

Higher resolution preserves details like hand shape, finger position, body posture, and facial cues. These details are useful for ASL recognition, but larger images increase memory and processing cost.

**Recommendation:** keep `448` for now. If memory becomes a major issue after reducing frame count, then try `384` or `336`.

---

## Model Settings

### `BASE_MODEL`

```python
BASE_MODEL = "unsloth/gemma-4-E4B-it"
```

This is the base multimodal model being fine-tuned.

The base model determines the starting visual-language ability. A stronger model may understand image-text tasks better, but it also requires more GPU memory.

**Recommendation:** keep this model. Do not move to a larger model until the E4B pipeline can train, evaluate, and produce useful Top-50 predictions.

---

## Generation Settings

### `MAX_NEW_TOKENS`

```python
MAX_NEW_TOKENS = 8
```

This controls how many tokens the model can generate during prediction.

Since this is a classification task, the model should only output one WLASL gloss label. A small value is good because it discourages long explanations.

**Recommendation:** keep `8`. Increase to `12` only if valid labels are being cut off.

---

## Sequence Length Settings

### `MAX_LENGTH` and `MAX_SEQ_LENGTH`

Recommended setting:

```python
MAX_SEQ_LENGTH = 100_000
MAX_LENGTH = MAX_SEQ_LENGTH
```

These settings are critical because each training example contains 30 images. The processor turns each image into many image tokens, so the full sequence can become very long.

If the max length is too small, the notebook may fail with an error like:

```text
Mismatch in image token count between text and input_ids
```

This usually means the text template expected many image tokens, but the tokenizer or trainer truncated the actual input.

Use the same length consistently in all three places:

```python
FastVisionModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    ...
)
```

```python
processor.tokenizer.model_max_length = MAX_LENGTH
processor.tokenizer.init_kwargs["model_max_length"] = MAX_LENGTH
```

```python
SFTConfig(
    max_length=MAX_LENGTH,
    ...
)
```

---

## Batch Size and Gradient Accumulation

### `PER_DEVICE_BATCH_SIZE`

```python
PER_DEVICE_BATCH_SIZE = 1
```

This is the number of samples processed on the GPU at one time.

For vision fine-tuning with 30 frames per sample, batch size 1 is the safest option because each example is large.

**Recommendation:** keep this at `1`.

### `GRAD_ACCUM_STEPS`

```python
GRAD_ACCUM_STEPS = 4
```

Gradient accumulation simulates a larger batch size without loading all samples into memory at once.

The effective batch size is:

```text
PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
```

With the current settings:

```text
1 * 4 = 4
```

A larger effective batch size usually makes training more stable. A smaller effective batch size trains faster per optimizer step but can be noisier.

**Recommendation:** keep `4` for the first real run. If the model collapses to predicting only a few labels, try:

```python
GRAD_ACCUM_STEPS = 8
```

---

## Learning Rate

### `LEARNING_RATE`

```python
LEARNING_RATE = 2e-4
```

The learning rate controls how aggressively the LoRA adapter updates during training.

A higher learning rate can make the model learn faster, but it also increases the risk of overfitting or prediction collapse. A lower learning rate is more stable but may require more epochs.

**Recommendation:** lower this for the first serious run:

```python
LEARNING_RATE = 1e-4
```

If the model is underfitting after a full run, try returning to:

```python
LEARNING_RATE = 2e-4
```

If the model collapses to a few common labels, try:

```python
LEARNING_RATE = 5e-5
```

---

## LoRA Settings

### `LORA_RANK`

```python
LORA_RANK = 16
```

LoRA rank controls how much trainable capacity the adapter has.

Higher rank means the adapter can learn more complex changes, but it also uses more memory and can overfit more easily.

**Recommendation:** keep `16` for the first real run. If the model clearly underfits after 1-2 full epochs, try:

```python
LORA_RANK = 32
```

### `LORA_ALPHA`

```python
LORA_ALPHA = 16
```

LoRA alpha controls the scaling strength of the LoRA updates. It is usually reasonable to set `LORA_ALPHA` equal to `LORA_RANK`.

**Recommendation:** keep this paired with rank:

```python
LORA_RANK = 16
LORA_ALPHA = 16
```

If increasing rank later:

```python
LORA_RANK = 32
LORA_ALPHA = 32
```

### `lora_dropout`

```python
lora_dropout = 0
```

LoRA dropout adds regularization to the adapter.

A value of `0` is common in Unsloth examples and is faster. However, if the dataset is small or the model overfits, adding a small amount of dropout can help.

**Recommendation:** keep `0` for the first run. If there is overfitting or prediction collapse, try:

```python
lora_dropout = 0.05
```

---

## Which Model Parts Are Fine-Tuned

The notebook currently uses:

```python
finetune_vision_layers=True
finetune_language_layers=True
finetune_attention_modules=True
finetune_mlp_modules=True
target_modules="all-linear"
```

These settings control where LoRA adapters are applied.

### `finetune_vision_layers=True`

This allows the visual part of the model to adapt to ASL frames.

**Recommendation:** keep `True`.

### `finetune_language_layers=True`

This allows the language part of the model to adapt to the classification format and output WLASL gloss labels.

**Recommendation:** keep `True`.

### `finetune_attention_modules=True`

This applies LoRA to attention layers, which are important for relating image and text tokens.

**Recommendation:** keep `True`.

### `finetune_mlp_modules=True`

This applies LoRA to MLP layers as well, giving the adapter more learning capacity.

**Recommendation:** keep `True` for the first run. If memory becomes a serious problem, one possible reduction is:

```python
finetune_mlp_modules=False
```

But reduce frame count before changing this.

### `target_modules="all-linear"`

This applies LoRA broadly to linear layers.

**Recommendation:** keep this. It gives the model enough flexibility to adapt to the ASL classification task.

---

## Memory-Saving Settings

### `load_in_4bit`

```python
load_in_4bit=True
```

This loads the base model in 4-bit quantized form. It dramatically reduces VRAM usage and makes Colab fine-tuning more practical.

**Recommendation:** keep `True`.

### `use_gradient_checkpointing`

```python
use_gradient_checkpointing="unsloth"
```

Gradient checkpointing saves memory by recomputing some activations during backpropagation instead of storing them. This makes training slower but significantly reduces VRAM usage.

**Recommendation:** keep:

```python
use_gradient_checkpointing="unsloth"
```

---

## Training Length

### `SMOKE_MAX_STEPS`

```python
SMOKE_MAX_STEPS = None
```

This controls whether the notebook runs a short smoke test or a full training run.

If set to an integer, training stops after that many steps. This is useful for checking that the pipeline works.

For debugging:

```python
SMOKE_MAX_STEPS = 10
```

For real training:

```python
SMOKE_MAX_STEPS = None
```

### `NUM_TRAIN_EPOCHS`

```python
NUM_TRAIN_EPOCHS = 1
```

This controls how many passes the model makes through the full training dataset.

One epoch may be enough to prove the setup works, but it may not be enough for meaningful ASL classification performance.

**Recommendation:** for the first real run:

```python
NUM_TRAIN_EPOCHS = 2
```

If the model is still improving and not overfitting, try:

```python
NUM_TRAIN_EPOCHS = 3
```

---

## Evaluation and Saving Strategy

Current notebook logic:

```python
eval_strategy="steps" if SMOKE_MAX_STEPS else "epoch"
eval_steps=5 if SMOKE_MAX_STEPS else None
save_strategy="steps" if SMOKE_MAX_STEPS else "epoch"
save_steps=5 if SMOKE_MAX_STEPS else None
```

This means:

- During smoke runs, evaluate and save every 5 steps.
- During full runs, evaluate and save once per epoch.

**Recommendation:** this is fine for now.

### `logging_steps`

```python
logging_steps=1
```

This logs training metrics every step. It is useful for debugging but can be noisy.

For smoke runs, keep:

```python
logging_steps=1
```

For real training, use:

```python
logging_steps=5
```

---

## Optimizer

### `optim`

```python
optim="adamw_8bit"
```

This uses an 8-bit AdamW optimizer, which saves memory compared with full-precision AdamW.

**Recommendation:** keep:

```python
optim="adamw_8bit"
```

---

## Weight Decay

### `weight_decay`

```python
weight_decay=0.001
```

Weight decay regularizes the model and can reduce overfitting.

**Recommendation:** keep `0.001`. If the model overfits, try:

```python
weight_decay=0.01
```

---

## Learning Rate Scheduler

### `lr_scheduler_type`

```python
lr_scheduler_type="cosine"
```

The cosine scheduler gradually decreases the learning rate during training. This is a good default for fine-tuning.

**Recommendation:** keep:

```python
lr_scheduler_type="cosine"
```

---

## Warmup

Current notebook:

```python
warmup_steps=0.03
```

This is probably not the right argument. `warmup_steps` usually expects an integer number of steps. A decimal like `0.03` is better represented as a ratio.

Recommended change:

```python
warmup_ratio=0.03
```

This means 3% of training is used for warmup.

Replace:

```python
warmup_steps=0.03
```

with:

```python
warmup_ratio=0.03
```

---

## Precision Settings

Current notebook:

```python
fp16=not torch.cuda.is_bf16_supported()
bf16=torch.cuda.is_bf16_supported()
```

This automatically uses BF16 if the GPU supports it, otherwise FP16. BF16 is usually more stable on newer GPUs.

**Recommendation:** keep this logic.

---

## Random Seed

### `SEED`

```python
SEED = 3407
```

The seed helps make training more reproducible.

**Recommendation:** keep it.

---

## Dataset Limits

### `TRAIN_LIMIT` and `VAL_LIMIT`

```python
TRAIN_LIMIT = None
VAL_LIMIT = None
```

These control whether the notebook uses the full dataset or only a small subset.

For debugging:

```python
TRAIN_LIMIT = 32
VAL_LIMIT = 16
```

For real training:

```python
TRAIN_LIMIT = None
VAL_LIMIT = None
```

---

## Evaluation Limit

Current notebook:

```python
EVAL_LIMIT = min(25, len(test_rows))
```

This only evaluates up to 25 test samples. That is fine for quick debugging, but it is not enough for a final result or Kaggle write-up.

For final evaluation, change it to:

```python
EVAL_LIMIT = len(test_rows)
```

---

## Recommended Settings for Next Run

For the next real fine-tuning attempt, use:

```python
NUM_FRAMES = 30
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 8

MAX_SEQ_LENGTH = 100_000
MAX_LENGTH = MAX_SEQ_LENGTH

PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 1e-4

LORA_RANK = 16
LORA_ALPHA = 16

SMOKE_MAX_STEPS = None
NUM_TRAIN_EPOCHS = 2
```

In `SFTConfig`, use:

```python
warmup_ratio=0.03
max_length=MAX_LENGTH
```

Remove or fix:

```python
max_seq_length=8196
```

For final evaluation, use:

```python
EVAL_LIMIT = len(test_rows)
```

---

## Recommended Training Plan

### Step 1: Smoke Test

Use a tiny subset first:

```python
TRAIN_LIMIT = 32
VAL_LIMIT = 16
SMOKE_MAX_STEPS = 10
```

Goal: confirm that model loading, training, saving, and evaluation all work. Do not judge model quality from this run.

### Step 2: First Real Run

Use the full dataset:

```python
TRAIN_LIMIT = None
VAL_LIMIT = None
SMOKE_MAX_STEPS = None
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 1e-4
```

This is the recommended first meaningful experiment.

### Step 3: If the Model Underfits

If accuracy is poor and the model does not seem to learn enough, try one of these:

```python
NUM_TRAIN_EPOCHS = 3
```

or:

```python
LEARNING_RATE = 2e-4
```

or:

```python
LORA_RANK = 32
LORA_ALPHA = 32
```

Change only one major setting at a time so the results are easier to interpret.

### Step 4: If Predictions Collapse

If the model predicts only a few labels repeatedly, try:

```python
LEARNING_RATE = 5e-5
```

or:

```python
GRAD_ACCUM_STEPS = 8
```

or:

```python
lora_dropout = 0.05
```

Also make sure evaluation is not limited to only 25 examples before drawing conclusions.

### Step 5: If Colab Runs Out of Memory

Reduce the frame count before changing the model.

Use this fallback order:

```text
30 frames @ 448
20 frames @ 448
16 frames @ 448
12 frames @ 448
8-12 frames @ 336/384
```

After changing the frame count, regenerate the dataset and update:

```python
NUM_FRAMES = ...
IMAGE_SIZE = ...
```

to match the new preprocessing settings.

---

## Short Summary

The most important recommended changes are:

```python
MAX_SEQ_LENGTH = 100_000
MAX_LENGTH = MAX_SEQ_LENGTH
```

Change:

```python
LEARNING_RATE = 2e-4
```

to:

```python
LEARNING_RATE = 1e-4
```

Change:

```python
NUM_TRAIN_EPOCHS = 1
```

to:

```python
NUM_TRAIN_EPOCHS = 2
```

Replace:

```python
warmup_steps=0.03
```

with:

```python
warmup_ratio=0.03
```

Remove or fix:

```python
max_seq_length=8196
```

And for final evaluation, change:

```python
EVAL_LIMIT = min(25, len(test_rows))
```

to:

```python
EVAL_LIMIT = len(test_rows)
```

These changes should make the training setup more stable, more consistent with the 30-frame vision input, and more useful for reporting real results.
