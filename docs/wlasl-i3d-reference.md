# WLASL I3D Reference

## What is I3D?

**I3D** means **Inflated 3D ConvNet**. It is a video classification model introduced for action recognition and later used as a strong baseline in the WLASL paper.

The core idea is simple:

- A normal 2D CNN looks at individual images.
- I3D extends, or “inflates,” those 2D filters into **3D filters**.
- That lets the model learn from both:
  - **space**: hand shape, body pose, face/upper-body appearance
  - **time**: motion across video frames

So for sign language, I3D is useful because signs are not just static hand shapes. They depend heavily on **movement over time**.

## Why I3D mattered in the WLASL paper

In the WLASL paper, I3D was the strongest reported baseline overall. It outperformed:

- **VGG-GRU**: appearance model using frame CNN features plus a recurrent GRU
- **Pose-GRU**: pose-sequence model using OpenPose keypoints plus GRU
- **Pose-TGCN**: pose-sequence model using temporal graph convolution

The paper used I3D with pretrained weights from ImageNet + Kinetics, then fine-tuned it on WLASL subsets.

Important training note from the paper:

- They tried SGD for I3D, but it did not converge well.
- They used **Adam optimizer** for all models, including I3D.
- Models were trained up to **200 epochs**, with early stopping when validation accuracy stopped increasing.

## WLASL evaluation metric

The paper reports **Top-K classification accuracy**:

- **Top-1**: correct label must be the model’s first prediction.
- **Top-5**: correct label can be anywhere in the top 5 predictions.
- **Top-10**: correct label can be anywhere in the top 10 predictions.

This matters because many ASL signs are visually similar, so Top-5/Top-10 can show whether the model is close even when Top-1 is wrong.

## WLASL model metrics

Source: Li et al., **“Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison”**, arXiv:1910.11006.

| Model | WLASL100 Top-1 | WLASL100 Top-5 | WLASL100 Top-10 | WLASL300 Top-1 | WLASL300 Top-5 | WLASL300 Top-10 | WLASL1000 Top-1 | WLASL1000 Top-5 | WLASL1000 Top-10 | WLASL2000 Top-1 | WLASL2000 Top-5 | WLASL2000 Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pose-GRU | 46.51 | 76.74 | 85.66 | 33.68 | 64.37 | 76.05 | 30.01 | 58.42 | 70.15 | 22.54 | 49.81 | 61.38 |
| Pose-TGCN | 55.43 | 78.68 | 87.60 | 38.32 | 67.51 | 79.64 | 34.86 | 61.73 | 71.91 | 23.65 | 51.75 | 62.24 |
| VGG-GRU | 25.97 | 55.04 | 63.95 | 19.31 | 46.56 | 61.08 | 14.66 | 37.31 | 49.36 | 8.44 | 23.58 | 32.58 |
| **I3D** | **65.89** | **84.11** | **89.92** | **56.14** | **79.94** | **86.98** | **47.33** | **76.44** | **84.33** | **32.48** | **57.31** | **66.31** |

## I3D takeaway

I3D was the best WLASL baseline because it directly models video motion and appearance together.

However, even I3D only reached **32.48% Top-1 accuracy on WLASL2000**, showing that large-vocabulary isolated sign recognition is difficult. The dataset has many classes, signer variation, dialect variation, visually similar signs, and relatively few examples per class.

For our Gemma-4 ASL work, I3D is a useful historical baseline, especially the WLASL2000 result:

```text
I3D on WLASL2000:
Top-1:  32.48%
Top-5:  57.31%
Top-10: 66.31%
```

## How this relates to Gemma-4 ASL fine-tuning

I3D is a specialized video classifier. Gemma-4 multimodal fine-tuning is different because it uses a vision-language model and prompts the model to output a gloss label.

That means our Gemma-4 setup needs extra controls that I3D did not need:

- strict label allowlist
- consistent train/eval prompt format
- deterministic label extraction
- small `max_new_tokens` for single-label prediction
- frame sampling quality checks
- final evaluation on canonical WLASL Top-50

The WLASL I3D numbers are still useful as a benchmark for how hard the dataset is, but they are not a direct apples-to-apples comparison unless the same subset, labels, splits, and evaluation protocol are used.
