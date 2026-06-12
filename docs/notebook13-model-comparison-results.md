# Notebook 13 — Zahid vs. Zahid+WLASL adapter comparison

Latest run captured in `notebooks/output/model_comparison_11_vs_14.json`.

## Setup

- **Base model:** `unsloth/gemma-4-26B-A4B-it`
- **Eval cap (`EVAL_MAX_SAMPLES`):** 200
- **Frames per sample:** 30 raw frames, subsampled to 6 for inference
- **Adapters compared:**
  - `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora` (Notebook 11)
  - `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora` (Notebook 14)

Each adapter is run on the same six splits: `zahid_val`, `zahid_test`, `wlasl_val`, `wlasl_test`, `combined_val`, `combined_test`. `combined_*` is the concatenation of the matching Zahid and WLASL split.

## Important framing

- `EVAL_MAX_SAMPLES = 200` is an **upper bound**, not a target. The Zahid bundle and the combined splits have ≥ 200 samples per split and the loop hits the cap. The WLASL Top-50 bundle currently has only 59 val and 63 test preprocessed clips, so those splits stop at the data limit (see "Data ceiling" below).
- The numbers below are **out-of-domain for Notebook 11**: the Notebook 11 adapter was trained on Zahid only and is being evaluated on Zahid (in-domain) and on WLASL (out-of-domain). The Notebook 14 adapter was trained on Zahid + WLASL combined, so for it, both domains are in-domain.
- Notebook 11 is the baseline; Notebook 14 is the candidate. Negative accuracy deltas on Zahid splits mean the combined adapter regressed on in-domain Zahid (likely an overfit / catastrophic-forgetting signal on the smaller combined dataset). Positive delta on `wlasl_val` is consistent with the combined adapter's intended purpose.

## Results

### Notebook 11 — Zahid-only pretrain

Adapter: `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`

| Split        | Samples | Correct | Invalid | Accuracy | Invalid rate |
|--------------|--------:|--------:|--------:|---------:|-------------:|
| zahid_val    |     200 |     121 |       1 |   60.5%  |        0.5%  |
| zahid_test   |     200 |     130 |       0 |   65.0%  |        0.0%  |
| wlasl_val    |      59 |      24 |       0 |   40.7%  |        0.0%  |
| wlasl_test   |      63 |      38 |       0 |   60.3%  |        0.0%  |
| combined_val |     200 |     121 |       1 |   60.5%  |        0.5%  |
| combined_test|     200 |     127 |       0 |   63.5%  |        0.0%  |

### Notebook 14 — Zahid + WLASL combined

Adapter: `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`

| Split        | Samples | Correct | Invalid | Accuracy | Invalid rate |
|--------------|--------:|--------:|--------:|---------:|-------------:|
| zahid_val    |     200 |     109 |       7 |   54.5%  |        3.5%  |
| zahid_test   |     200 |     111 |       9 |   55.5%  |        4.5%  |
| wlasl_val    |      59 |      30 |       6 |   50.8%  |       10.2%  |
| wlasl_test   |      63 |      36 |       3 |   57.1%  |        4.8%  |
| combined_val |     200 |     109 |       8 |   54.5%  |        4.0%  |
| combined_test|     200 |     112 |       9 |   56.0%  |        4.5%  |

### Deltas (Notebook 14 minus Notebook 11)

| Split        | Accuracy Δ | Correct Δ | Invalid rate Δ |
|--------------|-----------:|----------:|---------------:|
| zahid_val    |     −6.0%  |       −12 |          +3.0% |
| zahid_test   |     −9.5%  |       −19 |          +4.5% |
| wlasl_val    |    +10.2%  |        +6 |         +10.2% |
| wlasl_test   |     −3.2%  |        −2 |          +4.8% |
| combined_val |     −6.0%  |       −12 |          +3.5% |
| combined_test|     −7.5%  |       −15 |          +4.5% |

## Reading the results

- **Zahid (in-domain for both):** Notebook 11 wins on accuracy (60.5 / 65.0 vs 54.5 / 55.5) and on invalid rate (0.5 / 0.0 vs 3.5 / 4.5). Adding the WLASL data to training regressed Zahid performance.
- **WLASL (in-domain only for Notebook 14):** Notebook 14 wins on `wlasl_val` (50.8% vs 40.7%, +6 correct), but Notebook 11 wins on `wlasl_test` (60.3% vs 57.1%, +2 correct). The val/test inconsistency on the small WLASL splits (59 / 63) means the WLASL numbers are noisy and should not be over-interpreted.
- **Combined:** Notebook 11 wins by 6.0% (val) and 7.5% (test) on the merged split, because the Zahid half dominates the count and Notebook 11 is stronger on Zahid.
- **Invalid rate:** Notebook 14's invalid rate is materially higher (3.5–10.2% vs 0.0–0.5%), suggesting the combined adapter occasionally falls back to non-allowlist text on harder clips. The notebook's gatekeeper catches these and reports them as `__invalid__` rather than coercing to a fake gloss.

## Data ceiling

The eval cap is 200, but the actual evaluated sizes per split are:

| Split        | Cap | Actual | Reason                                  |
|--------------|----:|-------:|-----------------------------------------|
| zahid_val    | 200 |    200 | Hit cap                                  |
| zahid_test   | 200 |    200 | Hit cap                                  |
| wlasl_val    | 200 |     59 | **Data limit** — only 59 preprocessed WLASL clips available for this split |
| wlasl_test   | 200 |     63 | **Data limit** — only 63 preprocessed WLASL clips available for this split |
| combined_val | 200 |    200 | Hit cap (Zahid + WLASL; Zahid is the larger half) |
| combined_test| 200 |    200 | Hit cap                                  |

The WLASL Top-50 bundle (`data/video_finetune/top50/`) was built from the preprocessed clips in `data/WLASL/start_kit/videos/`. That folder only contains 5–8 clips per Top-50 gloss. The 723 WLASL instances flagged `missing_preprocessed_clip` in `top50/failures.jsonl` correspond to clips that are listed in `WLASL_v0.3.json` but are not on disk in the preprocessed shape the extraction script expects. The `raw_videos/` folder (1,962 files, YouTube-ID named) is a separate, unprocessed cache and would require running the WLASL preprocessing pipeline before it can be extracted into the standard frame format.

Until the WLASL preprocessed set is expanded, the WLASL eval rows above are the best evidence we have and should be reported as a small-N result, not a definitive benchmark.

## Headline numbers for slides / docs

- **Notebook 11 (Zahid pretrain):** Zahid val **121/200 = 60.5%**; Zahid test **130/200 = 65.0%**; WLASL val **24/59 = 40.7%**; WLASL test **38/63 = 60.3%**.
- **Notebook 14 (Zahid + WLASL combined):** Zahid val **109/200 = 54.5%**; Zahid test **111/200 = 55.5%**; WLASL val **30/59 = 50.8%**; WLASL test **36/63 = 57.1%**.
- **Net:** Adding WLASL training data helps the in-domain WLASL val split by 6 correct (40.7% → 50.8%) but regresses in-domain Zahid by 12–19 correct across val/test. The smaller, noisier WLASL split prevents a clean verdict — both adapters sit in the same accuracy band on `wlasl_test`.

## Reproducing

1. Run Notebook 11 (or use the existing adapter) — produces `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`.
2. Run Notebook 14's combined training — produces `AlexD281/asl-gemma4-26b-a4b-zahid-wlasl-combined-top50-lora`.
3. Open `notebooks/13_colab_hf_model_eval_top50.ipynb` in a Colab H100/H200 runtime.
4. Confirm `EVAL_MAX_SAMPLES = 200` in the config cell, `MODEL_REPOS` lists both adapters, and the Zahid/WLASL zips are in Drive at the documented paths.
5. Run all cells. The summary JSON is written to:
   ```text
   /content/asl_gemma4_26b_model_comparison/model_comparison_11_vs_14.json
   ```
6. Copy that JSON to `notebooks/output/model_comparison_11_vs_14.json` to refresh this report.

## Source of truth

- Eval JSON: `notebooks/output/model_comparison_11_vs_14.json`
- Notebook: `notebooks/13_colab_hf_model_eval_top50.ipynb`
- Failure log (why WLASL is small): `data/video_finetune/top50/failures.jsonl`
