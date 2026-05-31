# Notebook 12 WLASL Zahid-labelset batch results

## Context

Notebook 12 was used to batch-test the Notebook 11 model on a 50-video WLASL upload zip.

Important framing:

- The model used was the Notebook 11 Zahid-pretrained adapter:
  - `AlexD281/asl-gemma4-26b-a4b-zahid-pretrain-lora`
- The model was **not trained on WLASL**.
- Therefore these results are **out-of-domain WLASL evaluation**, not in-domain training-set validation.
- The first WLASL zip used an older conversational Top-50 label list and produced misleadingly low accuracy because the expected labels did not match the model label space.

## Batch zip used for the valid run

```text
data/notebook12_wlasl_zahid_labelset_50_video_eval.zip
```

Zip contents:

- 50 WLASL `.mp4` clips
- `labels.csv`
- `README.md`
- 47 unique Zahid/model labels
- 3 duplicate/easy labels to reach 50 total:
  - `yes`
  - `no`
  - `fine`

## Summary

| Run | Label alignment | Correct | Total | Accuracy |
|---|---:|---:|---:|---:|
| Old WLASL conversational Top-50 zip | Wrong label space | 3 | 50 | 6% |
| Correct WLASL Zahid-labelset zip | Model label space | 33 | 50 | 66% |

The corrected run strongly suggests the Notebook 12 batch pipeline is working. The earlier bad result was caused by expected-label mismatch, not by batch inference being broken.

## Corrected batch result

```python
{
  'total': 50,
  'labeled': 50,
  'ok': 49,
  'correct': 33,
  'incorrect': 17,
  'uncertain': 1,
  'failed': 0,
  'accuracy_on_labeled': 0.66,
}
```

Notebook log path from the run:

```text
/content/drive/MyDrive/asl/notebook12_wlasl_top50_batch_results.jsonl
```

Note: this log filename is stale/misleading. It should be renamed in the notebook to something like:

```text
/content/drive/MyDrive/asl/notebook12_wlasl_zahid_labelset_batch_results.jsonl
```

## Per-video results

| # | Video | Expected | Predicted | Status | Correct |
|---:|---|---|---|---|---|
| 1 | `001_all_01912.mp4` | all | book | ok | False |
| 2 | `002_before_05724.mp4` | before | before | ok | True |
| 3 | `003_better_06062.mp4` | better | hearing | ok | False |
| 4 | `004_bird_06326.mp4` | bird | bird | ok | True |
| 5 | `005_birthday_06355.mp4` | birthday | white | ok | False |
| 6 | `006_black_06455.mp4` | black | hat | ok | False |
| 7 | `007_blue_06822.mp4` | blue | blue | ok | True |
| 8 | `008_book_07075.mp4` | book | finish | ok | False |
| 9 | `009_but_08421.mp4` | but | but | ok | True |
| 10 | `010_candy_08909.mp4` | candy | candy | ok | True |
| 11 | `011_chair_09847.mp4` | chair | chair | ok | True |
| 12 | `012_clothes_11305.mp4` | clothes | clothes | ok | True |
| 13 | `013_college_11704.mp4` | college | college | ok | True |
| 14 | `014_color_11752.mp4` | color | color | ok | True |
| 15 | `015_computer_12306.mp4` | computer | study | ok | False |
| 16 | `016_cook_13154.mp4` | cook | college | ok | False |
| 17 | `017_cow_13681.mp4` | cow | cow | ok | True |
| 18 | `018_dance_14621.mp4` | dance | dance | ok | True |
| 19 | `019_deaf_14855.mp4` | deaf | candy | ok | False |
| 20 | `020_door_17317.mp4` | door | door | ok | True |
| 21 | `021_drink_17725.mp4` | drink | fine | ok | False |
| 22 | `022_enjoy_19255.mp4` | enjoy | enjoy | ok | True |
| 23 | `023_fine_21884.mp4` | fine | fine | ok | True |
| 24 | `024_finish_21933.mp4` | finish | finish | ok | True |
| 25 | `025_fish_22109.mp4` | fish | fish | ok | True |
| 26 | `026_hat_26688.mp4` | hat | hat | ok | True |
| 27 | `027_hearing_26982.mp4` | hearing | None | uncertain | False |
| 28 | `028_hot_28074.mp4` | hot | hot | ok | True |
| 29 | `029_like_33277.mp4` | like | fine | ok | False |
| 30 | `030_many_34822.mp4` | many | many | ok | True |
| 31 | `031_meet_35506.mp4` | meet | meet | ok | True |
| 32 | `032_mother_36927.mp4` | mother | will | ok | False |
| 33 | `033_need_37879.mp4` | need | yes | ok | False |
| 34 | `034_no_38482.mp4` | no | yes | ok | False |
| 35 | `035_now_38982.mp4` | now | now | ok | True |
| 36 | `036_orange_40114.mp4` | orange | orange | ok | True |
| 37 | `037_paper_41008.mp4` | paper | paper | ok | True |
| 38 | `038_study_55356.mp4` | study | study | ok | True |
| 39 | `039_use_61031.mp4` | use | use | ok | True |
| 40 | `040_visit_61804.mp4` | visit | hearing | ok | False |
| 41 | `041_white_63191.mp4` | white | white | ok | True |
| 42 | `042_will_63358.mp4` | will | blue | ok | False |
| 43 | `043_woman_63662.mp4` | woman | woman | ok | True |
| 44 | `044_wrong_64082.mp4` | wrong | wrong | ok | True |
| 45 | `045_year_64201.mp4` | year | year | ok | True |
| 46 | `046_yes_64275.mp4` | yes | yes | ok | True |
| 47 | `047_your_64423.mp4` | your | your | ok | True |
| 48 | `048_yes_64294.mp4` | yes | yes | ok | True |
| 49 | `049_no_38538.mp4` | no | fine | ok | False |
| 50 | `050_fine_21885.mp4` | fine | fine | ok | True |

## Observations

- The batch result improved from **6%** to **66%** after aligning expected labels to the model label vocabulary.
- This confirms label-space mismatch was the dominant cause of the earlier failed batch run.
- The remaining errors are plausible out-of-domain errors because WLASL signer/video distribution differs from Zahid training data.
- Some confusions repeat and may be useful for later analysis:
  - `no -> yes` / `no -> fine`
  - `drink -> fine`
  - `like -> fine`
  - `mother -> will`
  - `will -> blue`
  - `visit -> hearing`

## Recommended follow-up

1. Patch Notebook 12 batch output path from `top50` to `zahid_labelset` to avoid future confusion.
2. Add a notebook warning that `labels.csv` must match the model label space.
3. Optionally create a confusion-matrix helper from the batch JSONL logs.
