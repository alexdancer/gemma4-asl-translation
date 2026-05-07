# Issue #21 — Constrained Top-50 Root-Cause Report

Parent diagnostic issue: #17  
Implementation dependency: #20  
Checkpoint: `checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline`  
Held-out split: `data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl`  
Constrained artifacts: `evaluation/results/unsloth_top50_q64_full_dashboard_baseline_constrained`  
Baseline free-generation artifacts: `evaluation/results/unsloth_top50_q64_full_dashboard_baseline`

## Result

**Classification: output-control failure with usable learned signal.**

The constrained diagnostic improved held-out top-1 accuracy from **40%** to **70%**. That is a +30 percentage-point gain and reaches the project’s **Go** threshold when the model is forced to choose from the canonical Top-50 labels. The original free-generation run had a **60% invalid-output rate**, so the Yellow result is primarily caused by free generation drifting outside the allowed label vocabulary, not by total failure to learn pose-to-gloss signal.

## Metrics

| Metric | Free generation | Constrained Top-50 | Delta |
|---|---:|---:|---:|
| Held-out samples | 50 | 50 | — |
| Top-1 accuracy | 40% | 70% | +30% |
| Correct predictions | 20 | 35 | +15 |
| Invalid-output rate | 60% | 0% by design | -60% |

## Key evidence

- Free generation correct: **20 / 50**
- Constrained scoring correct: **35 / 50**
- Free-generation wrong → constrained correct rescues: **15**
- Free-generation correct → constrained wrong regressions: **0**
- Constrained wrong after diagnosis: **15**

The important pattern is that constrained scoring preserved every free-generation correct answer and rescued 15 examples that free generation missed, mostly because free generation emitted invalid labels like `thank`, `waving`, `wave`, or `hello`.

## Examples rescued by constrained scoring

- `center_09719`: expected `center`, free generated `thank` (valid=False), constrained chose `center`
- `bird_06341`: expected `bird`, free generated `gaze` (valid=False), constrained chose `bird`
- `paper_70211`: expected `paper`, free generated `sign` (valid=False), constrained chose `paper`
- `need_37891`: expected `need`, free generated `point` (valid=False), constrained chose `need`
- `candy_08926`: expected `candy`, free generated `waving` (valid=False), constrained chose `candy`
- `deaf_14899`: expected `deaf`, free generated `thank` (valid=False), constrained chose `deaf`
- `yes_64275`: expected `yes`, free generated `thank` (valid=False), constrained chose `yes`
- `finish_70361`: expected `finish`, free generated `crying` (valid=False), constrained chose `finish`
- `college_68026`: expected `college`, free generated `wave` (valid=False), constrained chose `college`
- `but_08432`: expected `but`, free generated `waving` (valid=False), constrained chose `but`
- `white_63210`: expected `white`, free generated `question` (valid=False), constrained chose `white`
- `orange_40122`: expected `orange`, free generated `thank` (valid=False), constrained chose `orange`
- `visit_61815`: expected `visit`, free generated `thank` (valid=False), constrained chose `visit`
- `cook_68029`: expected `cook`, free generated `hsabaotbrdybso` (valid=False), constrained chose `cook`
- `business_08363`: expected `business`, free generated `thank` (valid=False), constrained chose `business`


## Remaining constrained failures

These are the examples still wrong even after forcing the model to choose from Top-50 labels:

- `hearing_26986`: expected `hearing`, free generated `thank`, constrained chose `like`; top score `like`=-16.0312
- `black_06483`: expected `black`, free generated `hello`, constrained chose `like`; top score `like`=-11.9453
- `no_38541`: expected `no`, free generated `thank`, constrained chose `deaf`; top score `deaf`=-18.7500
- `wrong_64094`: expected `wrong`, free generated `waving`, constrained chose `no`; top score `no`=-9.7891
- `fine_70234`: expected `fine`, free generated `noise`, constrained chose `no`; top score `no`=-7.4141
- `now_68114`: expected `now`, free generated `thank`, constrained chose `no`; top score `no`=-12.2422
- `fish_22109`: expected `fish`, free generated `waving`, constrained chose `no`; top score `no`=-10.8906
- `city_10899`: expected `city`, free generated `wave`, constrained chose `dance`; top score `dance`=-11.4453
- `blue_06840`: expected `blue`, free generated `thank`, constrained chose `no`; top score `no`=-7.2812
- `computer_12306`: expected `computer`, free generated `wave`, constrained chose `dance`; top score `dance`=-13.4844
- `all_68001`: expected `all`, free generated `waving`, constrained chose `drink`; top score `drink`=-10.3438
- `before_05746`: expected `before`, free generated `wav`, constrained chose `no`; top score `no`=-13.0938
- `africa_01392`: expected `africa`, free generated `thank`, constrained chose `no`; top score `no`=-13.2422
- `study_55370`: expected `study`, free generated `thank`, constrained chose `like`; top score `like`=-13.0156
- `brown_70242`: expected `brown`, free generated `greetings`, constrained chose `yes`; top score `yes`=-12.8438


## Recommendation

Proceed to the output-control branch first:

1. **Start #22 — Output-control prompt experiment for invalid Top-50 predictions.**
   - Try a stricter prompt that explicitly says to choose exactly one label from the canonical Top-50 list.
   - Keep output artifacts separate from the baseline and constrained diagnostic.
   - Success target: preserve most of the constrained 70% signal while reducing invalid free-generation outputs materially.

2. If prompt-only control is not enough, proceed to **#23 — Demo-safe constrained inference mode**.
   - The constrained result is strong enough to justify a demo-safe constrained mode, as long as we label it honestly as constrained Top-50 inference.

3. Do **not** start #24 tuning yet.
   - Tuning may still help, but the immediate blocker is output control. The model already reaches the Go threshold under constrained scoring.

## Caveat

Constrained scoring is diagnostic/demo-support evidence, not a replacement for the original strict free-generation proof metric. The official free-generation metric remains 40% until output-control changes are evaluated separately.
