# Top-50 Unsloth q64 Proof Decision

Parent issue: #10  
Decision issue: #16  
Evaluation issue: #15  
Checkpoint: `checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline`  
Metrics source: `evaluation/results/unsloth_top50_q64_full_dashboard_baseline/metrics.json`  
Predictions source: `evaluation/results/unsloth_top50_q64_full_dashboard_baseline/predictions.csv`

## Decision

**Yellow.**

The checkpoint shows real learning signal, but it is not strong enough to move directly into demo integration or full-250 scaling.

## Metrics

| Metric | Value |
|---|---:|
| Held-out samples | 50 |
| Strict normalized top-1 accuracy | 40.00% |
| Correct predictions | 20 / 50 |
| Invalid-output rate | 60.00% |
| Invalid outputs | 30 / 50 |
| Class count | 50 |

## Threshold ladder

| Band | Threshold | Outcome |
|---|---|---|
| Strong go | >= 80% | Proceed to demo integration, then scale |
| Go | >= 70% and < 80% | Proceed to scoped demo integration before scaling |
| Yellow | >= 40% and < 70% | Tune/diagnose before demo or scaling |
| No-go/debug | < 40% | Investigate data, prompt, checkpoint loading, or evaluator bugs |

## Evidence

The model got 20 held-out examples exactly correct, so this is not random/no-signal behavior. However, 30 predictions were outside the canonical Top-50 manifest.

Most common invalid predictions:

- `thank`: 12
- `waving`: 5
- `wave`: 3
- `hello`: 1
- `gaze`: 1
- `sign`: 1
- `point`: 1
- `noise`: 1
- `crying`: 1
- `wav`: 1
- `question`: 1
- `hsabaotbrdybso`: 1


Correct held-out examples included:

- `decide_15040`: expected `decide`, predicted `decide`
- `clothes_11305`: expected `clothes`, predicted `clothes`
- `backpack_04627`: expected `backpack`, predicted `backpack`
- `hot_28122`: expected `hot`, predicted `hot`
- `door_68038`: expected `door`, predicted `door`
- `book_68012`: expected `book`, predicted `book`
- `cow_13681`: expected `cow`, predicted `cow`
- `like_33277`: expected `like`, predicted `like`
- `again_01468`: expected `again`, predicted `again`
- `forget_22962`: expected `forget`, predicted `forget`
- `drink_17728`: expected `drink`, predicted `drink`
- `birthday_06369`: expected `birthday`, predicted `birthday`
- `kiss_31757`: expected `kiss`, predicted `kiss`
- `enjoy_19266`: expected `enjoy`, predicted `enjoy`
- `dance_14628`: expected `dance`, predicted `dance`
- `different_16199`: expected `different`, predicted `different`
- `color_11775`: expected `color`, predicted `color`
- `mother_36944`: expected `mother`, predicted `mother`
- `coffee_68025`: expected `coffee`, predicted `coffee`
- `chair_09855`: expected `chair`, predicted `chair`


## Recommendation

Do **not** proceed to demo integration (#18) yet, and do **not** scale to full-250 yet.

Activate #17: constrained Top-50 scoring diagnostic. The immediate question is whether the adapter learned useful pose-to-gloss signal but failed to stay inside the allowed label vocabulary, or whether the underlying classification signal is still weak.

## Concrete next steps

1. Run a constrained Top-50 scoring diagnostic against the same held-out test split.
2. Compare constrained-label accuracy to free-generation strict accuracy.
3. If constrained accuracy improves meaningfully, fix output control: label list in prompt, decode constraints, shorter answer schema, or post-generation nearest-label mapping for demo only.
4. If constrained accuracy remains near 40%, tune training/encoding: more epochs, different LR, signer-aware split later, better pose encoding, or a smaller/cleaner label subset.
