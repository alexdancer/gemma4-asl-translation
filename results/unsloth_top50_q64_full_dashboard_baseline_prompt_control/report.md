# Issue #22 - Prompt-Control Output Experiment

## Result

Prompt/output-control changes are enough to try before retraining: invalid outputs fell while strict exact-match accuracy was preserved or improved.

## Metrics

| Metric | Baseline free generation | Prompt control | Constrained diagnostic |
|---|---:|---:|---:|
| Held-out samples | 50 | 50 | 50 |
| Top-1 accuracy | 40% | 80% | 70% |
| Invalid-output rate | 60% | 0% | 0% |
| Correct predictions | 20 | 40 | 35 |

## Deltas

- Prompt-control vs baseline accuracy: +40%
- Prompt-control vs baseline invalid-output rate: -60%
- Prompt-control vs constrained accuracy: +10%

## Recommendation

- Enough before retraining: True
