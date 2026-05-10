# Multi-word Evaluation Improvement Plan

## Current State (from latest measured run)

Source run used for scoring:
- `evaluation/results/cactus_model_validation/cactus_predictions_test50.csv`
- Derived outputs written to `evaluation/results/multiword_asl_cactus_test50/`

Measured metrics:
- `sample_count`: **50**
- `exact_sequence_accuracy`: **0.18**
- `word_error_rate`: **0.82**
- `substitutions`: **41**
- `insertions`: **0**
- `deletions`: **0**
- `matched_word_recall`: **0.18**
- `timestamp_boundary_mae_ms`: **null** (no boundary timestamps in source)

Important caveat:
- This is parity-flow data mapped from mostly single-word predictions, not a true multi-word sentence benchmark yet.

---

## Recommended Improvements (priority order)

1. Build and run a **true multi-word evaluation dataset**
   - Include sequence-level ground truth and model predictions in `expected_words[]` and `predicted_words[]`.
   - Include optional `start_ms/end_ms` boundaries for timestamp metrics.

2. Reduce substitution errors first
   - Tighten output normalization/canonicalization.
   - Improve decode constraints for transcript output.
   - Add alias/vocabulary mapping before final scoring.

3. Align train/eval with production pipeline
   - Evaluate data produced by the same backend path: video upload -> frame extraction -> pose extraction -> inference.

4. Add confidence calibration reporting
   - Bucket by `sequence_confidence` and compute accuracy per bucket.

5. Enable timestamp quality optimization
   - Ensure predicted word boundaries are emitted so timestamp MAE is measurable and optimizable.

---

## How to Implement Recommendation #1 (true multi-word eval dataset)

### Goal
Create a repeatable evaluation input artifact that represents real multi-word tasks and can be scored by:
- `scripts/evaluation/evaluate_multiword_asl.py`

### Target input schema (JSONL, one sample per line)
```json
{
  "sample_id": "clip_0001",
  "expected_words": [
    {"word": "hello", "start_ms": 0, "end_ms": 320},
    {"word": "how", "start_ms": 320, "end_ms": 520},
    {"word": "you", "start_ms": 520, "end_ms": 760}
  ],
  "predicted_words": [
    {"word": "hello", "start_ms": 20, "end_ms": 300},
    {"word": "how", "start_ms": 340, "end_ms": 530},
    {"word": "are", "start_ms": 530, "end_ms": 780}
  ]
}
```

### Implementation steps

1. **Define the evaluation split**
   - Create a fixed list of clip IDs for multi-word evaluation (e.g., 100–300 clips).
   - Store split manifest under `evaluation/data/multiword_eval_split.json`.

2. **Prepare expected annotations**
   - For each clip, store ordered ground-truth words and (if available) word boundaries.
   - Save as JSONL: `evaluation/data/multiword_expected.jsonl`.

3. **Run backend inference over the same clips**
   - Use current `/v1/translate-sign` runtime path (cloud-first backend pipeline).
   - Persist per-clip prediction payloads, including transcript words and boundaries if present.
   - Save raw runtime outputs to `evaluation/results/multiword_runtime_predictions_raw.jsonl`.

4. **Build normalized eval rows**
   - Create a transformer script to merge expected + predicted by `sample_id` into final eval JSONL:
     - output path: `evaluation/results/multiword_eval_rows.jsonl`
     - enforce required fields and deterministic ordering.

5. **Run evaluator**
   - Execute:
     - `.venv/bin/python scripts/evaluation/evaluate_multiword_asl.py --input-jsonl evaluation/results/multiword_eval_rows.jsonl --out-dir evaluation/results/multiword_eval_run`
   - Artifacts:
     - `predictions.csv`
     - `metrics.json`

6. **Add regression gate (optional but recommended)**
   - Define minimum acceptable thresholds (initially informational, later hard-gated), e.g.:
     - WER <= baseline + tolerance
     - no drop in exact sequence accuracy beyond tolerance
   - Add CI job that runs evaluator on a stable sample subset.

### Validation checklist
- [ ] All rows have `sample_id`, non-empty `expected_words`, non-empty `predicted_words`
- [ ] Word order preserved from source annotations/inference
- [ ] Timestamp fields are integers when provided
- [ ] Metrics are reproducible across reruns on same inputs
- [ ] Report includes both lexical quality (WER/accuracy) and timing quality (MAE)

### Deliverables
- Dataset split manifest
- Expected annotation JSONL
- Runtime prediction JSONL
- Final merged eval rows JSONL
- `metrics.json` + `predictions.csv` artifacts
- Brief markdown run summary with baseline comparison
