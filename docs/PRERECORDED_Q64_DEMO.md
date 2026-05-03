# Prerecorded Top-50 q64 Demo Path

This path is demo-scoped. It proves that one known-good compact q64 record can
run through the validated Top-50 checkpoint and the shared q64 prediction
contract to produce visible gloss output. It is not production-grade ASL
recognition, and it does not claim coverage outside the supported Top-50 signs.

Live camera inference and full-250 scaling remain deferred until this scoped
checkpoint path is reviewed.

## Smoke Run

```bash
./venv/bin/python scripts/run_prerecorded_q64_demo.py \
  --checkpoint checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline \
  --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --record-id hearing_26986 \
  --out-dir evaluation/results/prerecorded_q64_demo
```

The readiness artifact is written to
`evaluation/results/prerecorded_q64_demo/prerecorded_q64_demo_readiness.json`.
It reports the model path, input record id, raw prediction, normalized gloss,
visible gloss, inference mode, validity, and confidence availability.

For CI or dependency-light contract testing, add `--mock`. Mock mode exercises
the same q64 contract and artifact path without loading the checkpoint.
