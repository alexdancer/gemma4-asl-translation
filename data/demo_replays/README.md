# Demo Replay Scripts

Fallback B bypasses live capture, feature extraction, and model inference. Use
it only when the demo needs a guaranteed coherent ending:

```bash
python scripts/demo/run_precomputed_replay.py \
  --replay-path data/demo_replays/fallback_b_judge_demo.json \
  --no-sleep
```

The command activates instantly for test drills and prints `mode: replay` plus
the scripted confidence-aware outputs. Add `--sleep` when presenters want the
replay to follow each step's `at_ms` timing during the stage narrative.
