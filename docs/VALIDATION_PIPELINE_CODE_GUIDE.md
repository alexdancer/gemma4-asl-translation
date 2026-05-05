# Validation Pipeline Code Guide

This guide explains the code added across the recent staged validation work: cached pose -> q64 verification, real video -> q64 smoke, Python video prompt-control smoke, and Cactus prompt-control parity harness.

## Big picture

The project is building a careful ladder from known q64 evaluation records toward real video and Cactus runtime validation:

```text
cached/precomputed pose
  -> q64 JSONL contract
  -> real video extraction smoke
  -> Python prompt-control model smoke
  -> Cactus prompt-control parity harness
```

The goal is to isolate failures. If something breaks, we can tell whether the problem is q64 formatting, video/MediaPipe extraction, Python model inference, Cactus runtime behavior, or output normalization.

Think of the stages as answering different engineering questions:

```text
Python eval       = did the model learn the q64 task?
Video/q64 smoke   = can real/prerecorded video become valid q64 input?
Python video smoke = can video-derived q64 run through the validated Python path?
Cactus parity     = does Cactus reproduce the validated Python behavior?
```

The Cactus parity harness is therefore not primarily an accuracy benchmark. It is a runtime-boundary regression test. Once the fine-tuned Gemma model is converted/uploaded to Cactus, the harness checks whether Cactus preserves the same normalized gloss behavior as the Python reference on the same q64 samples.

## Key contract

All stages preserve the same q64 prompt-control shape:

```text
instruction: classify compact ASL pose encoding
input:
  sample_id=<id>
  encoding=q64_full clip=4 alphabet=<alphabet>
  frames=<n> features_per_frame=<m>
  pose_q64=<encoded rows>
output: <expected gloss>
```

Prompt-control inference then normalizes model output using the same helpers from `src/evaluation/unsloth_asl.py`:

- `build_prompt_control_prompt(...)`
- `normalize_model_output(...)`
- `infer_q64_record(...)`

## Issue #27 / #28 foundation: cached/video pose to q64

### Files

- `src/data/cached_pose_q64.py`
- `scripts/data/verify_cached_pose_q64.py`
- `tests/data/test_cached_pose_q64_verification.py`
- `src/data/video_pose_q64_smoke.py`
- `scripts/data/smoke_video_pose_q64.py`
- `tests/data/test_video_pose_q64_smoke.py`

### What it does

`cached_pose_q64.py` turns a cached `.npz` pose archive into a q64 JSONL-compatible record and validates it against a known source q64 record.

`video_pose_q64_smoke.py` does the same basic conversion, but starts from a real video path and a pose extractor seam:

```text
video path -> PoseExtractor.extract_from_video(...) -> pose components -> q64 record -> smoke report
```

### Important design choice: 4D MediaPipe landmarks -> 3D q64 features

MediaPipe landmarks include four values per landmark:

```text
x, y, z, visibility
```

The q64 training/evaluation contract expects three values per landmark:

```text
x, y, z
```

So the video/q64 code trims 4D landmarks down to xyz before q64 encoding. That keeps the feature count at `177`:

```text
body:       17 * 3 = 51
left hand:  21 * 3 = 63
right hand: 21 * 3 = 63
total:              177
```

### Why mocked extractors exist

The CLI tests use fake pose extractors because real MediaPipe/video decoding is heavyweight and environment-dependent. The tests prove orchestration and artifact shape while leaving real extraction as an operational smoke command.

### Main commands

Cached pose:

```bash
PYTHONPATH=. ./venv/bin/python scripts/data/verify_cached_pose_q64.py \
  --pose-path path/to/sample.npz \
  --sample-id hearing_26986 \
  --expected-gloss hearing
```

Video pose q64 smoke:

```bash
PYTHONPATH=. ./venv/bin/python scripts/data/smoke_video_pose_q64.py \
  --video-path path/to/hearing_26986.mp4 \
  --sample-id hearing_26986 \
  --expected-gloss hearing \
  --max-frames 8
```

CI-safe mock command:

```bash
PYTHONPATH=. ./venv/bin/python scripts/data/smoke_video_pose_q64.py \
  --video-path path/to/mock.mp4 \
  --sample-id hearing_26986 \
  --expected-gloss hearing \
  --mock-extractor
```

## Test folder reorganization

Tests were moved from a flat `tests/` folder into domain folders:

```text
tests/data/
tests/demo/
tests/evaluation/
tests/export/
tests/runtime/
tests/training/
```

This makes the suite easier to navigate:

- data conversion/extraction tests -> `tests/data/`
- demo smoke tests -> `tests/demo/`
- evaluator/reference tests -> `tests/evaluation/`
- Cactus export tests -> `tests/export/`
- runtime/parity harness tests -> `tests/runtime/`
- legacy/model training tests -> `tests/training/`

One flaky prerecorded fallback test was stabilized by seeding demo-model training in:

- `scripts/demo/run_prerecorded_fallback.py`

## Issue #29: Python video prompt-control smoke

### Files

- `src/demo/python_video_prompt_control.py`
- `scripts/demo/run_python_video_prompt_control_smoke.py`
- `tests/demo/test_python_video_prompt_control_smoke.py`

### What it does

This is the first end-to-end Python path:

```text
video -> q64 record -> Python prompt-control predictor -> readiness artifact
```

It reuses the video/q64 stage and then runs prompt-control inference through either:

- real `RealUnslothASLGlossPredictor`, or
- mocked predictor for tests.

### Main module flow

`run_python_video_prompt_control_smoke(...)`:

1. Builds a q64 JSONL record from the video with `run_video_pose_q64_smoke(...)`.
2. Loads Top-50 labels from manifest.
3. Builds a prompt-control predictor.
4. Runs `infer_q64_record(...)`.
5. Computes:
   - raw model output
   - normalized gloss
   - expected gloss
   - valid-label status
   - correctness
   - timing metadata
6. Writes:

```text
python_video_prompt_control_smoke_readiness.json
```

### Artifact scope

The artifact is intentionally scoped:

```text
python_video_prompt_control_smoke
```

It does not claim production ASL recognition. It proves the Python video -> q64 -> prompt-control orchestration path.

### Main command

```bash
PYTHONPATH=. ./venv/bin/python scripts/demo/run_python_video_prompt_control_smoke.py \
  --video-path path/to/hearing_26986.mp4 \
  --checkpoint checkpoints/unsloth_gemma-4-E4B-it_q64_full_top50_baseline \
  --sample-id hearing_26986 \
  --expected-gloss hearing \
  --out-dir evaluation/results/python_video_prompt_control_smoke \
  --max-frames 8
```

CI-safe mock command:

```bash
PYTHONPATH=. ./venv/bin/python scripts/demo/run_python_video_prompt_control_smoke.py \
  --video-path path/to/mock.mp4 \
  --checkpoint path/to/mock-checkpoint \
  --sample-id hearing_26986 \
  --expected-gloss hearing \
  --out-dir /tmp/python_video_prompt_control \
  --max-frames 8 \
  --mock-extractor \
  --mock-model-output hearing
```

## Issue #30: Cactus prompt-control parity harness

### Files

- `docs/ISSUE_30_CACTUS_PROMPT_CONTROL_PARITY_PLAN.md`
- `src/mobile/cactus_prompt_control_parity.py`
- `scripts/mobile/run_cactus_prompt_control_parity.py`
- `tests/runtime/test_cactus_prompt_control_parity.py`
- `docs/IMPLEMENTATION_REFERENCE.md`

### What it does

This harness compares Cactus completion behavior against Python prompt-control reference behavior.

Why it exists: Python success does not guarantee Cactus success. Cactus introduces a runtime boundary with possible prompt formatting, tokenizer, adapter/export, decoding, output parsing, and local-vs-cloud execution drift. The harness makes that boundary observable before we claim mobile/runtime parity.

It starts with the prompt-control reference fixture from issue #26 and runs selected q64 samples through a Cactus runner seam:

```text
reference fixture + q64 record
  -> prompt-control prompt
  -> Cactus runner
  -> normalized output
  -> compare against Python reference
  -> parity_report.json
```

### Why Cactus Engine, not Cactus Graph

The Cactus docs distinguish:

- Cactus Engine: LLM/chat/completion runtime.
- Cactus Graph: lower-level tensor compute.

Prompt-control parity is text generation, so this harness targets Cactus Engine semantics.

### Runtime modes

The report distinguishes two modes:

```text
runtime_mode: mock
runtime_mode: cactus_engine
```

`real_cactus_parity_proven` is only true when:

1. `runtime_mode == "cactus_engine"`
2. every selected sample matches
3. no runtime errors occurred
4. Cactus did not hand off to cloud

Mock tests can prove harness behavior, but they cannot prove real Cactus parity.

### Main module pieces

`CactusPromptControlParityConfig`

Holds paths and max sample count:

- reference fixture path
- q64 records path
- manifest path
- Cactus weights path
- output directory
- max samples

`CactusCompletionResult`

Normalized result from any Cactus runner:

- raw output text
- success flag
- error string
- response metadata

`CactusPromptRunner`

Protocol seam used by tests and real runtime:

```python
complete(prompt: str, *, sample_id: str) -> CactusCompletionResult
```

`MockCactusPromptRunner`

Used in tests and CI-safe CLI runs.

`RealCactusEnginePromptRunner`

Operational runner for a future real Cactus-converted weights folder. It:

1. validates weights dir exists
2. imports Cactus Python SDK
3. initializes model
4. calls completion
5. parses Cactus JSON response
6. destroys model in `finally`
7. returns errors as reportable runtime failures

### Important review fixes

The independent review added protections for:

1. **Cloud handoff overclaim**

If Cactus metadata includes:

```json
{"cloud_handoff": true}
```

then local Cactus parity is not proven. The sample becomes a runtime error instead of a pass.

2. **Duplicate q64 sample IDs**

Duplicate `sample_id`s in q64 records now fail clearly instead of silently overwriting.

3. **Null Cactus response**

A `null` response is handled as an empty string so output parsing is stable.

### Report fields

Default report path:

```text
evaluation/results/cactus_prompt_control_parity/parity_report.json
```

Top-level fields:

- `scope`
- `runtime_mode`
- `real_cactus_parity_proven`
- `reference_checkpoint_path`
- `cactus_weights_path`
- `reference_path`
- `records_path`
- `manifest_path`
- `summary`
- `samples`

Per-sample fields:

- `sample_id`
- `selection_role`
- `expected_gloss`
- `python_reference`
- `cactus`
- `normalized_gloss_matches_python`
- `valid_label_matches_python`
- `correct_matches_expected`
- `runtime_error`
- `cactus_response_metadata`

### Main command

Mock / CI-safe:

```bash
PYTHONPATH=. ./venv/bin/python scripts/mobile/run_cactus_prompt_control_parity.py \
  --reference evaluation/results/prompt_control_reference/reference.json \
  --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --cactus-weights mock-cactus-weights \
  --out-dir /tmp/cactus_prompt_control_parity \
  --max-samples 1 \
  --mock-cactus-output hearing
```

Future real Cactus run, once converted weights exist:

```bash
PYTHONPATH=. ./venv/bin/python scripts/mobile/run_cactus_prompt_control_parity.py \
  --reference evaluation/results/prompt_control_reference/reference.json \
  --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --cactus-weights path/to/converted-cactus-weights \
  --out-dir evaluation/results/cactus_prompt_control_parity \
  --max-samples 1
```

## Tests to run

Full suite:

```bash
PYTHONPATH=. ./venv/bin/pytest -q tests
```

Recent issue #30 focused suite:

```bash
PYTHONPATH=. ./venv/bin/pytest -q \
  tests/runtime/test_cactus_prompt_control_parity.py \
  tests/evaluation/test_prompt_control_reference.py \
  tests/evaluation/test_unsloth_asl_evaluator.py
```

Video/q64 + Python smoke focused suite:

```bash
PYTHONPATH=. ./venv/bin/pytest -q \
  tests/data/test_video_pose_q64_smoke.py \
  tests/demo/test_python_video_prompt_control_smoke.py
```

## Current honest status

What is proven:

- q64 artifact shape and validation logic.
- video/q64 orchestration with mocked extractor in tests.
- Python video prompt-control orchestration with mocked extractor/model in tests.
- Cactus parity comparison/report logic with mocked Cactus output in tests.
- clear failure reporting for missing real Cactus weights/runtime errors.

What is not proven yet:

- Real MediaPipe extraction on the final chosen demo video.
- Real Python checkpoint output on that video.
- Real Cactus Engine parity on converted weights.

Needed follow-up:

1. Produce or locate a Cactus-converted weights folder.
2. Run the operational Cactus parity command without `--mock-cactus-output`.
3. Inspect `parity_report.json` for `real_cactus_parity_proven: true`.
