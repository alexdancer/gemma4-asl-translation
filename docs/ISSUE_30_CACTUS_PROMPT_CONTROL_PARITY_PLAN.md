# Issue #30 — Cactus Prompt-Control Parity Harness Plan

## Current decision

Issue #30 should build the **parity harness and report contract now**, with Cactus runtime execution behind a real-run seam. We do **not** currently have a Cactus-converted weights folder, so CI/dev tests must use mocked Cactus outputs and must not claim real Cactus parity.

The purpose is not to re-measure whether the model learned ASL. The Python/Unsloth evaluation path already answers that. The purpose is to check whether a Cactus-converted/runtime version preserves the validated Python prompt-control behavior on the same q64 inputs.

Mental model:

```text
Python eval       = did the model learn?
Video/q64 smoke   = can real inputs become valid q64?
Cactus parity     = did the Cactus runtime preserve Python behavior?
```

If we later upload/convert the current fine-tuned Gemma model to Cactus, this harness compares Cactus normalized output against the Python reference fixture. It answers runtime-boundary questions like prompt/tokenizer/export/decoding drift, not full test-set accuracy unless we expand it to more samples.

## Cactus docs grounding

- Use **Cactus Engine / Python SDK** for prompt-control text generation, not Cactus Graph. Graph is lower-level tensor compute; this issue compares LLM completion behavior.
- Python SDK mirrors the C FFI: initialize a model, call completion, then destroy the model.
- `cactus_complete` accepts a JSON chat-message array and returns JSON containing fields such as `success`, `error`, `response`, timing, token counts, and RAM usage.
- Fine-tuned LoRA adapters must first be converted into a Cactus weights directory using `cactus convert <base-model> <out-dir> --lora <adapter>` before a real parity run is possible.

Docs:
- https://docs.cactuscompute.com/latest/python/
- https://docs.cactuscompute.com/latest/docs/cactus_engine/
- https://docs.cactuscompute.com/latest/docs/finetuning/
- https://docs.cactuscompute.com/latest/docs/compatibility/

## Out of scope for this issue

- Producing real Cactus-converted weights.
- Proving real device/mobile parity.
- Building mobile UI integration.
- Using Cactus Graph.

Those are follow-up work once a converted weights folder exists.

## Proposed implementation files

- `src/mobile/cactus_prompt_control_parity.py`
- `scripts/mobile/run_cactus_prompt_control_parity.py`
- `tests/export/test_cactus_prompt_control_parity.py` or `tests/runtime/test_cactus_prompt_control_parity.py`
- Optional docs update in `docs/IMPLEMENTATION_REFERENCE.md`

## Harness behavior

1. Load prompt-control reference fixture from issue #26, normally:
   - `evaluation/results/prompt_control_reference/reference.json`
2. Select smoke sample first (`selection_role == "smoke"`), with `--max-samples` support for later expansion.
3. Load q64 record for each selected `sample_id` from the known q64 records JSONL.
4. Build the same prompt-control prompt used by Python evaluation via `build_prompt_control_prompt(record, labels)`.
5. Send prompt to a `CactusPromptRunner` seam.
6. Normalize Cactus raw output with the same `normalize_model_output(raw_output, labels)` helper used by Python prompt-control evaluation.
7. Compare Cactus behavior against the Python reference fixture by:
   - normalized gloss equality
   - valid-label status equality
   - expected gloss correctness
8. Write a separate scoped parity report.

## Runner seam

Use a protocol/dataclass boundary so tests do not depend on Cactus installation.

```python
class CactusPromptRunner(Protocol):
    runtime_mode: str

    def complete(self, prompt: str, *, sample_id: str) -> CactusCompletionResult:
        ...
```

`CactusCompletionResult` should record:
- raw response text used for normalization
- raw Cactus response JSON when available
- success/error
- timing/token/RAM metadata when available

### Mock runner

Used in tests and CLI with `--mock-cactus-output`.

### Real runner

Uses Cactus Python SDK if installed:

```python
from src.cactus import cactus_init, cactus_complete, cactus_destroy
```

Expected lifecycle:

1. Validate `--cactus-weights` exists and is a directory.
2. `model = cactus_init(str(weights), None, False)`
3. `messages = json.dumps([{"role": "user", "content": prompt}])`
4. `options = json.dumps({"temperature": 0.0, "max_tokens": 8})`
5. `result = json.loads(cactus_complete(model, messages, options, None, None))`
6. Use `result["response"]` as raw model output.
7. Always `cactus_destroy(model)` in `finally`.

If Cactus SDK import/init/completion fails, write a report row with `runtime_error` instead of crashing without artifact.

## Report artifact

Default path:

```text
evaluation/results/cactus_prompt_control_parity/parity_report.json
```

Top-level fields:
- `scope`: `cactus_prompt_control_parity`
- `runtime_mode`: `mock` or `cactus_engine`
- `real_cactus_parity_proven`: `true` only when `runtime_mode == "cactus_engine"` and all selected samples match
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
- `python_reference.raw_model_output`
- `python_reference.normalized_gloss`
- `python_reference.valid_label`
- `cactus.raw_model_output`
- `cactus.normalized_gloss`
- `cactus.valid_label`
- `normalized_gloss_matches_python`
- `valid_label_matches_python`
- `correct_matches_expected`
- `runtime_error`
- `cactus_response_metadata`

## CLI shape

```bash
PYTHONPATH=. ./venv/bin/python scripts/mobile/run_cactus_prompt_control_parity.py \
  --reference evaluation/results/prompt_control_reference/reference.json \
  --records data/processed/exports/asl_unsloth_pose_train_q64_full_top50_test.jsonl \
  --manifest data/processed/exports/asl_unsloth_pose_train_q64_full_top50_manifest.json \
  --cactus-weights path/to/converted-cactus-weights \
  --out-dir evaluation/results/cactus_prompt_control_parity \
  --max-samples 1
```

CI-safe mock example:

```bash
PYTHONPATH=. ./venv/bin/python scripts/mobile/run_cactus_prompt_control_parity.py \
  --reference tests/fixtures/prompt_control_reference.json \
  --records tests/fixtures/q64_records.jsonl \
  --manifest tests/fixtures/manifest.json \
  --cactus-weights mock-cactus-weights \
  --out-dir /tmp/cactus_prompt_control_parity \
  --max-samples 1 \
  --mock-cactus-output hearing
```

## TDD checklist

1. Failing test: matching mock Cactus output writes pass report.
2. Failing test: mismatched normalized output writes fail report, no crash.
3. Failing test: invalid Cactus output marks invalid label and mismatch.
4. Failing test: runner exception captured as runtime error in report.
5. Failing test/CLI: mock run prints summary and writes artifact.
6. Implement minimal module/CLI.
7. Run targeted tests and full suite.
8. Separate review pass for correctness/bugs.

## Open question

We currently do not have a real Cactus-converted weights folder. Real parity proof should wait until a follow-up issue creates or locates that folder.
