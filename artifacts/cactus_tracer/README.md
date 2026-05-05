# Cactus tracer slice artifacts (issue #32)

This folder contains one local end-to-end tracer run for the frozen baseline checkpoint.

## Files

- `frozen_baseline_metadata.json`
  - Captures checkpoint id/path, git SHA, and conversion output version.
- `converted_weights/v1/`
  - Versioned conversion output directory (`v1`) with `VERSION` marker.
  - `conversion_manifest.json` records conversion mode and status.
- `local_completion_v1.json`
  - One completion artifact including `response`, `timing_ms`, `success`, and `error` fields.
- `run_summary.json`
  - Pointers to all generated tracer artifacts and status summaries.

## Runtime mode notes

If Cactus runtime bindings are unavailable, the tracer emits a deterministic fallback completion (`runtime_mode: deterministic_fallback`) while still preserving the required artifact contract.
