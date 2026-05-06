# Engineering History and Implementation Notes (Top-50 -> Cactus -> iOS)

As of 2026-05-05 (America/Chicago).

## 1) What the system does right now

This repo now supports a staged, evidence-first pipeline:

1. **Evaluate model behavior on shared q64 ASL contract** (primary proof path).
2. **Freeze deterministic artifacts** for reproducibility and demo-readiness.
3. **Run parity checks at runtime boundaries** (Python reference vs Cactus path).
4. **Generate iOS tracer artifacts** that prove the UI-trigger-to-inference contract, even when full iOS/Cactus runtime is not available in Linux CI.
5. **Validate submission-lock package and freeze-gate readiness** with strict machine-checkable rules.

In practical terms: we can prove what is working, what is mocked, and what is still pending device/runtime validation without over-claiming.

---

## 2) SWE architecture and code seams

### Core evaluation and normalization contract
- `src/evaluation/unsloth_asl.py`
  - shared q64 record loading
  - label normalization/validation
  - prompt-control prompt builder
  - normalized output comparison helpers

This module is the canonical behavior contract used by both evaluation and parity layers.

### Runtime boundary tracers
- `src/mobile/cactus_prompt_control_parity.py` (Issue #30)
  - compares Cactus completion behavior against Python reference fixtures
  - writes `evaluation/results/cactus_prompt_control_parity/parity_report.json`
  - uses **runner seam** (`CactusPromptRunner`) for mock and real Cactus execution
- `src/mobile/cactus_tracer_slice.py` (Issue #32)
  - freeze metadata + conversion manifest + local completion + run summary
  - strict acceptance now requires real export + cactus_engine success
- `src/mobile/parity_tracer_slice.py` (Issue #34)
  - one smoke-sample Python-vs-Cactus parity report path
  - dual-runner design (`ParityPromptRunner`) for deterministic testing
- `src/mobile/ios_tracer_slice.py` (Issue #35)
  - records button-tap -> local inference -> UI payload contract
  - writes `artifacts/ios_tracer_slice/local_inference_result.json`

### Script adapters (CLI entry points)
- `scripts/mobile/run_cactus_prompt_control_parity.py`
- `scripts/mobile/run_cactus_tracer_slice.py`
- `scripts/mobile/run_parity_tracer_slice.py`
- `scripts/mobile/run_ios_tracer_slice.py`
- `scripts/release/validate_submission_lock.py` (Issue #44)

### Test surface
- `tests/runtime/test_cactus_prompt_control_parity.py`
- `tests/test_cactus_tracer_slice.py`
- `tests/test_parity_tracer_slice.py`
- `tests/runtime/test_ios_tracer_slice.py`
- `tests/release/test_submission_lock_validator.py`

The pattern is consistent: deterministic artifact outputs + explicit success/failure semantics + reportability under failure.

---

## 3) Issue-by-issue implementation ledger

## Issue #18 — prerecorded Top-50 demo path
**Implementation goal:** provide a known-good demo path using shared q64 contract.

**What it does:** allows controlled demo execution without relying on unstable live capture conditions.

---

## Issue #22 — prompt/output-control guardrails
**Implementation goal:** harden prompt-control comparison rules.

**What it does:** enforces stricter output normalization/comparison behavior so prompt-control diagnostics are dependable.

---

## Issue #23 — demo-safe constrained Top-50 inference
**Implementation goal:** add constrained diagnostic/demo path.

**What it does:** provides optional constrained label selection for safer demos; this is diagnostic/fallback, not primary proof metric.

---

## Issue #26 — prompt-control reference fixture
**Implementation goal:** freeze a stable Python reference artifact for parity checks.

**What it does:** creates a deterministic reference (`reference.json`) for smoke/demo sample comparisons.

---

## Issue #27 — cached pose q64 verification
**Implementation goal:** verify external cached pose archives obey the q64 shape contract.

**What it does:** checks sample compatibility (including frame/feature shape expectations) before downstream evaluation.

---

## Issue #30 — Cactus prompt-control parity harness
**Implementation goal:** compare Cactus runtime behavior against frozen Python prompt-control references.

**What it does:**
- selects smoke/demo samples from reference fixture
- rebuilds prompt using shared prompt builder
- runs Cactus (mock or real)
- normalizes outputs with same label normalization logic
- reports per-sample match/mismatch and runtime errors

**Important semantics:** if runtime is mock or runtime errors occur, report still writes and explicitly avoids claiming proven real Cactus parity.

---

## Issue #32 — Cactus tracer slice
**Implementation goal:** create end-to-end proof artifact chain for baseline freeze -> export -> completion.

**What it does:** writes:
- frozen baseline metadata (`frozen_baseline_metadata.json`)
- conversion manifest (`converted_weights/v1/conversion_manifest.json`)
- local completion artifact (`local_completion_v1.json`)
- run summary (`run_summary.json`)

**Acceptance semantics (strict):** success only if real export succeeded and runtime completed via `cactus_engine` successfully.

---

## Issue #34 — parity tracer slice
**Implementation goal:** one-sample focused parity gate between Python and Cactus paths.

**What it does:**
- selects one smoke sample
- runs both Python reference and Cactus runner on same prompt
- compares normalized gloss validity and expected-gloss correctness
- writes `artifacts/cactus_tracer/parity_report_v1.json`

**Outcome semantics:** CLI exits non-zero if `summary.all_matches` is false.

---

## Issue #35 — iOS tracer slice #1
**Implementation goal:** prove button->local inference->UI contract on iOS scaffold.

**What it does now:**
- SwiftUI scaffold includes button and result fields
- local response fixture (`local_cactus_response.json`) feeds deterministic inference payload
- artifact records trigger/action/UI result

**Current limitation:** Linux environment cannot prove Xcode/iPhone runtime behavior; device validation is explicitly marked as follow-up.

---

## Issue #44 — submission lock validator
**Implementation goal:** machine-validate readiness checklist and freeze gates.

**What it does:**
- validates required package input keys
- validates status enums and non-empty values
- rejects placeholder demo-video values when marked ready
- validates gate status/date rules
- writes readiness artifact and returns non-zero on validation errors

---

## 4) Engineering issues encountered and how we fixed them

1. **False-positive success risk in tracer fallback paths**
   - Problem: fallback paths could be interpreted as successful proof.
   - Fix: strict acceptance semantics in Issue #32 (`acceptance_proof_satisfied` now requires real export + real cactus_engine success + no error).

2. **Over-claim risk when Cactus runtime/weights are unavailable**
   - Problem: parity workflows could be mistaken for real-runtime proof when running mock or missing weights.
   - Fix: Issue #30/34 report contracts encode runtime mode and runtime errors; `real_cactus_parity_proven` remains false unless real conditions are met.

3. **Portability/reproducibility drift from absolute paths**
   - Problem: machine-local absolute paths reduce reproducibility.
   - Fix: repo-relative path normalization in artifacts (e.g., checkpoint paths, weights paths).

4. **Submission checklist quality pitfalls**
   - Problem: invalid dates, non-dict structures, missing keys, and placeholder links could pass informal checks.
   - Fix: Issue #44 validator adds strict schema/value checks and explicit failure reporting with tests.

5. **Environment gap for iOS runtime validation**
   - Problem: CI/Linux cannot run Xcode+iPhone execution.
   - Fix: Issue #35 separates deterministic contract proof from device-runtime follow-up and documents the required device verification steps.

---

## 5) Explicit behavior contracts and failure semantics

- **Artifacts are first-class outputs**: each major flow writes JSON artifacts as evidence.
- **Failure is explicit, not silent**: runtime/import/weights/date/schema failures are carried into report fields and/or non-zero exit codes.
- **Mock vs real mode is explicit**: runtime mode is always encoded in reports.
- **Primary metric remains Python free-generation normalized exact-match**: constrained/prompt-control/cactus tracers are diagnostics/parity checks, not replacements for primary proof.

---

## 6) Current status summary

Implemented and tested tracer stack now covers:
- Cactus parity harness contract (#30)
- Cactus tracer artifact chain with strict proof semantics (#32)
- Python-vs-Cactus smoke parity report path (#34)
- iOS UI-trigger tracer scaffold and artifact path (#35)
- Submission lock machine validator and tests (#44)

Remaining proof gaps are explicitly known and tracked:
- real Cactus weights/runtime parity runs
- on-device iPhone runtime validation capture

This is an intentionally honest state: strong deterministic engineering evidence now, with clear boundaries around what is not yet proven in-device/runtime.
