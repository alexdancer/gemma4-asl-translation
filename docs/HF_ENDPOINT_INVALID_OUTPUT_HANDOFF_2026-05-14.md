# HF Endpoint Invalid Output Bug Handoff — 2026-05-14

## TL;DR
Phone app path works, container deploy works, endpoint live, but translation still fails with:
- `INFERENCE_FAILED: invalid model output`

This is now a **model-output contract bug**, not transport/auth/startup.

---

## Current user-visible symptom
From RN app (local backend bridge):
- error shown as `INFERENCE_FAILED: invalid model output`

From hosted endpoint logs:
- `{"event": "cloud_infer_error", ... "error": "invalid model output", "code": "INFERENCE_FAILED"}`
- `POST /v1/translate-sign ... 502 Bad Gateway`

---

## Proven working components
1. RN app reaches local FastAPI bridge (`:8000`).
2. Local bridge reaches hosted HF endpoint.
3. Hosted custom container starts cleanly and loads model weights.
4. Video ingest/frame extraction/pose path executes.

So pipeline connectivity is good.

---

## What was changed in code

### 1) Upstream error passthrough (improved observability)
**File:** `src/cloud_translate_api.py`
- Added `HTTPError` handling that reads upstream JSON body and propagates:
  - `error_code`
  - `message`
  - `retryable`
  - `details`
- This replaced prior generic-only `UPSTREAM_FAILURE` text.

### 2) Timeout wiring for RN local bridge
**Files:**
- `src/fastapi_apps.py`
- `scripts/dev/run_mobile_stack.sh`
- `tests/runtime/test_fastapi_apps.py`
- Adapter now supports env timeout (`ASL_CLOUD_TIMEOUT_SECONDS`) and passes it to WSGI app.

### 3) Fail-closed invalid output guardrails
**File:** `src/hf_custom_endpoint_service.py`
- Reject filename echoes and structured/telemetry blobs (`pose_summary`, `frame_count`, etc).
- Reject empty model output.

### 4) Controlled salvage parser (latest)
**File:** `src/hf_custom_endpoint_service.py`
- If invalid output is structured JSON, attempt safe salvage from:
  - `{"translation": "..."}` only
- Normalize `Translation:`/`Output:` prefixes.
- Revalidate after salvage.
- If salvage fails, still return `INFERENCE_FAILED`.

---

## Deployed images/tags during debugging
- `real-v11-amd64`
- `real-v12-amd64`
- `real-v13-amd64`
- `real-v14-amd64` (current)

Current endpoint confirms running on:
- `alexd05/asl-hf-custom-endpoint:real-v14-amd64`

Digest for v14 manifest list:
- `sha256:5870d957777747de597d533388453ffb1eccc43d3990baf91d5d2ae7bf716e16`

---

## Key runtime clues from logs
- Transformers warning references tokenizer regex issue for Mistral family from `/repository`.
- Generation warning shows some flags ignored (e.g. `temperature`).
- Endpoint still emits `invalid model output` after stricter prompting/salvage.

Interpretation:
- Model/runtime artifact behavior likely not aligned with expected output contract.
- Could be artifact mismatch, prompt mismatch, or generation parameter incompatibility.

---

## Tests status before handoff
- `tests/runtime/test_cloud_translate_api.py` -> pass (32)
- `tests/runtime/test_hf_custom_endpoint_service.py` -> pass (18)

These confirm code behavior as implemented; they do **not** guarantee model semantic quality.

---

## Suggested next-agent plan

### Phase A — Capture raw model text safely
1. Add temporary debug field/log behind env flag (redacted, capped length) to capture pre-validation model output.
2. Run one failing request and preserve exact raw output artifact.

### Phase B — Runtime/model alignment
1. Verify exact model files in `/repository` correspond to intended ASL Gemma artifact.
2. Confirm tokenizer/model class pairing and load args (`fix_mistral_regex=True` if applicable).
3. Remove/adjust ignored generation params; verify valid args for loaded architecture.

### Phase C — Contract strategy decision
Choose one:
- Strict sentence-only output (keep fail-closed, improve prompt/inference until compliant), or
- Structured contract (`{"translation": "..."}`) as primary output and parse deterministically.

### Phase D — Revalidate in HITL
1. Direct hosted curl with sample clip.
2. Local bridge curl (`:8000`).
3. RN device test.
4. Save evidence bundle with request IDs and response bodies.

---

## Repro command (hosted direct)
```bash
curl -i 'https://ycd984e4ou9u6p8s.us-east-1.aws.endpoints.huggingface.cloud/v1/translate-sign' \
  -H "Authorization: Bearer $ASL_HF_TOKEN" \
  -H "X-API-Key: dev-local-key-1" \
  -F "video=@/tmp/asl-smoke/sample.mp4;type=video/mp4"
```

## Repro command (local bridge)
```bash
curl -i 'http://127.0.0.1:8000/v1/translate-sign' \
  -H "Authorization: Bearer dev-local-key-1" \
  -F "video=@/tmp/asl-smoke/sample.mp4;type=video/mp4"
```

---

## Bottom line for handoff
- Infra path is up.
- Failure narrowed to model output validity at hosted inference stage.
- Latest code now surfaces true root error and blocks fake-success responses.
