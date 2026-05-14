# HF Containerization Handoff — 2026-05-13

## Purpose
This document is a handoff package for the next agent to continue work from the current local integration baseline to the target deployment direction:

- **Target direction:** one HF custom endpoint container that runs the full ASL backend pipeline path for `/v1/translate-sign`
- **User intent:** prioritize containerizing the real path (backend scripts + model runtime path) and validate it in containerized/deployed conditions

---

## Executive Summary
We completed a local integration repair and validation phase, then prepared a deterministic smoke harness.

### What is now working
- Local chain with stub backend:
  - `cloud API (:8000)` -> `HF endpoint service (:9000)` -> response payload
  - deterministic smoke run returns PASS
- Contract mismatch between middle layer and endpoint route was fixed.

### What is not yet complete
- Real model runtime proof in a deployment-like container with all required dependencies/assets.
- End-to-end validation on actual hosted HF endpoint using the intended container image/runtime assumptions.

---

## Architecture Map (Zoomed Out, Domain Vocabulary)

### 1) Ingress/Adapter Layer
- `src/fastapi_apps.py`
  - FastAPI adapter converting request to WSGI environ
  - Routes all methods/paths to underlying WSGI handlers

### 2) Core Translation Pipeline (`/v1/translate-sign`)
- `src/cloud_translate_api.py`
  - API key auth/rate-limit
  - multipart video parse
  - **video ingest** (normalize/probe)
  - **frame extraction**
  - **pose extraction**
  - **inference handoff**
  - response normalization + proof fields

### 3) HF Endpoint Service (B path)
- `src/hf_custom_endpoint_service.py`
  - exposes `/v1/translate-sign` and `/v1/chat/completions`
  - supports `stub` and `real` backend modes
  - wraps/adapts runtime backend behavior for endpoint-compatible serving

### 4) Media/Pose Pipeline Modules
- `src/video_ingest.py`
- `src/frame_extraction.py`
- `src/pose_handoff.py`

### 5) Runtime Entrypoints
- `scripts/runtime/serve_cloud_translate_api.py` (local :8000)
- `scripts/runtime/serve_hf_custom_endpoint_service.py` (local :9000)

### 6) Validation Layer
- `tests/runtime/test_cloud_translate_api.py`
- `tests/runtime/test_hf_custom_endpoint_service.py`
- `scripts/smoke/run_local_chain_smoke.sh`

---

## Implemented Changes in This Session Window

## A) Contract Fix: cloud->endpoint handoff
**File:** `src/cloud_translate_api.py`

### Change
`_default_cloud_infer(...)` now switches request shape by upstream URL path:
- If endpoint path ends with `/v1/translate-sign` -> send `multipart/form-data` with `video` file part.
- Else preserve legacy JSON/base64 payload path.

### Why
This fixed the previous mismatch where `cloud_translate_api` sent JSON to an endpoint route that expects multipart video.

---

## B) Tests Added/Updated
**File:** `tests/runtime/test_cloud_translate_api.py`

### Added
- `test_default_cloud_infer_uses_multipart_for_translate_sign_endpoint`
  - validates content-type boundary
  - validates file part metadata/name
  - validates payload contains raw video bytes

---

## C) Deterministic smoke runner (Option A)
**File:** `scripts/smoke/run_local_chain_smoke.sh`

### Features
- venv check + source
- required env validation
- token fail-fast when backend=real
- port availability checks
- launches both services
- readiness checks
- chain request execution
- PASS/FAIL output
- artifact/log outputs under `/tmp/asl-smoke`

### Important correction applied
- Cloud API readiness check cannot rely on `/healthz` (no dedicated route in `cloud_translate_app`)
- script was corrected to check reachability via `/v1/translate-sign`

---

## Current Evidence Snapshot

### Local stub chain
- Script run succeeded (`PASS: local chain smoke succeeded`) with full payload fields.

### Real backend readiness
- HF side can report ready when token is set.
- Prior failures were often auth/runtime configuration related, not transport contract related.

---

## Known Risks / Review Notes

1. URL-path-based contract detection is pragmatic but heuristic.
   - It assumes `/v1/translate-sign` implies multipart contract.
2. Smoke script validates integration behavior but not full production parity.
3. Model/runtime dependency packaging for deployed container remains unproven in this handoff.

---

## What the User Wants Next (Priority)
Move from local wiring/debug into **container-first proof**:

- Ensure container includes required backend runtime/scripts/deps for full `/v1/translate-sign` flow.
- Include/resolve Gemma model runtime path strategy.
- Validate in containerized and hosted endpoint conditions.

---

## Next-Agent Execution Plan (Concrete)

### Phase 1 — Container gap audit
1. Inspect:
   - `deploy/hf-endpoint/Dockerfile`
   - endpoint entrypoint/start command
   - python deps (`requirements.txt` + optional native deps)
2. Cross-check against required runtime modules:
   - video ingest path
   - frame extraction path (ffmpeg/opencv dependencies)
   - pose extraction path (mediapipe and any native requirements)
   - HF inference client dependencies and auth env requirements
3. Produce missing dependency matrix (present/missing/uncertain).

### Phase 2 — Build container for full pipeline
1. Patch Dockerfile to include all required runtime deps.
2. Ensure runtime envs are explicitly mapped/documented:
   - `ASL_HF_ENDPOINT_BACKEND`
   - `ASL_HF_ENDPOINT_MODEL_ID`
   - `ASL_HF_ENDPOINT_MODEL_VERSION`
   - `ASL_HF_TOKEN`
   - auth keys for `/v1/translate-sign`
3. Build image locally and run container with explicit ports.

### Phase 3 — Container-local proof
1. Execute smoke against container endpoint (not host-run python scripts).
2. Capture:
   - health
   - request/response payload
   - logs
3. Verify proof fields and semantic response expectations.

### Phase 4 — Hosted HF endpoint proof
1. Push image tag.
2. Deploy/update HF endpoint config.
3. Run hosted smoke curl.
4. Capture final evidence bundle.

---

## Acceptance Criteria for Handoff Completion

1. **Container parity:** `/v1/translate-sign` path works from inside deployed container runtime, not just local script mode.
2. **Real backend proof:** backend=real returns successful, non-error output for smoke clip.
3. **Evidence artifacts:** command logs + response JSON + health output preserved in docs/evidence.
4. **No contract regressions:** existing runtime tests still pass.

---

## Recommended Operator Commands (for next agent)

```bash
# Local regression guard
python -m pytest tests/runtime/test_cloud_translate_api.py -q

# Deterministic local chain smoke (stub/real based on env)
bash scripts/smoke/run_local_chain_smoke.sh

# Real mode precondition
export ASL_HF_TOKEN='hf_...'
```

---

## Files Most Relevant to Continue

- `src/cloud_translate_api.py`
- `src/hf_custom_endpoint_service.py`
- `src/fastapi_apps.py`
- `tests/runtime/test_cloud_translate_api.py`
- `tests/runtime/test_hf_custom_endpoint_service.py`
- `scripts/smoke/run_local_chain_smoke.sh`
- `deploy/hf-endpoint/Dockerfile`
- `scripts/runtime/serve_hf_custom_endpoint_service.py`

---

## Branch/State Notes
- Branch in use during this work: `feat/asl-cactus-custom-endpoint`
- Working tree has other modified/untracked files beyond this doc; next agent should scope edits carefully.

---

## One-line handoff prompt for next agent
"Start at container gap audit for `deploy/hf-endpoint/Dockerfile`, then implement full `/v1/translate-sign` runtime dependency parity in the HF image, validate locally in-container, and then validate on hosted HF endpoint with backend=real evidence artifacts."

---

## 2026-05-14 Execution Log (Container-first + Hosted Debug)

### What was done next (and why)

1. Added endpoint-runtime dependency parity to custom container image.
   - Why: `/v1/translate-sign` full path requires ffmpeg/ffprobe + cv2 + mediapipe, not just API server deps.

2. Patched MediaPipe runtime compatibility for linux/amd64 image behavior.
   - Why: hosted run previously failed with `POSE_EXTRACTION_UNAVAILABLE` / `mediapipe ... solutions` mismatch.

3. Added structured upstream inference error surfacing from HF endpoint local infer seam.
   - Why: previous generic `UPSTREAM_FAILURE` hid root cause and blocked targeted fix.

### Concrete code/container changes

- `deploy/hf-endpoint/Dockerfile`
  - Added apt packages: `ffmpeg`, `libglib2.0-0`, `libgl1`, `libsm6`, `libxext6`, `libxrender1`.
  - Switched to runtime-specific req file.

- `deploy/hf-endpoint/requirements.runtime.txt` (new)
  - Runtime deps only: `fastapi`, `uvicorn`, `huggingface_hub`, `numpy`, `python-multipart`.
  - Pinned compatibility versions:
    - `opencv-python-headless==4.10.0.84`
    - `mediapipe==0.10.14`

- `src/data/pose_extractor.py`
  - Added compatibility fallback for locating MediaPipe solutions module.

- `src/hf_custom_endpoint_service.py`
  - `_build_local_cloud_infer` now raises structured `CloudInferError` with exact upstream exception class/message.
  - Empty-output inference now mapped as structured `INFERENCE_FAILED` instead of opaque runtime failure.

### Build + deploy evidence

- Built/pushed amd64 image tags:
  - `alexd05/asl-hf-custom-endpoint:real-v5`
  - `alexd05/asl-hf-custom-endpoint:real-v6`

- Active deployed image after latest update:
  - `alexd05/asl-hf-custom-endpoint:real-v6`

- Endpoint:
  - Name: `asl-gemma4-e2b-q64-top50-mer-ohp`
  - URL: `https://ycd984e4ou9u6p8s.us-east-1.aws.endpoints.huggingface.cloud`

### Hosted validation outcomes

- `GET /healthz`:
  - `ready=true`, `backend=real`, `load_error=""`.

- `POST /v1/translate-sign` (with both headers: HF bearer + X-API-Key):
  - Now reaches inference and returns structured upstream error (not transport/container error):
  - Root cause extracted from response:
    - `model_not_supported` on chat surfaces for `AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit`
    - `text_generation` fallback ends in `StopIteration`

### Current blocker (precise)

Containerization goal met (ingest/frame/pose stack runs in hosted endpoint runtime), but real-model completion still blocked by HF InferenceClient provider/task behavior for this model id:

- Chat methods reject model as non-chat (`model_not_supported`).
- Text-generation fallback path currently resolves to `StopIteration` in hosted provider routing.

### Artifact paths (latest run)

- `/tmp/asl-smoke/hosted_health_v6.json`
- `/tmp/asl-smoke/hosted_translate_v6.json`

### Security note

The HF token was exposed in interactive commands during debugging. Rotate token immediately and update endpoint secret/env with new token value before further runs.
