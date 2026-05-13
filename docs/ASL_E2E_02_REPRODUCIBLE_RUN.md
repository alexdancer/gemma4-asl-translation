# ASL-E2E-02 Reproducible RN -> FastAPI -> Cactus -> HF Happy Path

Issue: #84

## Goal

Provide a reproducible operator flow that validates the RN-facing API path:

- RN upload target: `/v1/translate-sign`
- Runtime chain: FastAPI local API -> Cactus hybrid -> HF dev custom endpoint
- Required proof fields in success response:
  - `runtime_mode`
  - `cloud_handoff`
  - `model_id`
  - `model_version`

## Prerequisites

- Local stack can run with:
  - `scripts/dev/run_mobile_stack.sh`
- A valid local test video clip (mov/mp4) to upload.
- `jq` installed.

Required env mapping:

- RN/client -> FastAPI auth: `x-api-key` must match one key in `ASL_V1_API_KEYS`
- FastAPI -> Cactus auth: `ASL_CLOUD_API_KEY` must equal Cactus `ASL_CACTUS_SERVICE_API_KEY`
- Cactus -> HF auth: `ASL_HF_TOKEN`
- Cactus HF base: `ASL_HF_OPENAI_BASE_URL`

## Repro flow

1) Start local stack (simulator or device mode):

```bash
bash scripts/dev/run_mobile_stack.sh simulator
```

2) In a second terminal, run the reproducible happy-path verifier:

```bash
VIDEO_PATH=/absolute/path/to/clip.mov \
API_URL=http://127.0.0.1:8000/v1/translate-sign \
API_KEY=dev-local-key-1 \
bash scripts/runtime/verify_rn_fastapi_cactus_hf_e2e.sh
```

For physical device runs (same backend API path), use Mac LAN API URL:

```bash
VIDEO_PATH=/absolute/path/to/clip.mov \
API_URL=http://<MAC_LAN_IP>:8000/v1/translate-sign \
API_KEY=dev-local-key-1 \
bash scripts/runtime/verify_rn_fastapi_cactus_hf_e2e.sh
```

## Expected pass signals

- Verifier prints response JSON.
- Response contains non-empty `prediction`.
- `cloud_handoff` is `true`.
- Proof fields exist: `runtime_mode`, `cloud_handoff`, `model_id`, `model_version`.
- `request_id` in response matches submitted `x-request-id`.

## Contract-regression test evidence

Run targeted runtime tests:

```bash
.venv/bin/python -m pytest -q \
  tests/runtime/test_e2e_rn_fastapi_cactus_hf_chain.py \
  tests/runtime/test_fastapi_apps.py \
  tests/runtime/test_cactus_hybrid_service.py \
  tests/runtime/test_cloud_translate_api.py
```

These tests validate the RN-facing API route, Cactus proof enforcement/fail-closed behavior, and end-to-end chain behavior using deterministic seams.
