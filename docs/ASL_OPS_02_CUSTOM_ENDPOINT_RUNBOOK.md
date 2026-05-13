# ASL-OPS-02 Runbook: HF Custom Endpoint + Cactus Hybrid (Dev -> Post-Green Hardening)

Issues: #82, #86
Parent: #80

## 1) Locked request path and ownership

Use this fixed path:

RN app -> local FastAPI `/v1/translate-sign` -> Cactus hybrid service -> HF custom endpoint `/v1/chat/completions`

Ownership boundaries:
- RN app: upload and display only (no manual endpoint/key entry in normal flow)
- FastAPI local API (`serve_cloud_translate_api.py`): ingest/frame/pose extraction and contract normalization
- Cactus hybrid (`serve_cactus_hybrid_service.py`): routing authority, prompt shaping, fail-closed upstream handling
- HF custom endpoint container: model-serving adapter with OpenAI-chat-compatible surface

## 2) Exact environment-variable map (and equality constraints)

### Local API / FastAPI side
- `ASL_V1_API_KEYS`: API keys accepted from RN callers (`x-api-key` on `/v1/translate-sign`)
- `ASL_CLOUD_INFER_URL`: URL for Cactus hybrid service (usually `http://127.0.0.1:9000/` for local stack)
- `ASL_CLOUD_API_KEY`: bearer token FastAPI sends to Cactus

### Cactus hybrid side
- `ASL_CACTUS_SERVICE_API_KEY`: bearer token Cactus expects from FastAPI callers
- `ASL_HF_OPENAI_BASE_URL`: HF endpoint OpenAI-compatible base URL (dev endpoint first; typically ends with `/v1`)
- `ASL_HF_TOKEN`: static bearer token for Cactus -> HF handoff
- `ASL_CACTUS_MODEL_VERSION`: model version label emitted in proof fields
- `ASL_HF_ROUTE_MODE`: route mode (`chat`, `auto`, or `completion`; this runbook locks to `chat` for first-cut path)

### Equality constraints you must preserve
- FastAPI -> Cactus auth must match:
  - `ASL_CLOUD_API_KEY == ASL_CACTUS_SERVICE_API_KEY`
- RN caller key must be present in FastAPI allow-list:
  - submitted `x-api-key` ∈ `ASL_V1_API_KEYS`

## 3) Dev endpoint deploy/update workflow (operator)

### 3.1 Create/update HF dev endpoint (custom container)

In Hugging Face Inference Endpoints:
1. Select custom container deployment.
2. Set exposed API contract to include:
   - `GET /healthz`
   - `POST /v1/chat/completions`
3. Set/confirm endpoint environment variables and secrets for model load.
4. Deploy to a dev-only endpoint first.
5. Record:
   - dev endpoint base URL
   - endpoint revision/image tag
   - deployment timestamp

### 3.2 Point Cactus to the dev endpoint

Set Cactus runtime env:
- `ASL_HF_OPENAI_BASE_URL=<HF_DEV_ENDPOINT_BASE_URL>` (must be OpenAI-compatible base; Cactus appends `/chat/completions`)
- `ASL_HF_TOKEN=<HF_DEV_BEARER_TOKEN>`
- `ASL_HF_ROUTE_MODE=chat` (or locked runtime mode for the release)
- `ASL_CACTUS_MODEL_VERSION=<version-label>`

Restart service/process after env updates.

### 3.3 Keep FastAPI/Cactus bridge stable

Do not change RN route contract while updating endpoint wiring:
- Keep RN target as `/v1/translate-sign`
- Keep FastAPI -> Cactus URL/key mapping intact

## 4) Verification commands (health, contract, proof)

> These are executable operator checks. Replace placeholders before running.
>
> Variable convention used below:
> - `HF_BASE_URL` = endpoint origin **without** `/v1` suffix (example: `https://<endpoint-host>`)
> - chat contract check appends `/v1/chat/completions` explicitly.

### 4.1 HF endpoint health/readiness

```bash
curl -sS "$HF_BASE_URL/healthz" | jq -e '.ok == true and .ready == true'
```

Pass signal:
- command exits `0`.

### 4.2 HF endpoint chat contract (minimal OpenAI shape)

```bash
curl -sS -X POST "$HF_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $ASL_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
    "messages": [
      {"role": "system", "content": "Return only uppercase gloss text."},
      {"role": "user", "content": "pose_summary: frame_count=16"}
    ]
  }' | jq -e '.choices[0].message.content != null and .choices[0].message.content != ""'
```

Pass signal:
- command exits `0`.

### 4.3 End-to-end proof fields at RN-facing API boundary

Preferred scripted verifier:

```bash
VIDEO_PATH=/absolute/path/clip.mov \
API_URL=http://127.0.0.1:8000/v1/translate-sign \
API_KEY=dev-local-key-1 \
bash scripts/runtime/verify_rn_fastapi_cactus_hf_e2e.sh
```

Pass signals:
- `request_id` matches submitted request header.
- `prediction` exists.
- Proof fields exist and are populated:
  - `runtime_mode`
  - `cloud_handoff` (expected `true` on cloud handoff success)
  - `model_id`
  - `model_version`

### 4.4 Fail-closed behavior check

> This script assumes a systemd host with `sudo` access and env file at `/etc/asl/cactus-hybrid.env`.
> For pure local dev without systemd, use §4.3 scripted verifier and targeted runtime tests.

Use existing proof script:

```bash
bash scripts/runtime/prove_cactus_hybrid.sh
```

Pass signals:
- Normal success response includes required proof fields.
- Broken-token step returns fail-closed upstream error.
- Script restores env and restarts service after test.

## 5) Post-green hardening checklist

After first green E2E in dev endpoint:

### 5.1 Token rotation approach
- Rotate `ASL_HF_TOKEN` on a regular cadence (time-based and incident-based).
- Keep old/new overlap window short during rollout.
- Verify both health and E2E proof checks immediately after rotation.
- Store token only in secret manager or service env file with least-privilege access.

### 5.2 Richer usage/observability metadata
- Add request correlation in logs at every seam:
  - RN request ID
  - FastAPI request ID
  - Cactus request ID
  - upstream request evidence where available
- Add counters/alerts for:
  - upstream timeout rate
  - fail-closed error rate
  - proof-field validation failures
- Keep provider internals sanitized for client responses; emit full diagnostics only server-side.

### 5.3 Retry policy guidance
- Keep fail-closed behavior as default.
- Use bounded retries only for transient classes (timeouts/5xx/network reset).
- Avoid retries for deterministic auth/schema failures.
- Keep total upstream budget bounded (current Cactus boundary budget: 20s).

## 6) Promotion gate (dev endpoint -> production config)

Recommend promotion only when all are true:
1. HF `/healthz` is stable over repeated checks.
2. Chat contract checks pass with expected response shape.
3. RN-facing E2E proof succeeds with required proof fields preserved.
4. At least one fail-closed scenario verified and observable.
5. Rotation + observability checklist items are scheduled/owned.

## 7) Upstream provider references used

- Hugging Face Inference Endpoints docs:
  - https://huggingface.co/docs/inference-endpoints/index
- Hugging Face custom container guidance:
  - https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_container
- Hugging Face Inference Providers (OpenAI-compatible usage context):
  - https://huggingface.co/docs/inference-providers/index
- OpenAI Chat Completions API shape (compatibility target):
  - https://platform.openai.com/docs/api-reference/chat

## 8) Scope note

This runbook is ops-focused documentation for #82 and #86. It does not change runtime code paths by itself.
