# HITL #85 Exact Setup + Run Instructions (Device)

Use this for real-device sign-off run.

## 0) OpenAI-compatible endpoint value (exact)

Set `ASL_HF_OPENAI_BASE_URL` to your Hugging Face endpoint base **with `/v1` suffix**.

Example:

```bash
export ASL_HF_OPENAI_BASE_URL='https://abc123.us-east-1.aws.endpoints.huggingface.cloud/v1'
```

Cactus appends `/chat/completions`.
Final upstream URL becomes:

`https://abc123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions`

## 1) Required env vars (copy/paste block)

From repo root (`/Users/alex/Documents/ASL-project/sign-language-asl`):

```bash
export ASL_HF_OPENAI_BASE_URL='https://<your-hf-endpoint-host>/v1'
export ASL_HF_TOKEN='hf_xxx_your_token'

# FastAPI -> Cactus auth (must match)
export ASL_CACTUS_SERVICE_API_KEY='asl-local-long-random-secret-change-me'
export ASL_CLOUD_API_KEY="$ASL_CACTUS_SERVICE_API_KEY"

# Cactus service URL used by FastAPI
export ASL_CLOUD_INFER_URL='http://127.0.0.1:9000/'

# RN caller key allow-list on FastAPI boundary
export ASL_V1_API_KEYS='dev-local-key-1'

# Optional proof version label
export ASL_CACTUS_MODEL_VERSION='2026-05-13'

# Route mode locked for this path
export ASL_HF_ROUTE_MODE='chat'
```

## 2) Quick sanity checks

```bash
python - <<'PY'
import os
keys=[
  'ASL_HF_OPENAI_BASE_URL','ASL_HF_TOKEN','ASL_CACTUS_SERVICE_API_KEY',
  'ASL_CLOUD_API_KEY','ASL_CLOUD_INFER_URL','ASL_V1_API_KEYS','ASL_HF_ROUTE_MODE'
]
for k in keys:
    v=os.getenv(k,'')
    print(f"{k}:", 'SET' if v else 'MISSING', f"len={len(v)}")
print('KEY_MATCH:', os.getenv('ASL_CLOUD_API_KEY')==os.getenv('ASL_CACTUS_SERVICE_API_KEY'))
PY
```

Expected:
- all `SET`
- `KEY_MATCH: True`

## 3) Start stack (device mode)

```bash
bash scripts/dev/run_mobile_stack.sh device
```

Keep terminal open.

## 4) Preflight verifier commands (separate terminal)

```bash
mkdir -p evidence/hitl-02

bash -n scripts/runtime/verify_rn_fastapi_cactus_hf_e2e.sh \
  | tee evidence/hitl-02/00-bashn-verify-script.txt

bash -n scripts/runtime/prove_cactus_hybrid.sh \
  | tee evidence/hitl-02/01-bashn-proof-script.txt

. .venv/bin/activate
python -m pytest tests/runtime/test_e2e_rn_fastapi_cactus_hf_chain.py -q \
  | tee evidence/hitl-02/02-runtime-e2e-test.txt
python -m pytest tests/runtime/test_cactus_hybrid_service.py -q \
  | tee evidence/hitl-02/03-runtime-cactus-test.txt
```

## 5) Real-device happy path

1. Open app on physical iPhone.
2. Do normal upload flow (no endpoint/key UI input).
3. Upload known-good short ASL video.
4. Wait for translation result.
5. Save evidence:
   - app success screenshot -> `evidence/hitl-02/A1-app-happy-path.png`
   - backend logs around request -> `evidence/hitl-02/A2-backend-happy-path.log`
   - response JSON -> `evidence/hitl-02/A3-response-happy-path.json`

## 6) Real-device fail-closed scenario

Use bad HF token temporarily.

```bash
export ASL_HF_TOKEN='hf_BAD_TOKEN_FOR_FAIL_CLOSED_CHECK'
```

1. Repeat same upload flow on device.
2. Confirm explicit user-visible error (no silent fallback).
3. Save evidence:
   - app failure screenshot -> `evidence/hitl-02/B1-app-fail-closed.png`
   - backend fail logs -> `evidence/hitl-02/B2-backend-fail-closed.log`
   - response JSON -> `evidence/hitl-02/B3-response-fail-closed.json`

Restore valid token after test:

```bash
export ASL_HF_TOKEN='hf_xxx_your_real_token'
```

## 7) Evidence assertions

```bash
jq -e '.prediction != null and .prediction != ""' evidence/hitl-02/A3-response-happy-path.json
jq -e '.runtime_mode != null and .cloud_handoff == true and .model_id != null and .model_version != null' evidence/hitl-02/A3-response-happy-path.json
jq -e '.ok == false and .error_code != null and .error_message != null and .error_message != ""' evidence/hitl-02/B3-response-fail-closed.json
```

Pass = all commands exit `0`.

## 8) Promotion go/no-go gate

GO only if all true:
- happy-path translation shown on device
- no manual endpoint/key UI entry used
- proof fields present in happy-path JSON
- fail-closed behavior shown with explicit user-visible error
- evidence files saved under `evidence/hitl-02/`

If any gate fails -> NO-GO + note blocker.
