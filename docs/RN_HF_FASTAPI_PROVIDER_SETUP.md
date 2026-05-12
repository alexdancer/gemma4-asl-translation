# React Native + Local API + Hugging Face Adapter Setup (Detailed)

This guide shows the full end-to-end setup for running the React Native app with:

1. RN client (`apps/mobile-rn`)
2. Local ASL cloud API (`scripts/runtime/serve_cloud_translate_api.py`)
3. A FastAPI adapter in front of Hugging Face inference

---

## Why this adapter exists

The local ASL API expects an upstream provider contract shaped like:

- Request (to upstream): bearer auth + JSON payload containing `request_id`, `model`, and `input` (including `video_base64`)
- Response (from upstream): must provide prediction + confidence fields that local API can normalize

Hugging Face model outputs vary by task/model, so a small FastAPI adapter is the safest way to normalize HF responses into the contract your local API expects.

---

## Architecture

```text
RN app (iOS/Android)
  -> POST/GET with X-API-Key
Local API :8000 (/v1/translate-sign)
  -> Bearer token + JSON
FastAPI adapter :9000
  -> Bearer HF token
Hugging Face endpoint
```

---

## Prerequisites

- macOS dev machine with repo cloned at:
  - `/Users/alex/Documents/ASL-project/sign-language-asl`
- Node/npm working
- Python 3 available
- iOS tooling for RN (`bundle`, CocoaPods, Xcode) if running iOS
- A Hugging Face token and endpoint URL

---

## Step 1) Prepare Hugging Face credentials

1. In Hugging Face: **Settings → Access Tokens**
2. Create a token with inference access.
3. Keep these values ready:
   - `HF_TOKEN` = your HF token
   - `HF_MODEL_URL` = your actual HF inference endpoint URL

> Do not use placeholders at runtime. Use real values.

---

## Step 2) Create a FastAPI adapter

Create a new file (outside or inside this repo; example below uses `/tmp`):

```bash
cat >/tmp/adapter_hf.py <<'PY'
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import time
import requests

app = FastAPI()

ADAPTER_BEARER = os.environ["ADAPTER_BEARER"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_MODEL_URL = os.environ["HF_MODEL_URL"]


class InputPayload(BaseModel):
    filename: str | None = None
    video_base64: str | None = None
    encoding: str | None = None


class InferRequest(BaseModel):
    request_id: str
    model: str | None = None
    input: InputPayload


@app.post("/")
def infer(req: InferRequest, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing auth")

    token = authorization.split(" ", 1)[1].strip()
    if token != ADAPTER_BEARER:
        raise HTTPException(status_code=401, detail="invalid auth")

    started = time.time()

    # IMPORTANT: adjust payload for your exact HF model contract.
    hf_payload = {
        "inputs": req.input.video_base64,
        "parameters": {"return_full_text": False},
    }

    resp = requests.post(
        HF_MODEL_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        json=hf_payload,
        timeout=30,
    )

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"hf error: {resp.status_code} {resp.text[:300]}")

    data = resp.json()

    # Normalize common HF output patterns.
    prediction = None
    confidence = None

    if isinstance(data, dict):
        prediction = data.get("prediction") or data.get("translation")
        confidence = data.get("confidence")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        prediction = data[0].get("generated_text") or data[0].get("label")
        confidence = data[0].get("score", 0.5)

    if not prediction:
        raise HTTPException(status_code=422, detail=f"cannot map HF response: {str(data)[:400]}")

    if confidence is None:
        confidence = 0.5

    latency_ms = int((time.time() - started) * 1000)

    return {
        "request_id": req.request_id,
        "prediction": str(prediction),
        "confidence": float(confidence),
        "latency_ms": latency_ms,
        "provider_raw": data,
    }
PY
```

---

## Step 3) Run the adapter service (:9000)

```bash
python3 -m venv /tmp/venv-adapter
source /tmp/venv-adapter/bin/activate
pip install fastapi uvicorn requests pydantic

export ADAPTER_BEARER='adapter-secret-123'
export HF_TOKEN='hf_xxx_real_token'
export HF_MODEL_URL='https://YOUR_REAL_HF_ENDPOINT'

uvicorn adapter_hf:app --app-dir /tmp --host 0.0.0.0 --port 9000
```

Keep this terminal running.

---

## Step 4) Run local ASL API (:8000)

Open a second terminal:

```bash
cd /Users/alex/Documents/ASL-project/sign-language-asl

# local API auth keys accepted from clients (RN sends this via X-API-Key)
export ASL_V1_API_KEYS='dev-local-key-1'

# upstream target = your adapter
export ASL_CLOUD_INFER_URL='http://127.0.0.1:9000/'
export ASL_CLOUD_API_KEY='adapter-secret-123'

python3 scripts/runtime/serve_cloud_translate_api.py --host 0.0.0.0 --port 8000
```

Keep this terminal running.

---

## Step 5) Run React Native app

Open a third terminal:

```bash
cd /Users/alex/Documents/ASL-project/sign-language-asl/apps/mobile-rn
npm install --include=dev
cd ios && bundle install && bundle exec pod install && cd ..
npm start
```

Open a fourth terminal (same directory) and launch app:

```bash
npm run ios
# or
npm run android
```

---

## Step 6) Fill app fields correctly

In RN UI:

- **Cloud endpoint**:
  - iOS simulator: `http://127.0.0.1:8000/v1/translate-sign`
  - physical phone: `http://<YOUR_MAC_LAN_IP>:8000/v1/translate-sign`
- **API key (for /v1/translate-sign)**:
  - `dev-local-key-1` (must match `ASL_V1_API_KEYS`)

> For physical phone testing, do not use `127.0.0.1`. Use your Mac LAN IP.

---

## Validation checklist

1. Adapter terminal shows incoming POST requests.
2. Local API terminal shows request handling without 401/502 failures.
3. RN app status progresses from upload → processing → result.
4. If auth fails, verify:
   - RN API key matches `ASL_V1_API_KEYS`
   - `ASL_CLOUD_API_KEY` matches `ADAPTER_BEARER`

---

## Common failure modes + fixes

### 401 from local API
- Cause: RN key missing/wrong.
- Fix: ensure RN API key input exactly equals one key in `ASL_V1_API_KEYS`.

### 502 from local API (upstream failure)
- Cause: adapter unreachable or HF call failing.
- Fix:
  - check adapter is running on `:9000`
  - verify `ASL_CLOUD_INFER_URL` and `ASL_CLOUD_API_KEY`
  - verify `HF_TOKEN` and `HF_MODEL_URL`

### 422 invalid provider response
- Cause: HF response shape not mapped by adapter.
- Fix: update mapping logic in adapter for your specific model output schema.

### RN cannot connect on physical phone
- Cause: using localhost/127.0.0.1.
- Fix: use `http://<MAC_LAN_IP>:8000/v1/translate-sign`.

---

## Security notes

- Never hardcode real keys in source files.
- Use env vars for `HF_TOKEN`, `ADAPTER_BEARER`, and backend keys.
- Rotate keys after demos/shared testing.

---

## Optional next hardening

- Add retry/backoff in adapter for transient HF errors.
- Add stricter schema validation per chosen HF model.
- Add health endpoint to adapter (`/healthz`) and local preflight checks.
