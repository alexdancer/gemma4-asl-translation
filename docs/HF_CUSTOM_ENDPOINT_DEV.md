# HF Custom Endpoint (Dev) — `/v1/chat/completions`

This document covers the tracer-bullet service for issue #81.

## Purpose
Provide a minimal OpenAI-compatible chat endpoint for Cactus cloud handoff bring-up.

## Runtime contract
- `GET /healthz`
- `POST /v1/chat/completions`

Successful chat response includes minimal fields only:
- `id`
- `object`
- `created`
- `model`
- `choices[0].message.content`

## Environment variables
- `ASL_HF_ENDPOINT_MODEL_ID` (default: `AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit`)
- `ASL_HF_ENDPOINT_MODEL_VERSION` (default: `dev`)
- `ASL_HF_ENDPOINT_BACKEND` (default: `stub`)

## Local run
```bash
python3 scripts/runtime/serve_hf_custom_endpoint_service.py --host 0.0.0.0 --port 8080
```

## Health check
```bash
curl -s http://127.0.0.1:8080/healthz | jq .
```

## Valid chat request
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit",
    "messages": [
      {"role":"system","content":"You are an ASL translator."},
      {"role":"user","content":"Translate this clip"}
    ]
  }' | jq .
```

## Deterministic invalid request response (non-200)
```bash
curl -s -o /tmp/hf-invalid.json -w '%{http_code}\n' \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[]}'
cat /tmp/hf-invalid.json | jq .
```

## Container build/run
```bash
docker build -f deploy/hf-endpoint/Dockerfile -t asl-hf-custom-endpoint:dev .
docker run --rm -p 8080:8080 \
  -e ASL_HF_ENDPOINT_MODEL_ID=AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit \
  asl-hf-custom-endpoint:dev
```
