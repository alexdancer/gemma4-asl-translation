# Cactus Hybrid Service Deployment (Arch host, Issue #75)

This deploys the **upstream inference service** used by `/v1/translate-sign`.

## What this service guarantees
- Runs under systemd supervision on Arch.
- Uses Cactus hybrid service as routing authority.
- Uses Hugging Face OpenAI-compatible cloud handoff for phase 1.
- Returns required proof fields on **every success**:
  - `runtime_mode`
  - `cloud_handoff`
  - `model_id`
  - `model_version`
- Fails closed (non-200) when cloud handoff fails or proof fields are missing.

## 1) Install service files

```bash
sudo install -d /opt/sign-language-asl
sudo rsync -a --delete /path/to/sign-language-asl/ /opt/sign-language-asl/

sudo install -d /etc/asl
sudo cp /opt/sign-language-asl/deploy/env/cactus-hybrid.env.example /etc/asl/cactus-hybrid.env
sudoedit /etc/asl/cactus-hybrid.env

sudo cp /opt/sign-language-asl/deploy/systemd/asl-cactus-hybrid.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now asl-cactus-hybrid.service
```

## 2) Verify health and proof contract

```bash
curl -sS -X POST http://127.0.0.1:9000/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <ASL_CACTUS_SERVICE_API_KEY>' \
  -d '{"request_id":"proof-1","model":"cactus-asl-v2","input":{"filename":"probe.mp4","pose_summary":{"frame_count":10,"first_ts_ms":0,"last_ts_ms":333}}}' | jq
```

Expected fields in response:
`runtime_mode`, `cloud_handoff`, `model_id`, `model_version`.

## 3) Wire backend to this upstream

In backend environment (where `serve_cloud_translate_api.py` runs):

```bash
export ASL_CLOUD_INFER_URL='http://<arch-host>:9000/'
export ASL_CLOUD_API_KEY='<same value as ASL_CACTUS_SERVICE_API_KEY>'
```

## 4) Fail-closed check

Temporarily break HF token and ensure upstream returns non-200:

```bash
sudo sed -i 's/^ASL_HF_TOKEN=.*/ASL_HF_TOKEN=broken/' /etc/asl/cactus-hybrid.env
sudo systemctl restart asl-cactus-hybrid.service

curl -i -X POST http://127.0.0.1:9000/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <ASL_CACTUS_SERVICE_API_KEY>' \
  -d '{"request_id":"failclosed-1","model":"cactus-asl-v2","input":{"filename":"probe.mp4"}}'
```

Expected: `503 Service Unavailable` with `error_code=CLOUD_HANDOFF_FAILED`.
