# Arch Operator Runbook: Cactus Hybrid + Hugging Face Handoff

This runbook documents production-style operator steps for hosting the Cactus hybrid upstream service on an Arch Linux GPU machine and wiring the backend `/v1/translate-sign` API to it.

Scope for issue #78:
- host setup + service launch (systemd)
- secret placement and env-variable mapping
- reverse proxy + TLS example
- proof and fail-closed verification commands
- references to Cactus docs used for setup decisions

---

## 0) Architecture + responsibility split

Runtime path:
1. RN app calls backend `/v1/translate-sign`
2. Backend calls Cactus hybrid service (`http://<host>:9000/`)
3. Cactus hybrid service performs cloud handoff via HF OpenAI-compatible endpoint
4. Cactus response includes proof fields; backend enforces proof contract

Required proof fields on successful upstream responses:
- `runtime_mode`
- `cloud_handoff`
- `model_id`
- `model_version`

---

## 1) Prerequisites on Arch host

- Repo cloned on host (example path used below):
  - `/home/alex-server/Documents/ASL-Hackathon/sign-language-asl`
- Python available (`python3`)
- `systemd` running
- outbound network access to Hugging Face router endpoint

Optional but recommended packages:

```bash
sudo pacman -Sy --needed python python-pip curl jq nginx
```

Install project Python dependencies used by runtime services:

```bash
cd /home/alex-server/Documents/ASL-Hackathon/sign-language-asl
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## 2) Secrets and env placement

Create service env directory/file:

```bash
sudo install -d /etc/asl
sudo cp /home/alex-server/Documents/ASL-Hackathon/sign-language-asl/deploy/env/cactus-hybrid.env.example /etc/asl/cactus-hybrid.env
sudo chmod 600 /etc/asl/cactus-hybrid.env
sudoedit /etc/asl/cactus-hybrid.env
```

Set values in `/etc/asl/cactus-hybrid.env`:

```dotenv
# Shared bearer secret expected by the hybrid service.
# Backend caller MUST send the same value in Authorization header.
ASL_CACTUS_SERVICE_API_KEY=<strong-random-secret>

# HF OpenAI-compatible router base URL (required for /v1/chat/completions path)
ASL_HF_OPENAI_BASE_URL=https://router.huggingface.co/v1

# HF token with inference/read access to selected model
ASL_HF_TOKEN=hf_xxx

# Proof field value returned by service
ASL_CACTUS_MODEL_VERSION=2026-05-11
```

### Critical key mapping (backend -> upstream)

When running backend (`serve_cloud_translate_api.py`), set:

```bash
export ASL_CLOUD_INFER_URL='http://<arch-host>:9000/'
export ASL_CLOUD_API_KEY='<same value as ASL_CACTUS_SERVICE_API_KEY>'
```

Exact equality rule:
- `ASL_CLOUD_API_KEY` **must equal** `ASL_CACTUS_SERVICE_API_KEY`

---

## 3) systemd service install + launch (FastAPI via uvicorn)

Copy unit file:

```bash
sudo cp /home/alex-server/Documents/ASL-Hackathon/sign-language-asl/deploy/systemd/asl-cactus-hybrid.service /etc/systemd/system/
```

Update `/etc/systemd/system/asl-cactus-hybrid.service` for your host:
- `User=` / `Group=` (valid host account)
- `WorkingDirectory=` (actual repo path)
- `ExecStart=` (python path + script path using uvicorn-backed runtime)

If service runs from a home-directory path, set:
- `ProtectHome=false`

Then reload/start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now asl-cactus-hybrid.service
sudo systemctl status asl-cactus-hybrid.service --no-pager
journalctl -u asl-cactus-hybrid.service -n 100 --no-pager
```

Expected startup log signal now comes from uvicorn/FastAPI (for example: `Uvicorn running on http://0.0.0.0:9000`).

Troubleshooting quick map:
- `status=217/USER` -> invalid `User=`/`Group=`
- `status=200/CHDIR` -> bad `WorkingDirectory=`
- starts/fails with home path + `ProtectHome=true` -> set `ProtectHome=false`

---

## 4) Reverse proxy + TLS (Nginx example)

If backend cannot call port `9000` directly, front it with HTTPS.

### 4.1 Nginx server block

Create `/etc/nginx/conf.d/asl-cactus-hybrid.conf`:

```nginx
server {
  listen 443 ssl http2;
  server_name cactus.example.com;

  ssl_certificate /etc/letsencrypt/live/cactus.example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/cactus.example.com/privkey.pem;

  location / {
    proxy_pass http://127.0.0.1:9000/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

Reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 4.2 Backend mapping with TLS endpoint

```bash
export ASL_CLOUD_INFER_URL='https://cactus.example.com/'
export ASL_CLOUD_API_KEY='<same value as ASL_CACTUS_SERVICE_API_KEY>'
```

---

## 5) Proof verification (must show 4 fields)

Set local shell vars:

```bash
export ASL_CLOUD_API_KEY="$(sudo awk -F= '/^ASL_CACTUS_SERVICE_API_KEY=/{print $2}' /etc/asl/cactus-hybrid.env)"
MODEL_ID='meta-llama/Llama-3.1-8B-Instruct'
```

Run proof request:

```bash
curl -sS http://127.0.0.1:9000/ \
  -H "Authorization: Bearer $ASL_CLOUD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"proof-1\",\"model\":\"$MODEL_ID\",\"input\":{\"filename\":\"probe.mp4\",\"pose_summary\":\"proof run\"}}" | jq .
```

Strict proof-field check (machine-checkable pass/fail):

```bash
curl -sS http://127.0.0.1:9000/ \
  -H "Authorization: Bearer $ASL_CLOUD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"proof-1\",\"model\":\"$MODEL_ID\",\"input\":{\"filename\":\"probe.mp4\",\"pose_summary\":\"proof run\"}}" \
  | jq -e '.runtime_mode != null and .cloud_handoff != null and .model_id != null and .model_version != null'
```

Verify response includes:
- `runtime_mode`
- `cloud_handoff`
- `model_id`
- `model_version`

---

## 6) Fail-closed verification

Intentionally break HF token:

```bash
sudo cp /etc/asl/cactus-hybrid.env /tmp/cactus-hybrid.env.bak
sudo sed -i 's/^ASL_HF_TOKEN=.*/ASL_HF_TOKEN=broken/' /etc/asl/cactus-hybrid.env
sudo systemctl restart asl-cactus-hybrid.service
```

Then call service:

```bash
curl -i -sS http://127.0.0.1:9000/ \
  -H "Authorization: Bearer $ASL_CLOUD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"failclosed-1","model":"meta-llama/Llama-3.1-8B-Instruct","input":{"filename":"probe.mp4","pose_summary":"fail closed"}}'
```

Expected:
- non-200 (typically `503 Service Unavailable`)
- `error_code` = `CLOUD_HANDOFF_FAILED`

Restore env and restart:

```bash
sudo cp /tmp/cactus-hybrid.env.bak /etc/asl/cactus-hybrid.env
sudo systemctl restart asl-cactus-hybrid.service
```

---

## 7) Cactus docs references used

- Home: https://docs.cactuscompute.com/v1.14/
- Choose SDK: https://docs.cactuscompute.com/v1.14/docs/choose-sdk/
- React Native SDK: https://docs.cactuscompute.com/v1.14/react-native/
- Fine-tuning/deployment: https://docs.cactuscompute.com/v1.14/docs/finetuning/
- Cactus engine API: https://docs.cactuscompute.com/v1.14/docs/cactus_engine/

---

## 8) Operator evidence checklist

Capture and save for issue closure:
1. `systemctl status` showing active service
2. proof `curl | jq` output with 4 proof fields present
3. fail-closed `curl -i` output showing non-200 + `CLOUD_HANDOFF_FAILED`
4. backend env mapping proof (`ASL_CLOUD_INFER_URL`, `ASL_CLOUD_API_KEY` mapping notes)
5. proxy/TLS config snippet used in deployment
