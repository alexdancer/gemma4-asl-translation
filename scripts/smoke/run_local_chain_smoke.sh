#!/usr/bin/env bash
set -euo pipefail

# Deterministic local chain smoke runner
# Verifies: cloud API (:8000) -> HF custom endpoint (:9000, backend=real|stub)
# Output artifacts:
#   /tmp/asl-smoke/hf_health.json
#   /tmp/asl-smoke/chain_resp.json
#   /tmp/asl-smoke/hf9000.log
#   /tmp/asl-smoke/api8000.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-.venv-py312/bin/activate}"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "FAIL: venv activate script not found at $VENV_PATH"
  exit 2
fi
# shellcheck source=/dev/null
source "$VENV_PATH"

mkdir -p /tmp/asl-smoke

SMOKE_VIDEO="${SMOKE_VIDEO:-/tmp/asl-smoke/smoke.mp4}"
HF_HEALTH_JSON="/tmp/asl-smoke/hf_health.json"
CHAIN_RESP_JSON="/tmp/asl-smoke/chain_resp.json"
HF_LOG="/tmp/asl-smoke/hf9000.log"
API_LOG="/tmp/asl-smoke/api8000.log"

ASL_V1_API_KEYS="${ASL_V1_API_KEYS:-dev-local-key-1}"
ASL_DEV_CLIENT_KEY="${ASL_DEV_CLIENT_KEY:-${ASL_V1_API_KEYS%%,*}}"
ASL_CLOUD_API_KEY="${ASL_CLOUD_API_KEY:-$ASL_DEV_CLIENT_KEY}"
ASL_CLOUD_INFER_URL="${ASL_CLOUD_INFER_URL:-http://127.0.0.1:9000/v1/translate-sign}"
ASL_CLOUD_UPSTREAM_APP_KEY="${ASL_CLOUD_UPSTREAM_APP_KEY:-}"
ASL_HF_ENDPOINT_BACKEND="${ASL_HF_ENDPOINT_BACKEND:-real}"
ASL_HF_ENDPOINT_MODEL_ID="${ASL_HF_ENDPOINT_MODEL_ID:-AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit}"
ASL_HF_ENDPOINT_MODEL_VERSION="${ASL_HF_ENDPOINT_MODEL_VERSION:-gemma4-e2b-q64-top50}"

if [[ "$ASL_HF_ENDPOINT_BACKEND" == "real" && -z "${ASL_HF_TOKEN:-}" ]]; then
  echo "FAIL: ASL_HF_TOKEN is required when ASL_HF_ENDPOINT_BACKEND=real"
  exit 2
fi

if [[ "$ASL_CLOUD_INFER_URL" == *"endpoints.huggingface.cloud"* && "$ASL_CLOUD_INFER_URL" == *"/v1/translate-sign"* && -z "$ASL_CLOUD_UPSTREAM_APP_KEY" ]]; then
  echo "FAIL: ASL_CLOUD_UPSTREAM_APP_KEY is required for hosted HF /v1/translate-sign upstream auth"
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "FAIL: curl is required"
  exit 2
fi

if [[ ! -f "$SMOKE_VIDEO" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FAIL: smoke video missing and ffmpeg is not installed"
    exit 2
  fi
  ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=1 -pix_fmt yuv420p "$SMOKE_VIDEO" >/tmp/asl-smoke/ffmpeg.log 2>&1
fi

rm -f "$HF_HEALTH_JSON" "$CHAIN_RESP_JSON"

HF_PID=""
API_PID=""

cleanup() {
  set +e
  if [[ -n "$API_PID" ]]; then kill "$API_PID" 2>/dev/null || true; fi
  if [[ -n "$HF_PID" ]]; then kill "$HF_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

for port in 8000 9000; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: port $port is already in use"
    exit 2
  fi
done

export ASL_V1_API_KEYS
export ASL_CLOUD_API_KEY
export ASL_CLOUD_INFER_URL
export ASL_CLOUD_UPSTREAM_APP_KEY
export ASL_HF_ENDPOINT_BACKEND
export ASL_HF_ENDPOINT_MODEL_ID
export ASL_HF_ENDPOINT_MODEL_VERSION

python scripts/runtime/serve_hf_custom_endpoint_service.py --host 127.0.0.1 --port 9000 >"$HF_LOG" 2>&1 &
HF_PID=$!

python scripts/runtime/serve_cloud_translate_api.py --host 127.0.0.1 --port 8000 >"$API_LOG" 2>&1 &
API_PID=$!

ready=0
for _ in {1..30}; do
  if curl -sf http://127.0.0.1:9000/healthz >"$HF_HEALTH_JSON"; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "FAIL: HF health check did not become ready"
  tail -n 40 "$HF_LOG" || true
  exit 1
fi

api_ready=0
for _ in {1..30}; do
  # cloud_translate_app has no dedicated /healthz route; any HTTP response on
  # /v1/translate-sign proves the server is listening and routing requests.
  if curl -sS -o /dev/null http://127.0.0.1:8000/v1/translate-sign; then
    api_ready=1
    break
  fi
  sleep 1
done

if [[ "$api_ready" -ne 1 ]]; then
  echo "FAIL: cloud API did not become reachable on :8000"
  tail -n 60 "$API_LOG" || true
  exit 1
fi

if ! curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/translate-sign \
  -H "Authorization: Bearer $ASL_DEV_CLIENT_KEY" \
  -F "video=@${SMOKE_VIDEO};type=video/mp4" \
  >"$CHAIN_RESP_JSON"; then
  echo "FAIL: chain request failed (HTTP or transport)"
  if [[ -f "$CHAIN_RESP_JSON" ]]; then
    echo "--- chain response body ---"
    cat "$CHAIN_RESP_JSON" || true
  fi
  tail -n 60 "$API_LOG" || true
  tail -n 60 "$HF_LOG" || true
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path

health_path = Path('/tmp/asl-smoke/hf_health.json')
resp_path = Path('/tmp/asl-smoke/chain_resp.json')

def load_json(path: Path, label: str):
    raw = path.read_text()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        snippet = raw[:200].replace('\n', ' ')
        print(f"FAIL: invalid JSON in {label} ({path}): {snippet}")
        raise SystemExit(1)

health = load_json(health_path, 'hf health response')
resp = load_json(resp_path, 'chain response')

print('=== HF /healthz ===')
print(json.dumps(health, indent=2))
print('\n=== Chain response ===')
print(json.dumps(resp, indent=2))

if health.get('ready') is not True:
    print('\nFAIL: HF endpoint ready=false')
    raise SystemExit(1)

if resp.get('error_code'):
    print(f"\nFAIL: chain returned error_code={resp.get('error_code')}")
    raise SystemExit(1)

required = ['prediction', 'translation', 'runtime_mode', 'cloud_handoff', 'model_id', 'model_version']
missing = [k for k in required if k not in resp]
if missing:
    print(f"\nFAIL: missing required fields: {missing}")
    raise SystemExit(1)

print('\nPASS: local chain smoke succeeded')
PY

echo "Artifacts:"
echo "  $HF_HEALTH_JSON"
echo "  $CHAIN_RESP_JSON"
echo "  $HF_LOG"
echo "  $API_LOG"
