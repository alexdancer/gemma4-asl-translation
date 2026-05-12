#!/usr/bin/env bash
set -euo pipefail

# Proves #75 behavior for Arch-hosted Cactus hybrid service:
# 1) Success path with required proof fields
# 2) Fail-closed path when HF token is intentionally broken

SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:9000/}"
ENV_FILE="${ENV_FILE:-/etc/asl/cactus-hybrid.env}"
JQ_BIN="${JQ_BIN:-jq}"

if ! command -v "$JQ_BIN" >/dev/null 2>&1; then
  echo "ERROR: jq is required (install jq and retry)." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

SERVICE_KEY="$(sudo awk -F= '/^ASL_CACTUS_SERVICE_API_KEY=/{print $2}' "$ENV_FILE" | tail -n1)"
HF_TOKEN="$(sudo awk -F= '/^ASL_HF_TOKEN=/{print $2}' "$ENV_FILE" | tail -n1)"

if [[ -z "$SERVICE_KEY" ]]; then
  echo "ERROR: ASL_CACTUS_SERVICE_API_KEY missing in $ENV_FILE" >&2
  exit 1
fi
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: ASL_HF_TOKEN missing in $ENV_FILE" >&2
  exit 1
fi

echo "Using service URL: $SERVICE_URL"
echo "Loaded service key length: ${#SERVICE_KEY}"

echo ""
echo "== Step 1: Success proof call =="
SUCCESS_JSON="$(curl -sS -X POST "$SERVICE_URL" \
  -H "Authorization: Bearer $SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id":"proof-success",
    "model":"cactus-asl-v2",
    "input":{
      "filename":"asl_sample.mp4",
      "pose_summary":"proof run"
    }
  }')"

echo "$SUCCESS_JSON" | "$JQ_BIN" .

for field in runtime_mode cloud_handoff model_id model_version; do
  if ! echo "$SUCCESS_JSON" | "$JQ_BIN" -e ".${field}" >/dev/null 2>&1; then
    echo "FAIL: missing proof field '$field' on success response" >&2
    exit 1
  fi
done
echo "PASS: success response contains required proof fields"

echo ""
echo "== Step 2: Fail-closed proof (intentionally break HF token) =="
BROKEN_TOKEN="${HF_TOKEN}_BROKEN"

sudo cp "$ENV_FILE" "${ENV_FILE}.bak"
restore_env() {
  sudo mv "${ENV_FILE}.bak" "$ENV_FILE"
  sudo systemctl restart asl-cactus-hybrid.service >/dev/null 2>&1 || true
}
trap restore_env EXIT

sudo sed -i '' "s|^ASL_HF_TOKEN=.*$|ASL_HF_TOKEN=${BROKEN_TOKEN}|" "$ENV_FILE" 2>/dev/null || \
sudo sed -i "s|^ASL_HF_TOKEN=.*$|ASL_HF_TOKEN=${BROKEN_TOKEN}|" "$ENV_FILE"

sudo systemctl restart asl-cactus-hybrid.service

FAIL_JSON="$(curl -sS -X POST "$SERVICE_URL" \
  -H "Authorization: Bearer $SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id":"proof-failclosed",
    "model":"cactus-asl-v2",
    "input":{
      "filename":"asl_sample.mp4",
      "pose_summary":"proof run"
    }
  }')"

echo "$FAIL_JSON" | "$JQ_BIN" .

ERR_CODE="$(echo "$FAIL_JSON" | "$JQ_BIN" -r '.error_code // empty')"
if [[ "$ERR_CODE" != "CLOUD_HANDOFF_FAILED" ]]; then
  echo "FAIL: expected error_code CLOUD_HANDOFF_FAILED, got '${ERR_CODE:-<empty>}'" >&2
  exit 1
fi

echo "PASS: fail-closed behavior confirmed"

echo ""
echo "== Step 3: Restore real HF token and restart =="
restore_env
trap - EXIT

echo "PASS: environment restored and service restarted"
echo "All checks passed."
