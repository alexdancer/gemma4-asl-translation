#!/usr/bin/env bash
set -euo pipefail

# Repro check for issue #84:
# RN-facing local API path: /v1/translate-sign
# Expected chain: FastAPI -> Cactus hybrid -> HF custom endpoint

API_URL="${API_URL:-http://127.0.0.1:8000/v1/translate-sign}"
API_KEY="${API_KEY:-dev-local-key-1}"
REQUEST_ID="${REQUEST_ID:-rid-e2e-84}"
VIDEO_PATH="${VIDEO_PATH:-}"
JQ_BIN="${JQ_BIN:-jq}"

if ! command -v "$JQ_BIN" >/dev/null 2>&1; then
  echo "ERROR: jq is required" >&2
  exit 1
fi

if [[ -z "$VIDEO_PATH" ]]; then
  echo "ERROR: set VIDEO_PATH to a real local clip file" >&2
  echo "Example:" >&2
  echo "  VIDEO_PATH=/absolute/path/clip.mov API_URL=http://127.0.0.1:8000/v1/translate-sign bash scripts/runtime/verify_rn_fastapi_cactus_hf_e2e.sh" >&2
  exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "ERROR: VIDEO_PATH not found: $VIDEO_PATH" >&2
  exit 1
fi

echo "POST $API_URL"
RESPONSE_JSON="$(curl -sS -X POST "$API_URL" \
  -H "x-api-key: $API_KEY" \
  -H "x-request-id: $REQUEST_ID" \
  -F "video=@${VIDEO_PATH};type=video/quicktime")"

echo "$RESPONSE_JSON" | "$JQ_BIN" .

if ! echo "$RESPONSE_JSON" | "$JQ_BIN" -e '.request_id == "'"$REQUEST_ID"'"' >/dev/null; then
  echo "FAIL: request_id mismatch" >&2
  exit 1
fi

for field in prediction runtime_mode cloud_handoff model_id model_version; do
  if ! echo "$RESPONSE_JSON" | "$JQ_BIN" -e ".${field}" >/dev/null; then
    echo "FAIL: missing field '$field'" >&2
    exit 1
  fi
done

if ! echo "$RESPONSE_JSON" | "$JQ_BIN" -e '.cloud_handoff == true' >/dev/null; then
  echo "FAIL: expected cloud_handoff true" >&2
  exit 1
fi

echo "PASS: reproducible happy-path proof fields are present."
