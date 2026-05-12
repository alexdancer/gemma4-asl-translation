#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-device}"
if [[ "$MODE" != "device" && "$MODE" != "simulator" ]]; then
  echo "Usage: $0 [device|simulator]" >&2
  exit 64
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RN_DIR="$ROOT_DIR/apps/mobile-rn"
PYTHON_BIN="$ROOT_DIR/.venv-py312/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MAC_IP="${ASL_DEV_MAC_IP:-}"
if [[ -z "$MAC_IP" ]]; then
  MAC_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
fi

if [[ "$MODE" == "device" && -z "$MAC_IP" ]]; then
  echo "Could not detect Mac LAN IP. Set ASL_DEV_MAC_IP, for example:" >&2
  echo "  ASL_DEV_MAC_IP=192.168.1.25 npm run dev:ios:device" >&2
  exit 1
fi

export ASL_CACTUS_SERVICE_API_KEY="${ASL_CACTUS_SERVICE_API_KEY:-dev-cactus-secret}"
export ASL_CLOUD_API_KEY="${ASL_CLOUD_API_KEY:-$ASL_CACTUS_SERVICE_API_KEY}"
export ASL_V1_API_KEYS="${ASL_V1_API_KEYS:-dev-local-key-1}"
export ASL_CLOUD_INFER_URL="${ASL_CLOUD_INFER_URL:-http://127.0.0.1:9000/}"
export ASL_CLOUD_MODEL="${ASL_CLOUD_MODEL:-AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit}"
export ASL_HF_OPENAI_BASE_URL="${ASL_HF_OPENAI_BASE_URL:-https://router.huggingface.co/v1}"
export ASL_HF_ROUTE_MODE="${ASL_HF_ROUTE_MODE:-auto}"
export ASL_CACTUS_MODEL_VERSION="${ASL_CACTUS_MODEL_VERSION:-gemma4-e2b-q64-top50}"

if [[ "$MODE" == "device" ]]; then
  RN_API_URL="http://$MAC_IP:8000/v1/translate-sign"
else
  RN_API_URL="http://127.0.0.1:8000/v1/translate-sign"
fi

cat > "$RN_DIR/src/inferenceLocal.generated.ts" <<EOF
// Auto-written by scripts/dev/run_mobile_stack.sh for local development.
// Re-run npm run dev:mobile:device or npm run dev:mobile:sim to switch targets.
export const LOCAL_INFERENCE_TARGET = {
  mode: '$MODE',
  apiUrl: '$RN_API_URL',
  apiKey: '$ASL_V1_API_KEYS',
} as const;
EOF

LOG_DIR="${ASL_DEV_LOG_DIR:-/tmp/asl-mobile-stack}"
mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  echo
  echo "Stopping ASL local dev stack..."
  cat > "$RN_DIR/src/inferenceLocal.generated.ts" <<'EOF'
// Auto-written by scripts/dev/run_mobile_stack.sh for local development.
// Safe default keeps direct npm test / manual simulator runs working.
export const LOCAL_INFERENCE_TARGET = {
  mode: 'simulator',
  apiUrl: 'http://127.0.0.1:8000/v1/translate-sign',
  apiKey: 'dev-local-key-1',
} as const;
EOF
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_bg() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/$name.log"
  echo "Starting $name (log: $log_file)"
  (cd "$ROOT_DIR" && "$@") >"$log_file" 2>&1 &
  pids+=("$!")
}

start_metro() {
  local log_file="$LOG_DIR/metro.log"
  echo "Starting Metro (log: $log_file)"
  (cd "$RN_DIR" && npm start -- --host 0.0.0.0) >"$log_file" 2>&1 &
  pids+=("$!")
}

if [[ -z "${ASL_HF_TOKEN:-}" ]]; then
  echo "Warning: ASL_HF_TOKEN is not set. The stack can start, but real upstream HF inference will fail until it is set."
fi

if [[ "$ASL_CLOUD_MODEL" == "AlexD281/asl-gemma4-e2b-q64-top50-merged-16bit" ]] && [[ "$ASL_HF_OPENAI_BASE_URL" == "https://router.huggingface.co/v1" ]]; then
  echo "Warning: this model is not served by HF router chat/completions and router completions is unavailable for it." >&2
  echo "Set ASL_HF_OPENAI_BASE_URL to your dedicated HF endpoint base URL for option-2 deployment." >&2
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null
import mediapipe as mp
if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'holistic'):
    raise SystemExit(1)
PY
then
  echo "Backend Python at $PYTHON_BIN cannot import mediapipe.solutions.holistic." >&2
  echo "Create the supported local backend venv with:" >&2
  echo "  /opt/homebrew/bin/python3.12 -m venv .venv-py312" >&2
  echo "  .venv-py312/bin/python -m pip install fastapi uvicorn python-multipart numpy opencv-python 'mediapipe==0.10.14'" >&2
  exit 1
fi

start_bg cactus-hybrid "$PYTHON_BIN" scripts/runtime/serve_cactus_hybrid_service.py --host 0.0.0.0 --port 9000
start_bg cloud-translate "$PYTHON_BIN" scripts/runtime/serve_cloud_translate_api.py --host 0.0.0.0 --port 8000
start_metro

if [[ "$MODE" == "device" ]]; then
  echo
  echo "Physical iPhone backend endpoint: $RN_API_URL"
  echo "Model route: RN app -> FastAPI :8000 -> Cactus hybrid service :9000 -> HF OpenAI-compatible cloud handoff"
  echo "Cloud model: $ASL_CLOUD_MODEL ($ASL_CACTUS_MODEL_VERSION)"
  echo "The RN app uses the generated local dev config, so no in-app endpoint typing is needed."
  echo "Make sure your iPhone and Mac are on the same Wi-Fi and approve any iOS Local Network prompt."
  echo
  echo "Launching on a connected iPhone..."
  (cd "$RN_DIR" && npx react-native run-ios --device) || {
    echo "Device launch failed. Keep this stack running, then launch from Xcode if signing/device selection is needed."
  }
else
  echo
  echo "Simulator backend endpoint: $RN_API_URL"
  echo "Model route: RN app -> FastAPI :8000 -> Cactus hybrid service :9000 -> HF OpenAI-compatible cloud handoff"
  echo "Cloud model: $ASL_CLOUD_MODEL ($ASL_CACTUS_MODEL_VERSION)"
  echo
  echo "Launching simulator..."
  (cd "$RN_DIR" && npm run ios) || {
    echo "Simulator launch failed. Keep this stack running, then run npm run ios from apps/mobile-rn for details."
  }
fi

echo
echo "ASL local dev stack is running. Press Ctrl+C to stop."
echo "Logs: $LOG_DIR"
wait
