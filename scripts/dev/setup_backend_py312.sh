#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY312="${PY312:-/opt/homebrew/bin/python3.12}"
if [[ ! -x "$PY312" ]]; then
  PY312="$(command -v python3.12 || true)"
fi
if [[ -z "$PY312" ]]; then
  echo "Python 3.12 is required for the legacy mediapipe.solutions API used by PoseExtractor." >&2
  echo "Install Python 3.12 first, then re-run this script." >&2
  exit 1
fi

"$PY312" -m venv .venv-py312
.venv-py312/bin/python -m pip install --upgrade pip
.venv-py312/bin/python -m pip install fastapi uvicorn python-multipart numpy opencv-python 'mediapipe==0.10.14'
.venv-py312/bin/python - <<'PY'
import mediapipe as mp
if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'holistic'):
    raise SystemExit('mediapipe.solutions.holistic unavailable')
print(f'Backend venv ready: mediapipe {mp.__version__} with solutions.holistic')
PY
