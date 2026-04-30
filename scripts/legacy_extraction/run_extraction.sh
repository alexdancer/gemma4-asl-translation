#!/bin/bash
set -e

echo "===== Checking MediaPipe installation ====="
python3.11 -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); print('Available attributes:', dir(mp))" || true

echo -e "\n===== Installing mediapipe-model-solutions ====="
python3.11 -m pip install -q mediapipe-model-solutions 2>&1 || echo "Install may have failed, proceeding..."

echo -e "\n===== Running pose extraction ====="
cd /app
python3.11 scripts/extract_poses_batch.py \
  --metadata-path /tmp/WLASL/start_kit/WLASL_v0.3.json \
  --video-dir /tmp/WLASL/start_kit/videos \
  --output-dir data/processed/poses \
  --log-level INFO

echo -e "\n===== Extraction complete ====="
ls -lh data/processed/poses/ | head -20
echo "Total pose files: $(ls data/processed/poses/ | wc -l)"
