#!/bin/bash
set -e

echo "===== Installing compatible MediaPipe version ====="
python3.11 -m pip install --quiet --upgrade 'mediapipe>=0.8.9,<0.10' 2>&1 || echo "Attempting alternative installation..."

# Try older version if that fails
python3.11 -m pip install --quiet --force-reinstall 'mediapipe==0.8.11' 2>&1 || echo "Using current version..."

echo -e "\n===== Verifying MediaPipe ====="
python3.11 -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); print('Has solutions:', hasattr(mp, 'solutions'))"

echo -e "\n===== Running pose extraction ====="
cd /app
python3.11 scripts/extract_poses_batch.py \
  --metadata-path /tmp/WLASL/start_kit/WLASL_v0.3.json \
  --video-dir /tmp/WLASL/start_kit/videos \
  --output-dir data/processed/poses \
  --log-level INFO

echo -e "\n===== Extraction complete ====="
ls -lh data/processed/poses/ 2>/dev/null | tail -20 || echo "Output directory check failed"
echo "Total pose files processed: $(ls data/processed/poses/ 2>/dev/null | wc -l || echo '0')"
