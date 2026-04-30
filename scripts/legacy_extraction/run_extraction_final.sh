#!/bin/bash
set -e

cd /app
export PYTHONPATH=/app:$PYTHONPATH

python3.11 << 'PYEOF'
import sys
sys.path.insert(0, '/app')

# Patch MediaPipe before any imports
import mediapipe_compat_simple as mp_compat

# Set up command line arguments
sys.argv = [
    'extract_poses_batch.py',
    '--metadata-path', '/tmp/WLASL/start_kit/WLASL_v0.3.json',
    '--video-dir', '/tmp/WLASL/start_kit/videos',
    '--output-dir', '/app/data/processed/poses',
    '--log-level', 'INFO'
]

# Import and run
from scripts import extract_poses_batch
extract_poses_batch.main()
PYEOF

echo "===== Pose extraction complete ====="
POSE_COUNT=$(find /app/data/processed/poses -type f 2>/dev/null | wc -l || echo '0')
echo "Total poses processed: $POSE_COUNT / 21083"
