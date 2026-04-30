#!/bin/bash
set -e

echo "===== Using MediaPipe compatibility layer ====="
cd /app

python3.11 << 'PYEOF'
import sys
import os
sys.path.insert(0, '/app')
os.chdir('/app')

# Patch MediaPipe before any other imports
import mediapipe_compat_simple

# Now import and run extraction
from scripts.extract_poses_batch import main
main()
PYEOF

echo -e "\n===== Extraction attempt complete ====="
POSE_COUNT=$(find /app/data/processed/poses -type f 2>/dev/null | wc -l || echo '0')
echo "Pose files created: $POSE_COUNT / 21083"
