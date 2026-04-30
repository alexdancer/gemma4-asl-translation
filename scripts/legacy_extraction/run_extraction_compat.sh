#!/bin/bash
set -e

echo "===== Running pose extraction with MediaPipe compatibility layer ====="
cd /app

# Run Python with the compatibility layer imported first
python3.11 << 'EOF'
import sys
sys.path.insert(0, '/app')

# Import compatibility layer
import mediapipe_compat

# Now run the extraction
import scripts.extract_poses_batch as extract_module
extract_module.main()
EOF

echo -e "\n===== Checking extraction progress ====="
POSE_COUNT=$(ls /app/data/processed/poses/ 2>/dev/null | wc -l || echo '0')
echo "Total pose files processed: $POSE_COUNT"

if [ "$POSE_COUNT" -gt "0" ]; then
    echo "✅ Extraction is running successfully!"
    ls -lh /app/data/processed/poses/ | head -10
else
    echo "⚠️ No pose files yet - extraction may still be in progress"
fi
