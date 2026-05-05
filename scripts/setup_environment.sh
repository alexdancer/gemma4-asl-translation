#!/bin/bash

# ASL Fine-Tuning Environment Setup
# Creates venv, installs dependencies, verifies CUDA/GPU

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON_VERSION="3.10"

echo "======================================"
echo "ASL Fine-Tuning Environment Setup"
echo "======================================"
echo ""

# Step 1: Check Python version
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.10+ and try again."
    exit 1
fi

PYTHON_INSTALLED=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python ${PYTHON_INSTALLED}"
echo ""

# Step 2: Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "✓ Virtual environment already exists at ${VENV_DIR}"
else
    python3 -m venv "$VENV_DIR"
    echo "✓ Created virtual environment at ${VENV_DIR}"
fi
echo ""

# Step 3: Activate venv and upgrade pip
echo "[3/6] Activating environment and upgrading pip..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"
echo ""

# Step 4: Install dependencies
echo "[4/6] Installing dependencies from requirements.txt..."
if [ ! -f "${PROJECT_ROOT}/requirements.txt" ]; then
    echo "ERROR: requirements.txt not found at ${PROJECT_ROOT}"
    exit 1
fi

pip install -r "${PROJECT_ROOT}/requirements.txt"
echo "✓ Dependencies installed"
echo ""

# Step 5: Verify CUDA/GPU
echo "[5/6] Checking CUDA and GPU availability..."
python3 << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available. Tests will run on CPU (slower).")
    print("         For GPU acceleration, install CUDA 12.1+ and reinstall torch.")

print()
EOF

echo "✓ CUDA/GPU check complete"
echo ""

# Step 6: Test imports
echo "[6/6] Testing critical imports..."
python3 << 'EOF'
import sys
errors = []

try:
    import torch
    print("torch")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    import transformers
    print("transformers")
except ImportError as e:
    errors.append(f"transformers: {e}")

try:
    import unsloth
    print("✓ unsloth")
except ImportError as e:
    errors.append(f"unsloth: {e}")

try:
    import pandas
    print("✓ pandas")
except ImportError as e:
    errors.append(f"pandas: {e}")

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError as e:
    errors.append(f"matplotlib: {e}")

try:
    from peft import get_peft_model
    print("peft")
except ImportError as e:
    errors.append(f"peft: {e}")

if errors:
    print()
    print("Some imports failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Import verification failed"
    exit 1
fi

echo "✓ All critical imports verified"
echo ""

# Final summary
echo "======================================"
echo "✓ Environment setup complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Activate: source ${VENV_DIR}/bin/activate"
echo "  2. Run tests: python scripts/legacy/test_finetuning.py --mock-model"
echo "  3. Or run notebook: jupyter notebook notebooks/05_test_pipeline.ipynb"
echo ""
