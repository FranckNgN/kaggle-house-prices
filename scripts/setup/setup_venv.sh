#!/bin/bash
# Setup script to create and configure virtual environment
# Run this once: ./setup_venv.sh

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

echo "======================================================================"
echo "SETTING UP VIRTUAL ENVIRONMENT"
echo "======================================================================"

# Check if venv exists
if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Virtual environment already exists at $VENV_PATH"
else
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$VENV_PATH" || python -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[SUCCESS] Virtual environment created!"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[INFO] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Setup complete!"
    echo ""
    echo "To activate the virtual environment in the future, run:"
    echo "  source .venv/bin/activate"
else
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

