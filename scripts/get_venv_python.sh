#!/bin/bash
# Bash script to get venv Python path
# Returns the path to Python in .venv, or creates venv if it doesn't exist

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

get_venv_python() {
    if [ -d "$VENV_PATH" ]; then
        PYTHON_EXE="$VENV_PATH/bin/python"
        if [ -f "$PYTHON_EXE" ]; then
            echo "$PYTHON_EXE"
            return 0
        fi
    fi
    return 1
}

create_venv() {
    echo "[INFO] Creating virtual environment at $VENV_PATH..." >&2
    python3 -m venv "$VENV_PATH" || python -m venv "$VENV_PATH"
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Virtual environment created!" >&2
        get_venv_python
    else
        echo "[ERROR] Failed to create venv" >&2
        which python3 || which python
    fi
}

VENV_PYTHON=$(get_venv_python)
if [ -z "$VENV_PYTHON" ]; then
    VENV_PYTHON=$(create_venv)
fi

echo "$VENV_PYTHON"

