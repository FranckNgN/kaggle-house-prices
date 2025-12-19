#!/usr/bin/env python
"""
Wrapper script to ensure Python commands run in virtual environment.
This script automatically activates venv and runs the specified command.

Usage:
    python scripts/run_with_venv.py <script> [args...]
    
Or import and use:
    from scripts.run_with_venv import get_venv_python
    python_exe = get_venv_python()
"""

import sys
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_venv_python():
    """
    Get the Python interpreter from venv, or system Python if venv not found.
    Returns Path object to Python executable.
    """
    # Check common venv locations
    venv_locations = [
        PROJECT_ROOT / ".venv",
        PROJECT_ROOT / "venv",
        PROJECT_ROOT / "env",
    ]
    
    for venv_path in venv_locations:
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            return python_exe
    
    # Fallback to system Python
    return Path(sys.executable)


def main():
    """Run command with venv Python."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_with_venv.py <script> [args...]")
        print("\nExample:")
        print("  python scripts/run_with_venv.py scripts/submit_to_kaggle.py data/submissions/model.csv")
        sys.exit(1)
    
    # Get venv Python (or system Python as fallback)
    venv_python = get_venv_python()
    
    # Run the command with venv Python
    script = sys.argv[1]
    args = sys.argv[2:]
    
    cmd = [str(venv_python), script] + args
    
    # Set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import os
    main()

