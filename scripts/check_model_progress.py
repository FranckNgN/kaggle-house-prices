#!/usr/bin/env python
"""
Check the progress of currently running model training processes.
"""
import sys
import os
import time
from pathlib import Path
import subprocess
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "notebooks" / "Models"
SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "submissions"

def get_model_scripts():
    """Get all model scripts."""
    scripts = []
    for script_file in sorted(MODELS_DIR.glob("*.py")):
        if script_file.name[0].isdigit():
            scripts.append(script_file.stem)
    return scripts

def check_submission_files():
    """Check which models have created submission files."""
    completed = []
    if SUBMISSIONS_DIR.exists():
        for subdir in SUBMISSIONS_DIR.iterdir():
            if subdir.is_dir():
                # Check if directory has a submission.csv file
                csv_files = list(subdir.glob("*.csv"))
                if csv_files:
                    completed.append(subdir.name)
    return completed

def get_running_python_processes():
    """Get information about running Python processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline'] or []
                # Check if it's running a model script
                for arg in cmdline:
                    if 'Models' in arg and arg.endswith('.py'):
                        model_name = Path(arg).stem
                        runtime = time.time() - proc.info['create_time']
                        processes.append({
                            'model': model_name,
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'runtime_seconds': runtime
                        })
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def main():
    print("=" * 70)
    print("MODEL TRAINING PROGRESS CHECK")
    print("=" * 70)
    
    all_models = get_model_scripts()
    print(f"\nTotal models to run: {len(all_models)}")
    print(f"Models: {', '.join(all_models)}")
    
    # Check running processes
    print("\n" + "-" * 70)
    print("CURRENTLY RUNNING MODELS")
    print("-" * 70)
    running = get_running_python_processes()
    
    if running:
        print(f"\nFound {len(running)} active model processes:\n")
        for proc in running:
            runtime_min = proc['runtime_seconds'] / 60
            print(f"  • {proc['model']}")
            print(f"    PID: {proc['pid']}")
            print(f"    Runtime: {runtime_min:.1f} minutes")
            print(f"    CPU: {proc['cpu_percent']:.1f}%")
            print(f"    Memory: {proc['memory_mb']:.1f} MB")
            print()
    else:
        print("\n  No model processes currently running.")
    
    # Check completed models (by submission files)
    print("-" * 70)
    print("COMPLETED MODELS (by submission files)")
    print("-" * 70)
    completed = check_submission_files()
    
    if completed:
        print(f"\nFound {len(completed)} completed models:\n")
        for model in sorted(completed):
            print(f"  ✓ {model}")
    else:
        print("\n  No submission files found yet.")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Running: {len(running)}")
    print(f"Completed (by files): {len(completed)}")
    print(f"Remaining: {len(all_models) - len(completed)}")
    
    # Check for results file
    results_file = PROJECT_ROOT / "runs" / "model_training_results.txt"
    if results_file.exists():
        print(f"\n[OK] Results file exists: {results_file}")
        print("  (This means the parallel script has finished)")
    else:
        print(f"\n[INFO] Results file not found yet (script still running)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)

