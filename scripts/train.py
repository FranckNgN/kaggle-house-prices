#!/usr/bin/env python
"""
Consolidated model training script.
Train all models or specific models sequentially.

Usage:
    python scripts/train.py                    # Train all models
    python scripts/train.py --models catboost  # Train specific model(s)
    python scripts/train.py --raw              # Train with raw preprocessing (skip feature engineering)
"""

import sys
import subprocess
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config_local.local_config as config

# Model scripts directory
MODELS_DIR = PROJECT_ROOT / "notebooks" / "Models"


def run_model_script(script_path: Path, enable_validation: bool = True) -> dict:
    """
    Run a single model script and return results.
    
    Args:
        script_path: Path to the model script
        enable_validation: Whether to run validation checks (default: True)
        
    Returns:
        Dictionary with execution results
    """
    model_name = script_path.stem
    print(f"[STARTING] {model_name}", flush=True)
    
    start_time = time.time()
    
    try:
        # Set PYTHONPATH to include project root (use absolute path)
        env = os.environ.copy()
        project_root_abs = str(PROJECT_ROOT.resolve())
        
        # Build PYTHONPATH with project root first
        current_pythonpath = env.get("PYTHONPATH", "")
        if current_pythonpath:
            # Avoid duplicates
            paths = [project_root_abs]
            for p in current_pythonpath.split(os.pathsep):
                if p and p != project_root_abs:
                    paths.append(p)
            env["PYTHONPATH"] = os.pathsep.join(paths)
        else:
            env["PYTHONPATH"] = project_root_abs
        
        # Set environment variable to enable validation if requested
        if enable_validation:
            env["ENABLE_MODEL_VALIDATION"] = "1"
        
        # Create a wrapper that sets up the path and then runs the script
        wrapper_code = f"""
import sys
import os
sys.path.insert(0, r'{project_root_abs}')
os.chdir(r'{PROJECT_ROOT.resolve()}')
with open(r'{script_path.resolve()}', 'r', encoding='utf-8') as f:
    code = compile(f.read(), r'{script_path.resolve()}', 'exec')
    exec(code, {{'__name__': '__main__', '__file__': r'{script_path.resolve()}'}})
"""
        
        result = subprocess.run(
            [sys.executable, "-u", "-c", wrapper_code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=7200  # 2 hour timeout per model
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[SUCCESS] {model_name} completed in {elapsed_time:.1f}s", flush=True)
            return {
                "model": model_name,
                "status": "success",
                "elapsed_time": elapsed_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"[FAILED] {model_name} failed after {elapsed_time:.1f}s", flush=True)
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[:20]:
                    if line.strip():
                        print(f"  {line}", flush=True)
            return {
                "model": model_name,
                "status": "failed",
                "elapsed_time": elapsed_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"[TIMEOUT] {model_name} exceeded 2 hour timeout", flush=True)
        return {
            "model": model_name,
            "status": "timeout",
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] {model_name} raised exception: {str(e)}", flush=True)
        return {
            "model": model_name,
            "status": "error",
            "elapsed_time": elapsed_time,
            "error": str(e)
        }


def get_model_scripts() -> List[Path]:
    """Get all model scripts sorted by number."""
    scripts = []
    for script_file in sorted(MODELS_DIR.glob("*.py")):
        if script_file.name[0].isdigit() and not script_file.name.endswith('.example'):
            scripts.append(script_file)
    return scripts


def get_model_script_by_name(model_name: str) -> Optional[Path]:
    """Get model script by name."""
    model_map = {
        'linear': '0linearRegression.py',
        'linear_updated': '1linearRegUpdated.py',
        'ridge': '2ridgeModel.py',
        'lasso': '3lassoModel.py',
        'elastic_net': '4elasticNetModel.py',
        'random_forest': '5randomForestModel.py',
        'svr': '6svrModel.py',
        'xgboost': '7XGBoostModel.py',
        'lightgbm': '8lightGbmModel.py',
        'catboost': '9catBoostModel.py',
        'blending': '10blendingModel.py',
        'stacking': '11stackingModel.py',
    }
    
    script_name = model_map.get(model_name.lower())
    if script_name:
        script_path = MODELS_DIR / script_name
        if script_path.exists():
            return script_path
    
    return None


def run_preprocessing(raw: bool = False) -> bool:
    """Run preprocessing pipeline."""
    if raw:
        print("\n" + "=" * 70)
        print("RUNNING RAW PREPROCESSING (without feature engineering)")
        print("=" * 70)
        script = PROJECT_ROOT / "scripts" / "run_raw_preprocessing.py"
    else:
        print("\n" + "=" * 70)
        print("RUNNING FULL PREPROCESSING PIPELINE")
        print("=" * 70)
        script = PROJECT_ROOT / "notebooks" / "preprocessing" / "run_preprocessing.py"
    
    if not script.exists():
        print(f"[ERROR] Preprocessing script not found: {script}")
        return False
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train machine learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py                           # Train all models
  python scripts/train.py --models catboost xgboost # Train specific models
  python scripts/train.py --raw                     # Train with raw preprocessing
        """
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Specific model names to train (e.g., catboost xgboost)'
    )
    parser.add_argument(
        '--raw', '-r',
        action='store_true',
        help='Use raw preprocessing (skip feature engineering)'
    )
    parser.add_argument(
        '--preprocess', '-p',
        action='store_true',
        help='Run preprocessing before training models'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    # Run preprocessing if requested
    if args.preprocess:
        if not run_preprocessing(raw=args.raw):
            print("\n[ERROR] Preprocessing failed. Cannot continue.")
            sys.exit(1)
    
    # Get model scripts to run
    if args.models:
        scripts_to_run = []
        for model_name in args.models:
            script = get_model_script_by_name(model_name)
            if script:
                scripts_to_run.append(script)
            else:
                print(f"[WARNING] Model '{model_name}' not found. Skipping.")
        if not scripts_to_run:
            print("[ERROR] No valid models to train.")
            sys.exit(1)
    else:
        scripts_to_run = get_model_scripts()
    
    print(f"\nFound {len(scripts_to_run)} model scripts to run:")
    for script in scripts_to_run:
        print(f"    - {script.name}")
    
    # Confirmation
    if not args.yes:
        print(f"\nThis will run {len(scripts_to_run)} models sequentially.")
        print("Estimated total time: ~{:.1f} hours".format(len(scripts_to_run) * 1.0))
        try:
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        except EOFError:
            print("Non-interactive mode detected. Proceeding...")
    
    start_time = time.time()
    results = []
    
    # Run models sequentially
    print("\n" + "=" * 70)
    print("RUNNING MODELS SEQUENTIALLY")
    print("=" * 70)
    print(f"Running {len(scripts_to_run)} models one after another...")
    print("-" * 70, flush=True)
    
    for idx, script in enumerate(scripts_to_run, 1):
        model_name = script.stem
        print(f"\n[{idx}/{len(scripts_to_run)}] Starting: {model_name}")
        print("-" * 70, flush=True)
        
        try:
            result = run_model_script(script, enable_validation=True)
            results.append(result)
            
            if result.get("status") == "success":
                elapsed = result.get("elapsed_time", 0)
                print(f"[SUCCESS] {model_name} completed in {elapsed/60:.1f} minutes", flush=True)
            else:
                print(f"[FAILED] {model_name} - Status: {result.get('status', 'unknown')}", flush=True)
                
        except Exception as e:
            print(f"[EXCEPTION] {model_name}: {str(e)}", flush=True)
            results.append({
                "model": model_name,
                "status": "exception",
                "error": str(e)
            })
        
        remaining = len(scripts_to_run) - idx
        if remaining > 0:
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / idx
            estimated_remaining = avg_time * remaining
            print(f"\n[PROGRESS] {idx}/{len(scripts_to_run)} models completed.")
            print(f"          Estimated time remaining: {estimated_remaining/60:.1f} minutes", flush=True)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    
    print(f"\nTotal models: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if successful:
        print(f"\n[SUCCESS] Successful models:")
        for r in successful:
            print(f"  - {r['model']} ({r.get('elapsed_time', 0)/60:.1f} minutes)")
    
    if failed:
        print(f"\n[FAILED] Failed models:")
        for r in failed:
            status = r.get("status", "unknown")
            print(f"  - {r['model']} ({status})")
    
    print("=" * 70)
    
    # Save results to file
    results_file = config.MODEL_TRAINING_RESULTS_TXT
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("MODEL TRAINING RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total execution time: {total_time:.1f}s\n")
        f.write(f"Successful: {len(successful)}/{len(results)}\n\n")
        
        for r in results:
            f.write(f"\n{r['model']}: {r.get('status', 'unknown')}\n")
            if r.get('elapsed_time'):
                f.write(f"  Time: {r['elapsed_time']:.1f}s\n")
            if r.get('stderr'):
                f.write(f"  Error:\n{r['stderr']}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user. Some models may still be running.")
        sys.exit(1)

