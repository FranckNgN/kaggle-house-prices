#!/usr/bin/env python
"""
Run all model training scripts with hybrid strategy:
- Linear models (0-4) run in parallel
- Demanding models (5-11) run sequentially, one by one, in order

This balances speed for fast models while preventing resource exhaustion
for memory-intensive models like XGBoost, CatBoost, and Stacking.
"""
import sys
import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        
        # Set environment variable to enable validation if requested
        if enable_validation:
            env["ENABLE_MODEL_VALIDATION"] = "1"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=3600  # 1 hour timeout per model
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
        print(f"[TIMEOUT] {model_name} exceeded 1 hour timeout", flush=True)
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


def get_model_scripts() -> list:
    """Get all model scripts sorted by number."""
    scripts = []
    for script_file in sorted(MODELS_DIR.glob("*.py")):
        if script_file.name[0].isdigit():  # Only numbered scripts
            scripts.append(script_file)
    return scripts


def categorize_models(scripts: list) -> tuple:
    """
    Categorize models into linear (fast) and demanding (slow) models.
    
    Returns:
        (linear_models, demanding_models) - both sorted by number
    """
    linear_models = []  # Models 0-4: linear, ridge, lasso, elastic net
    demanding_models = []  # Models 5-11: RF, SVR, XGBoost, LightGBM, CatBoost, blending, stacking
    
    for script in scripts:
        model_num = int(script.name[0]) if script.name[0].isdigit() else 99
        if model_num <= 4:
            linear_models.append(script)
        else:
            demanding_models.append(script)
    
    return linear_models, demanding_models


def main():
    """Main execution function."""
    print("=" * 70)
    print("HYBRID MODEL TRAINING")
    print("=" * 70)
    print("Strategy: Linear models in parallel, then demanding models sequentially")
    print("=" * 70)
    
    # Get all model scripts
    all_scripts = get_model_scripts()
    
    if not all_scripts:
        print("No model scripts found!")
        return
    
    # Categorize models
    linear_models, demanding_models = categorize_models(all_scripts)
    
    print(f"\nFound {len(all_scripts)} model scripts:")
    print(f"\n  Linear models (will run in parallel): {len(linear_models)}")
    for script in linear_models:
        print(f"    - {script.name}")
    
    print(f"\n  Demanding models (will run sequentially): {len(demanding_models)}")
    for script in demanding_models:
        print(f"    - {script.name}")
    
    # Check for --yes flag to skip confirmation
    skip_confirmation = '--yes' in sys.argv or '-y' in sys.argv
    
    if not skip_confirmation:
        # Ask for confirmation
        print(f"\nThis will:")
        print(f"  1. Run {len(linear_models)} linear models in parallel")
        print(f"  2. Then run {len(demanding_models)} demanding models one by one")
        print("Note: This may use significant CPU/memory resources.")
        try:
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        except EOFError:
            # Non-interactive mode, proceed automatically
            print("Non-interactive mode detected. Proceeding...")
    else:
        print(f"\nRunning models (--yes flag detected)...")
    
    start_time = time.time()
    results = []
    
    # ================================================================
    # PHASE 1: Run all linear models in parallel
    # ================================================================
    if linear_models:
        print("\n" + "=" * 70)
        print("PHASE 1: RUNNING LINEAR MODELS IN PARALLEL")
        print("=" * 70)
        
        max_workers = min(len(linear_models), os.cpu_count() or 4)
        print(f"Running {len(linear_models)} linear models with {max_workers} parallel workers...")
        print("-" * 70, flush=True)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all linear model tasks
            future_to_model = {
                executor.submit(run_model_script, script): script.stem
                for script in linear_models
            }
            
            print(f"[INFO] {len(future_to_model)} linear models submitted.", flush=True)
            
            # Process completed tasks as they finish
            completed_count = 0
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    remaining = len(linear_models) - completed_count
                    if remaining > 0:
                        print(f"[PROGRESS] {completed_count}/{len(linear_models)} linear models completed. {remaining} remaining.", flush=True)
                except Exception as e:
                    print(f"[EXCEPTION] {model_name}: {str(e)}", flush=True)
                    results.append({
                        "model": model_name,
                        "status": "exception",
                        "error": str(e)
                    })
        
        print(f"\n[PHASE 1 COMPLETE] All {len(linear_models)} linear models finished.", flush=True)
    
    # ================================================================
    # PHASE 2: Run demanding models sequentially, one by one
    # ================================================================
    if demanding_models:
        print("\n" + "=" * 70)
        print("PHASE 2: RUNNING DEMANDING MODELS SEQUENTIALLY")
        print("=" * 70)
        print(f"Running {len(demanding_models)} demanding models one by one in order...")
        print("-" * 70, flush=True)
        
        for idx, script in enumerate(demanding_models, 1):
            model_name = script.stem
            print(f"\n[{idx}/{len(demanding_models)}] Starting: {model_name}", flush=True)
            
            try:
                result = run_model_script(script)
                results.append(result)
                print(f"[{idx}/{len(demanding_models)}] Completed: {model_name}", flush=True)
            except Exception as e:
                print(f"[EXCEPTION] {model_name}: {str(e)}", flush=True)
                results.append({
                    "model": model_name,
                    "status": "exception",
                    "error": str(e)
                })
        
        print(f"\n[PHASE 2 COMPLETE] All {len(demanding_models)} demanding models finished.", flush=True)
    
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
    print(f"Total execution time: {total_time:.1f}s")
    
    if successful:
        print(f"\n✅ Successful models:")
        for r in successful:
            print(f"  - {r['model']} ({r.get('elapsed_time', 0):.1f}s)")
    
    if failed:
        print(f"\n❌ Failed models:")
        for r in failed:
            status = r.get("status", "unknown")
            print(f"  - {r['model']} ({status})")
            if r.get("stderr"):
                # Print first few lines of error
                error_lines = r['stderr'].split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"    {line}")
    
    print("=" * 70)
    
    # Save results to file
    results_file = PROJECT_ROOT / "runs" / "model_training_results.txt"
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
    
    # ================================================================
    # AUTOMATIC MODEL VALIDATION TESTS
    # ================================================================
    if len(successful) > 0:
        print("\n" + "=" * 70)
        print("RUNNING AUTOMATIC MODEL VALIDATION TESTS")
        print("=" * 70)
        
        try:
            # Run pytest tests for models
            test_dir = PROJECT_ROOT / "tests" / "test_models"
            if test_dir.exists():
                print(f"\nRunning model validation tests from: {test_dir}")
                
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for tests
                )
                
                print("\n" + "-" * 70)
                print("TEST RESULTS")
                print("-" * 70)
                print(result.stdout)
                
                if result.stderr:
                    print("\nTest Warnings/Errors:")
                    print(result.stderr)
                
                if result.returncode == 0:
                    print("\n" + "=" * 70)
                    print("✅ ALL MODEL VALIDATION TESTS PASSED")
                    print("=" * 70)
                else:
                    print("\n" + "=" * 70)
                    print("⚠️  SOME MODEL VALIDATION TESTS FAILED")
                    print("=" * 70)
                    print(f"Exit code: {result.returncode}")
                
                # Save test results
                test_results_file = PROJECT_ROOT / "runs" / "model_test_results.txt"
                with open(test_results_file, 'w') as f:
                    f.write("MODEL VALIDATION TEST RESULTS\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write(f"Tests passed: {result.returncode == 0}\n\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\nSTDERR:\n")
                        f.write(result.stderr)
                
                print(f"\nTest results saved to: {test_results_file}")
            else:
                print(f"\n⚠️  Test directory not found: {test_dir}")
                print("   Skipping automatic validation tests.")
                
        except subprocess.TimeoutExpired:
            print("\n⚠️  Test execution timed out after 5 minutes.")
        except Exception as e:
            print(f"\n⚠️  Error running validation tests: {str(e)}")
            print("   Continuing without test results...")
        
        # ================================================================
        # AUTOMATIC MODEL COMPARISON
        # ================================================================
        print("\n" + "=" * 70)
        print("RUNNING MODEL COMPARISON")
        print("=" * 70)
        
        try:
            # Run model comparison script
            compare_script = PROJECT_ROOT / "scripts" / "compare_models.py"
            if compare_script.exists():
                print(f"\nRunning model comparison: {compare_script.name}")
                
                result = subprocess.run(
                    [sys.executable, str(compare_script)],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                print("\n" + "-" * 70)
                print("COMPARISON RESULTS")
                print("-" * 70)
                print(result.stdout)
                
                if result.stderr:
                    print("\nComparison Warnings/Errors:")
                    print(result.stderr)
                
                if result.returncode == 0:
                    print("\n" + "=" * 70)
                    print("✅ MODEL COMPARISON COMPLETED")
                    print("=" * 70)
                    print("Comparison plots saved to: runs/latest/comparison/")
                else:
                    print("\n" + "=" * 70)
                    print("⚠️  MODEL COMPARISON HAD ISSUES")
                    print("=" * 70)
                    print(f"Exit code: {result.returncode}")
            else:
                print(f"\n⚠️  Comparison script not found: {compare_script}")
                print("   Skipping model comparison.")
                
        except subprocess.TimeoutExpired:
            print("\n⚠️  Model comparison timed out after 2 minutes.")
        except Exception as e:
            print(f"\n⚠️  Error running model comparison: {str(e)}")
            print("   Continuing without comparison...")
        
        # ================================================================
        # SHOW PERFORMANCE SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 70)
        
        try:
            # Run show_performance script
            perf_script = PROJECT_ROOT / "scripts" / "show_performance.py"
            if perf_script.exists():
                result = subprocess.run(
                    [sys.executable, str(perf_script)],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.stdout:
                    print(result.stdout)
                if result.stderr and result.returncode != 0:
                    print("Performance summary warnings:")
                    print(result.stderr)
            else:
                print("⚠️  Performance script not found. Skipping summary.")
                
        except Exception as e:
            print(f"⚠️  Error showing performance: {str(e)}")
    else:
        print("\n⚠️  No successful models to validate. Skipping tests and comparison.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Some models may still be running.")
        sys.exit(1)

