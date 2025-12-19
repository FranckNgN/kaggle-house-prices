#!/usr/bin/env python
"""
Run all model training scripts in parallel.
Each model runs in a separate process to maximize CPU utilization.
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
    print(f"[STARTING] {model_name}")
    
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
            print(f"[SUCCESS] {model_name} completed in {elapsed_time:.1f}s")
            return {
                "model": model_name,
                "status": "success",
                "elapsed_time": elapsed_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"[FAILED] {model_name} failed after {elapsed_time:.1f}s")
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
        print(f"[TIMEOUT] {model_name} exceeded 1 hour timeout")
        return {
            "model": model_name,
            "status": "timeout",
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] {model_name} raised exception: {str(e)}")
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


def main():
    """Main execution function."""
    print("=" * 70)
    print("PARALLEL MODEL TRAINING")
    print("=" * 70)
    
    # Get all model scripts
    model_scripts = get_model_scripts()
    
    if not model_scripts:
        print("No model scripts found!")
        return
    
    print(f"\nFound {len(model_scripts)} model scripts:")
    for script in model_scripts:
        print(f"  - {script.name}")
    
    # Check for --yes flag to skip confirmation
    skip_confirmation = '--yes' in sys.argv or '-y' in sys.argv
    
    if not skip_confirmation:
        # Ask for confirmation
        print(f"\nThis will run {len(model_scripts)} models in parallel.")
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
        print(f"\nRunning {len(model_scripts)} models in parallel (--yes flag detected)...")
    
    # Determine number of parallel workers
    # Use number of models or CPU count, whichever is smaller
    max_workers = min(len(model_scripts), os.cpu_count() or 4)
    print(f"\nRunning with {max_workers} parallel workers...")
    print("=" * 70)
    
    start_time = time.time()
    results = []
    
    # Run models in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(run_model_script, script): script.stem
            for script in model_scripts
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[EXCEPTION] {model_name}: {str(e)}")
                results.append({
                    "model": model_name,
                    "status": "exception",
                    "error": str(e)
                })
    
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

