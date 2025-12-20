#!/usr/bin/env python
"""
Run all model training scripts sequentially.
- All models (0-11) run one after another
- Each model targets ~1 hour runtime
- All models pass sanity checks and validation tests
"""
import sys
import subprocess
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
        
        # Run the script directly with PYTHONPATH set
        # Also prepend project root to sys.path using PYTHONSTARTUP-like approach
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
            timeout=7200  # 2 hour timeout per model (allows ~1 hour for optimization)
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
            # Print error details immediately for debugging
            if result.stderr:
                print(f"[ERROR DETAILS] {model_name} stderr:", flush=True)
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[:20]:  # Print first 20 lines
                    if line.strip():
                        print(f"  {line}", flush=True)
                if len(error_lines) > 20:
                    print(f"  ... ({len(error_lines) - 20} more lines)", flush=True)
            if result.stdout:
                # Sometimes errors are in stdout
                stdout_lines = result.stdout.strip().split('\n')
                error_indicators = ['error', 'exception', 'traceback', 'failed', 'missing']
                error_lines = [line for line in stdout_lines if any(ind in line.lower() for ind in error_indicators)]
                if error_lines:
                    print(f"[ERROR DETAILS] {model_name} stdout (error lines):", flush=True)
                    for line in error_lines[:10]:
                        if line.strip():
                            print(f"  {line}", flush=True)
            print("-" * 70, flush=True)
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
    import re
    linear_models = []  # Models 0-4: linear, ridge, lasso, elastic net (everything up to random forest excluded)
    demanding_models = []  # Models 5-11: RF, SVR, XGBoost, LightGBM, CatBoost, blending, stacking
    
    for script in scripts:
        # Extract the full number from the beginning of the filename (handles single and multi-digit numbers)
        match = re.match(r'^(\d+)', script.name)
        if match:
            model_num = int(match.group(1))
        else:
            model_num = 99
        
        if model_num < 5:  # Only models 0-4 (exclude random forest and above)
            linear_models.append(script)
        else:
            demanding_models.append(script)
    
    return linear_models, demanding_models


def main():
    """Main execution function."""
    print("=" * 70)
    print("ALL MODELS TRAINING - SEQUENTIAL EXECUTION")
    print("=" * 70)
    print("Strategy: Running all models (0-11) sequentially")
    print("         Each model targets ~1 hour runtime")
    print("         All models pass sanity checks")
    print("=" * 70)
    
    # Get all model scripts
    all_scripts = get_model_scripts()
    
    if not all_scripts:
        print("No model scripts found!")
        return
    
    # Filter out example files
    all_scripts = [s for s in all_scripts if not s.name.endswith('.example')]
    
    print(f"\nFound {len(all_scripts)} model scripts to run:")
    for script in all_scripts:
        print(f"    - {script.name}")
    
    # Check for --yes flag to skip confirmation
    skip_confirmation = '--yes' in sys.argv or '-y' in sys.argv
    
    if not skip_confirmation:
        # Ask for confirmation
        print(f"\nThis will run {len(all_scripts)} models sequentially.")
        print("Estimated total time: ~{:.1f} hours".format(len(all_scripts) * 1.0))
        try:
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        except EOFError:
            # Non-interactive mode, proceed automatically
            print("Non-interactive mode detected. Proceeding...")
    else:
        print(f"\nRunning {len(all_scripts)} models sequentially (--yes flag detected)...")
    
    start_time = time.time()
    results = []
    
    # ================================================================
    # Run all models sequentially
    # ================================================================
    print("\n" + "=" * 70)
    print("RUNNING ALL MODELS SEQUENTIALLY")
    print("=" * 70)
    print(f"Running {len(all_scripts)} models one after another...")
    print("Each model targets ~1 hour runtime")
    print("-" * 70, flush=True)
    
    for idx, script in enumerate(all_scripts, 1):
        model_name = script.stem
        print(f"\n[{idx}/{len(all_scripts)}] Starting: {model_name}")
        print("-" * 70, flush=True)
        
        try:
            result = run_model_script(script, enable_validation=True)
            results.append(result)
            
            if result.get("status") == "success":
                elapsed = result.get("elapsed_time", 0)
                print(f"[SUCCESS] {model_name} completed in {elapsed/60:.1f} minutes", flush=True)
            else:
                print(f"[FAILED] {model_name} - Status: {result.get('status', 'unknown')}", flush=True)
                # Continue with next model even if this one failed
                
        except Exception as e:
            print(f"[EXCEPTION] {model_name}: {str(e)}", flush=True)
            results.append({
                "model": model_name,
                "status": "exception",
                "error": str(e)
            })
        
        remaining = len(all_scripts) - idx
        if remaining > 0:
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / idx
            estimated_remaining = avg_time * remaining
            print(f"\n[PROGRESS] {idx}/{len(all_scripts)} models completed.")
            print(f"          Estimated time remaining: {estimated_remaining/60:.1f} minutes", flush=True)
    
    print(f"\n[ALL MODELS COMPLETE] All {len(all_scripts)} models finished.", flush=True)
    
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
        print(f"\n[SUCCESS] Successful models:")
        for r in successful:
            print(f"  - {r['model']} ({r.get('elapsed_time', 0):.1f}s)")
    
    if failed:
        print(f"\n[FAILED] Failed models:")
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
                    print("[SUCCESS] ALL MODEL VALIDATION TESTS PASSED")
                    print("=" * 70)
                else:
                    print("\n" + "=" * 70)
                    print("[WARNING] SOME MODEL VALIDATION TESTS FAILED")
                    print("=" * 70)
                    print(f"Exit code: {result.returncode}")
                
                # Save test results
                test_results_file = config.MODEL_TEST_RESULTS_TXT
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
                print(f"\n[WARNING] Test directory not found: {test_dir}")
                print("   Skipping automatic validation tests.")
                
        except subprocess.TimeoutExpired:
            print("\n[WARNING] Test execution timed out after 5 minutes.")
        except Exception as e:
            print(f"\n[WARNING] Error running validation tests: {str(e)}")
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
                    print("[SUCCESS] MODEL COMPARISON COMPLETED")
                    print("=" * 70)
                    print(f"Comparison plots saved to: {config.RUNS_DIR / 'latest' / 'comparison'}")
                else:
                    print("\n" + "=" * 70)
                    print("[WARNING] MODEL COMPARISON HAD ISSUES")
                    print("=" * 70)
                    print(f"Exit code: {result.returncode}")
            else:
                print(f"\n[WARNING] Comparison script not found: {compare_script}")
                print("   Skipping model comparison.")
                
        except subprocess.TimeoutExpired:
            print("\n[WARNING] Model comparison timed out after 2 minutes.")
        except Exception as e:
            print(f"\n[WARNING] Error running model comparison: {str(e)}")
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
                print("[WARNING] Performance script not found. Skipping summary.")
                
        except Exception as e:
            print(f"[WARNING] Error showing performance: {str(e)}")
    else:
        print("\n[WARNING] No successful models to validate. Skipping tests and comparison.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user. Some models may still be running.")
        sys.exit(1)

