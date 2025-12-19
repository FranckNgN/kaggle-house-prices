import papermill as pm
from pathlib import Path
import sys
import io
import subprocess
import os
import pandas as pd

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config for paths
try:
    import config_local.local_config as config
    from utils.validation import validate_dataframe, validate_column_parity, check_skewness
    from utils.engineering import reset_engineering_summary
    from utils.checks import (
        validate_preprocessing_stage,
        DataLeakageError,
        DataIntegrityError,
        print_error
    )
except ImportError:
    print("ERROR: Required modules not found.")
    sys.exit(1)

# ---------------------------------------------------
# VALIDATE STAGE OUTPUTS
# ---------------------------------------------------
# PATHS
ROOT = Path(__file__).resolve().parent                # .../notebooks/preprocessing
PREPROC_DIR = ROOT                                    # where the numbered notebooks are
EXT_PY = ".py"
EXT_IPYNB = ".ipynb"

def validate_stage(stage_num: int) -> None:
    """Validate the output files of a specific preprocessing stage."""
    print(f"\n--- Validating Stage {stage_num} ---")
    
    train_attr = f"TRAIN_PROCESS{stage_num}_CSV"
    test_attr = f"TEST_PROCESS{stage_num}_CSV"
    
    if not hasattr(config, train_attr) or not hasattr(config, test_attr):
        print(f"  ⚠️  Paths for stage {stage_num} not found in config. Skipping.")
        return

    train_path = getattr(config, train_attr)
    test_path = getattr(config, test_attr)
    
    if not train_path.exists() or not test_path.exists():
        print(f"  ⚠️  Output files for stage {stage_num} do not exist yet. Skipping.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Check for NAs (Starting from stage 1, we expect no NAs except maybe the target in some stages)
    validate_dataframe(train_df, f"Train Process {stage_num}")
    validate_dataframe(test_df, f"Test Process {stage_num}")
    
    # Check for column parity
    target_col = "logSP" if stage_num >= 2 else "SalePrice"
    validate_column_parity(train_df, test_df, target_col=target_col)
    
    # Check skewness (particularly relevant for stage 3 onwards)
    if stage_num >= 3:
        check_skewness(train_df, f"Train Process {stage_num}")

# ---------------------------------------------------
# CLEAN INTERIM DIRECTORY
# ---------------------------------------------------
def clean_interim() -> None:
    """Clean interim train and test directories by removing all CSV files."""
    print("Cleaning interim CSV files...")
    deleted_count = 0

    # Clean train directory
    train_dir = config.INTERIM_TRAIN_DIR
    if train_dir.exists():
        for file in train_dir.glob("*.csv"):
            try:
                file.unlink()
                deleted_count += 1
                print(f"  - Deleted: {file.name} from train/")
            except Exception as e:
                print(f"  ERROR: Could not delete {file.name}: {e}")
    else:
        train_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created train directory: {train_dir}")

    # Clean test directory
    test_dir = config.INTERIM_TEST_DIR
    if test_dir.exists():
        for file in test_dir.glob("*.csv"):
            try:
                file.unlink()
                deleted_count += 1
                print(f"  - Deleted: {file.name} from test/")
            except Exception as e:
                print(f"  ERROR: Could not delete {file.name}: {e}")
    else:
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created test directory: {test_dir}")

    if deleted_count == 0:
        print("  (No CSV files found)")
    else:
        print(f"[OK] Done. {deleted_count} CSV files deleted.")


# ---------------------------------------------------
# RUN SCRIPTS/NOTEBOOKS
# ---------------------------------------------------
def run_file(path: Path) -> None:
    """Execute a python script or notebook."""
    if path.suffix == ".py":
        print(f"\nRunning script: {path.name}")
        try:
            # Set PYTHONPATH to include project root so scripts can find config_local
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            
            # Use same python executable as current one
            result = subprocess.run([sys.executable, str(path)], check=True, capture_output=True, text=True, env=env)
            print(result.stdout)
            print(f"Finished: {path.name}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR in {path.name}")
            print(e.stdout)
            print(e.stderr)
            raise
    elif path.suffix == ".ipynb":
        print(f"\nRunning notebook: {path.name}")
        try:
            pm.execute_notebook(
                input_path=str(path),
                output_path=str(path),  # overwrite in-place
                log_output=False,
                progress_bar=False
            )
            print(f"Finished: {path.name}")
        except Exception as e:
            print(f"ERROR in {path.name}: {e}")
            raise


def main() -> None:
    """Main entry point for preprocessing pipeline."""
    reset_engineering_summary()
    clean_interim()

    # Auto-detect numbered py files or notebooks
    py_files = set(f for f in PREPROC_DIR.glob(f"*{EXT_PY}") if f.name[0].isdigit())
    ipynb_files = set(f for f in PREPROC_DIR.glob(f"*{EXT_IPYNB}") if f.name[0].isdigit())
    
    # Prioritize .py if both exist for same number
    files_to_run = []
    seen_numbers = set()
    
    # Sort all files by name to ensure order 1, 2, 3...
    all_files = sorted(list(py_files | ipynb_files), key=lambda x: x.name)
    
    for f in all_files:
        num = f.name[0]
        if num not in seen_numbers:
            # Look for .py version first
            py_ver = PREPROC_DIR / (f.stem + EXT_PY)
            if py_ver.exists():
                files_to_run.append(py_ver)
            else:
                files_to_run.append(f)
            seen_numbers.add(num)

    if not files_to_run:
        print("WARNING: No numbered preprocessing files found!")
        return

    print("\nFiles detected to run:")
    for f in files_to_run:
        print(f"  - {f.name}")

    print("\n" + "=" * 50)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 50)

    for f in files_to_run:
        stage_num = int(f.name[0])
        
        # ================================================================
        # STEP 1: Run the preprocessing file
        # ================================================================
        print(f"\n{'='*60}")
        print(f"RUNNING: {f.name}")
        print(f"{'='*60}")
        
        try:
            run_file(f)
            print(f"\n✅ {f.name} completed successfully")
        except Exception as e:
            print_error(
                f"Failed to run {f.name}",
                "EXECUTION ERROR"
            )
            print(f"\nError details: {str(e)}")
            print(f"\n❌ PIPELINE STOPPED: Cannot continue after file execution failure")
            sys.exit(1)
        
        # ================================================================
        # STEP 2: Validate immediately after file runs
        # ================================================================
        try:
            # This will STOP the pipeline if any check fails
            validate_preprocessing_stage(stage_num, stop_on_error=True)
            
        except DataLeakageError as e:
            print_error(
                f"DATA LEAKAGE detected in Stage {stage_num}",
                "CRITICAL ERROR"
            )
            print(f"\n{'='*60}")
            print(f"PIPELINE STOPPED")
            print(f"{'='*60}")
            print(f"\n⚠️  Fix the issue in {f.name} before continuing.")
            sys.exit(1)
            
        except DataIntegrityError as e:
            print_error(
                f"Data integrity issue in Stage {stage_num}",
                "CRITICAL ERROR"
            )
            print(f"\n{'='*60}")
            print(f"PIPELINE STOPPED")
            print(f"{'='*60}")
            print(f"\n⚠️  Fix the issue in {f.name} before continuing.")
            sys.exit(1)
            
        except Exception as e:
            print_error(
                f"Validation error in Stage {stage_num}: {str(e)}",
                "VALIDATION ERROR"
            )
            print(f"\n{'='*60}")
            print(f"PIPELINE STOPPED")
            print(f"{'='*60}")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ ALL FILES COMPLETED AND VALIDATED")
    print("=" * 50)


if __name__ == "__main__":
    main()
# ---------------------------------------------------