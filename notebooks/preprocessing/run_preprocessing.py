import papermill as pm
from pathlib import Path
import sys
import io
import subprocess
import os

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config for paths
try:
    import config_local.local_config as config
except ImportError:
    print("ERROR: config_local.local_config not found. Please create it from local_config.py.example")
    sys.exit(1)

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
ROOT = Path(__file__).resolve().parent                # .../notebooks/preprocessing
PREPROC_DIR = ROOT                                    # where the numbered notebooks are
EXT_PY = ".py"
EXT_IPYNB = ".ipynb"

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
        run_file(f)

    print("\n" + "=" * 50)
    print("ALL FILES COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
# ---------------------------------------------------