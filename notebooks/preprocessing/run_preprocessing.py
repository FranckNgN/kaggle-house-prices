import papermill as pm
from pathlib import Path
import sys
import io

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
EXT = ".ipynb"

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
# RUN NOTEBOOKS
# ---------------------------------------------------
def run_notebook(path: Path) -> None:
    """Execute a notebook using Papermill."""
    print(f"\nRunning notebook: {path.name}")
    try:
        pm.execute_notebook(
            input_path=str(path),
            output_path=str(path),  # overwrite in-place
            log_output=False,
            progress_bar=False
        )
        print(f"[OK] Finished: {path.name}")
    except Exception as e:
        print(f"[ERROR] ERROR in {path.name}: {e}")
        raise


def main() -> None:
    """Main entry point for preprocessing pipeline."""
    clean_interim()

    # Auto-detect numbered notebooks
    files = sorted(
        f for f in PREPROC_DIR.glob(f"*{EXT}")
        if f.name[0].isdigit()
    )

    if not files:
        print("WARNING: No numbered notebooks found!")
        return

    print("\nFiles detected:")
    for f in files:
        print(f"  - {f.name}")

    print("\n" + "=" * 50)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 50)

    for f in files:
        run_notebook(f)

    print("\n" + "=" * 50)
    print("[OK] ALL FILES COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
# ---------------------------------------------------