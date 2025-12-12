import papermill as pm
from pathlib import Path

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
ROOT = Path(__file__).resolve().parent                # .../notebooks/preprocessing
PREPROC_DIR = ROOT                                    # where the numbered notebooks are
EXT = ".ipynb"

# Absolute path to interim data
INTERIM_DIR = Path(r"D:\Project\Kaggle\house-prices-starter\data\interim")

# (Alternative relative version, if you ever move the repo)
# INTERIM_DIR = ROOT.parent.parent / "data" / "interim"

# ---------------------------------------------------
# CLEAN INTERIM DIRECTORY
# ---------------------------------------------------
def clean_interim():
    print("Cleaning interim CSV files...")
    deleted_count = 0

    if not INTERIM_DIR.exists():
        print(f"  WARNING: Interim folder does not exist: {INTERIM_DIR}")
        return

    for file in INTERIM_DIR.glob("*.csv"):
        try:
            file.unlink()
            deleted_count += 1
            print(f"  - Deleted: {file.name}")
        except Exception as e:
            print(f"  ERROR: Could not delete {file.name}: {e}")

    if deleted_count == 0:
        print("  (No CSV files found)")
    else:
        print(f"Done. {deleted_count} CSV files deleted.")


# ---------------------------------------------------
# RUN NOTEBOOKS
# ---------------------------------------------------
def run_notebook(path: Path):
    print(f"\nRunning notebook: {path.name}")
    try:
        pm.execute_notebook(
            input_path=str(path),
            output_path=str(path),  # overwrite in-place
            log_output=False,
            progress_bar=False
        )
        print(f"Finished: {path.name}")
    except Exception:
        print(f"ERROR in {path.name}")
        raise


def main():
    clean_interim()

    # auto-detect numbered notebooks
    files = sorted(
        f for f in PREPROC_DIR.glob(f"*{EXT}")
        if f.name[0].isdigit()
    )

    print("\nFiles detected:")
    for f in files:
        print("  -", f.name)

    print("\n=== STARTING PREPROCESSING PIPELINE ===")

    for f in files:
        run_notebook(f)

    print("\nALL FILES COMPLETED.")


if __name__ == "__main__":
    main()
# ---------------------------------------------------