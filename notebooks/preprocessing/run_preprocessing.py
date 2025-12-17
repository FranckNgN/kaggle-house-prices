import papermill as pm
from pathlib import Path

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
ROOT = Path(__file__).resolve().parent                # .../notebooks/preprocessing
PREPROC_DIR = ROOT                                    # where the numbered notebooks are
EXT = ".ipynb"

# Relative path to interim data (portable)
INTERIM_DIR = ROOT.parent.parent / "data" / "interim"

# ---------------------------------------------------
# CLEAN INTERIM DIRECTORY
# ---------------------------------------------------
def clean_interim() -> None:
    """Clean interim directory by removing all CSV files."""
    print("Cleaning interim CSV files...")
    deleted_count = 0

    if not INTERIM_DIR.exists():
        print(f"⚠️  WARNING: Interim folder does not exist: {INTERIM_DIR}")
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        return

    for file in INTERIM_DIR.glob("*.csv"):
        try:
            file.unlink()
            deleted_count += 1
            print(f"  - Deleted: {file.name}")
        except Exception as e:
            print(f"  ❌ ERROR: Could not delete {file.name}: {e}")

    if deleted_count == 0:
        print("  (No CSV files found)")
    else:
        print(f"✅ Done. {deleted_count} CSV files deleted.")


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
        print(f"✅ Finished: {path.name}")
    except Exception as e:
        print(f"❌ ERROR in {path.name}: {e}")
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
        print("⚠️  No numbered notebooks found!")
        return

    print("\n📋 Files detected:")
    for f in files:
        print(f"  - {f.name}")

    print("\n" + "=" * 50)
    print("🚀 STARTING PREPROCESSING PIPELINE")
    print("=" * 50)

    for f in files:
        run_notebook(f)

    print("\n" + "=" * 50)
    print("✅ ALL FILES COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
# ---------------------------------------------------