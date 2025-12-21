from pathlib import Path

# Import environment detection for hybrid local/Kaggle workflow
try:
    from config_local.environment import (
        is_kaggle_environment,
        get_base_path,
        get_data_path,
        get_working_data_path,
        get_kaggle_input_path,
        setup_kaggle_symlinks
    )
except ImportError:
    # Fallback if environment.py doesn't exist yet
    def is_kaggle_environment():
        return False
    def get_base_path():
        return Path(__file__).resolve().parents[1]
    def get_data_path():
        return get_base_path() / "data"
    def get_working_data_path():
        return get_base_path() / "data"
    def get_kaggle_input_path(file_path):
        return get_base_path() / file_path
    def setup_kaggle_symlinks(base_path=None):
        pass

# Base paths - environment aware
ROOT = get_base_path()

# Data directories - use working data path for outputs, data path for inputs
if is_kaggle_environment():
    # On Kaggle: inputs from /kaggle/input, outputs to /kaggle/working/data
    DATA_DIR = get_working_data_path()
    RAW_DIR = get_data_path()  # Competition data from /kaggle/input
    # Setup symlinks for compatibility (allows code to use local paths)
    setup_kaggle_symlinks(ROOT)
    # After symlink setup, use working directory for raw data access
    # (will resolve via symlinks to /kaggle/input)
    working_raw = DATA_DIR / "raw"
    if working_raw.exists() or not RAW_DIR.exists():
        RAW_DIR = working_raw
else:
    # Local environment: standard paths
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"

INTERIM_DIR = DATA_DIR / "interim"
INTERIM_TRAIN_DIR = INTERIM_DIR / "train"
INTERIM_TEST_DIR = INTERIM_DIR / "test"
OOF_DIR = INTERIM_DIR / "oof"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

# Runs and outputs - always use working directory
RUNS_DIR = ROOT / "runs"
KAGGLE_DIR = ROOT / ".kaggle"

# Raw data files - use environment-aware paths
# On Kaggle, these may be symlinked to /kaggle/input
TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
DATA_DESCRIPTION = RAW_DIR / "data_description.txt"

# Interim - Train files
TRAIN_PROCESS1_CSV = INTERIM_TRAIN_DIR / "train_process1.csv"
TRAIN_PROCESS2_CSV = INTERIM_TRAIN_DIR / "train_process2.csv"
TRAIN_PROCESS3_CSV = INTERIM_TRAIN_DIR / "train_process3.csv"
TRAIN_PROCESS4_CSV = INTERIM_TRAIN_DIR / "train_process4.csv"
TRAIN_PROCESS5_CSV = INTERIM_TRAIN_DIR / "train_process5.csv"
TRAIN_PROCESS6_CSV = INTERIM_TRAIN_DIR / "train_process6.csv"
TRAIN_PROCESS7_CSV = INTERIM_TRAIN_DIR / "train_process7.csv"
TRAIN_PROCESS8_CSV = INTERIM_TRAIN_DIR / "train_process8.csv"

# Interim - Test files
TEST_PROCESS1_CSV = INTERIM_TEST_DIR / "test_process1.csv"
TEST_PROCESS2_CSV = INTERIM_TEST_DIR / "test_process2.csv"
TEST_PROCESS3_CSV = INTERIM_TEST_DIR / "test_process3.csv"
TEST_PROCESS4_CSV = INTERIM_TEST_DIR / "test_process4.csv"
TEST_PROCESS5_CSV = INTERIM_TEST_DIR / "test_process5.csv"
TEST_PROCESS6_CSV = INTERIM_TEST_DIR / "test_process6.csv"
TEST_PROCESS7_CSV = INTERIM_TEST_DIR / "test_process7.csv"
TEST_PROCESS8_CSV = INTERIM_TEST_DIR / "test_process8.csv"

# Processed
FEATURE_SUMMARY_CSV = PROCESSED_DIR / "feature_summary.csv"
SALEPRICE_TRANSFORMS_CSV = PROCESSED_DIR / "salePrice_transforms.csv"

# Submissions
SAMPLE_SUBMISSION_CSV = SUBMISSIONS_DIR / "sample_submission.csv"
SUBMISSION_LOG_JSON = SUBMISSIONS_DIR / "submission_log.json"

# Kaggle
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"

# Runs
MODEL_PERFORMANCE_CSV = RUNS_DIR / "model_performance.csv"
MODEL_TRAINING_RESULTS_TXT = RUNS_DIR / "model_training_results.txt"
MODEL_TEST_RESULTS_TXT = RUNS_DIR / "model_test_results.txt"


def get_model_submission_path(model_name: str, filename: str = None) -> Path:
    """
    Get the path for a model's submission file, creating a subfolder for the model.
    
    Args:
        model_name: Name of the model (e.g., 'xgboost', 'random_forest')
        filename: Optional filename. If None, defaults to '{model_name}_Model.csv'
        
    Returns:
        Path object pointing to the submission file in the model's subfolder
    """
    model_folder = SUBMISSIONS_DIR / model_name.lower().replace(" ", "_")
    model_folder.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"{model_name.lower().replace(' ', '_')}_Model.csv"
        
    return model_folder / filename


def validate_paths() -> bool:
    """Validate that all required directories exist."""
    required_dirs = [
        RAW_DIR, 
        INTERIM_DIR, 
        INTERIM_TRAIN_DIR, 
        INTERIM_TEST_DIR, 
        PROCESSED_DIR, 
        SUBMISSIONS_DIR,
        RUNS_DIR
    ]
    missing = [d for d in required_dirs if not d.exists()]
    
    if missing:
        print("WARNING: Missing directories:")
        for d in missing:
            print(f"  - {d}")
            d.mkdir(parents=True, exist_ok=True)
        print("[OK] Created missing directories")
    
    return True


# Auto-validate on import
if __name__ != "__main__":
    validate_paths()