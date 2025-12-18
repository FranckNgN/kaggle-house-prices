from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_TRAIN_DIR = INTERIM_DIR / "train"
INTERIM_TEST_DIR = INTERIM_DIR / "test"
OOF_DIR = INTERIM_DIR / "oof"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

# Raw
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

# Interim - Test files
TEST_PROCESS1_CSV = INTERIM_TEST_DIR / "test_process1.csv"
TEST_PROCESS2_CSV = INTERIM_TEST_DIR / "test_process2.csv"
TEST_PROCESS3_CSV = INTERIM_TEST_DIR / "test_process3.csv"
TEST_PROCESS4_CSV = INTERIM_TEST_DIR / "test_process4.csv"
TEST_PROCESS5_CSV = INTERIM_TEST_DIR / "test_process5.csv"
TEST_PROCESS6_CSV = INTERIM_TEST_DIR / "test_process6.csv"

# Processed
FEATURE_SUMMARY_CSV = PROCESSED_DIR / "feature_summary.csv"
SALEPRICE_TRANSFORMS_CSV = PROCESSED_DIR / "salePrice_transforms.csv"

# Submissions
SAMPLE_SUBMISSION_CSV = SUBMISSIONS_DIR / "sample_submission.csv"


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
        SUBMISSIONS_DIR
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