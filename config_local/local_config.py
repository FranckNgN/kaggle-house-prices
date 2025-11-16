from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

# Raw
TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
DATA_DESCRIPTION = RAW_DIR / "data_description.txt"

# Interim
TRAIN_FILLED_CSV = INTERIM_DIR / "train_process1.csv"
TEST_FILLED_CSV = INTERIM_DIR / "test_process1.csv"

TRAIN_OUTLIER_FILLED_CSV = INTERIM_DIR / "train_process2.csv"

TRAIN_OUTLIER_FILLED_LOG1_CSV = INTERIM_DIR / "train_process3.csv"

TRAIN_OUTLIER_FILLED_LOG1_ENGINEERED_CSV = INTERIM_DIR / "train_process4.csv"
TEST_FILLED_ENGINEERED_CSV = INTERIM_DIR / "test_process4.csv"

TRAIN_OUTLIER_FILLED_LOG1_ENGINEERED_SCALED_CSV = INTERIM_DIR / "train_process5.csv"


TRAIN_OUTLIER_FILLED_LOG1_ENGINEERED_SCALED_CATENCODED_CSV = INTERIM_DIR / "train_process6.csv"
TEST_FILLED_ENGINEERED_CATENCODED_CSV = INTERIM_DIR / "test_process6.csv"

# Processed
FEATURE_SUMMARY_CSV = PROCESSED_DIR / "feature_summary.csv"
SALEPRICE_TRANSFORMS_CSV = PROCESSED_DIR / "salePrice_transforms.csv"

# Submissions
SAMPLE_SUBMISSION_CSV = SUBMISSIONS_DIR / "sample_submission.csv"