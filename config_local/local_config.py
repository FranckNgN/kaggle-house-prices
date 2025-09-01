import os

# Get the project root (parent of the config_local folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to data and files
DATA_DIR = os.path.join(BASE_DIR, 'data')

TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

TRAIN_FILLED_CSV = os.path.join(DATA_DIR, 'train_filled.csv')
TRAIN_OUTLIER_FILLED_CSV = os.path.join(DATA_DIR, 'train_outlier_filled.csv')
TRAIN_OUTLIER_FILLED_LOG1 = os.path.join(DATA_DIR, 'train_outlier_filled_log1.csv')
TEST_FILLED_CSV = os.path.join(DATA_DIR, 'test_filled.csv')

# (Optional) You can also add these for reuse elsewhere:
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
