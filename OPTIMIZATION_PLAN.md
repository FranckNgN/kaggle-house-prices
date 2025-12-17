# Optimization & Quality of Life Improvements

## 🔴 Critical Issues (Fix First)

### 1. Hardcoded Absolute Path
**File**: `notebooks/preprocessing/run_preprocessing.py:12`
```python
# Current (BAD):
INTERIM_DIR = Path(r"D:\Project\Kaggle\house-prices-starter\data\interim")

# Should be:
INTERIM_DIR = ROOT.parent.parent / "data" / "interim"
```

### 2. Missing Dependencies
**File**: `requirements.txt`
**Missing**: `papermill`, `xgboost`, `lightgbm`, `catboost`

### 3. Data Files in Git
**Issue**: Large CSV files tracked in repository
**Solution**: Update `.gitignore` to exclude:
- `data/interim/*.csv`
- `data/submissions/*.csv`
- `data/processed/*.csv` (or keep only summaries)

### 4. Broken Checks Module
**File**: `utils/checks.py`
**Issue**: References non-existent config paths
**Fix**: Update to use current config structure or remove outdated checks

### 5. Duplicate Scripts
**Issue**: Two `run_preprocessing.py` files (one outdated in `scripts/`)
**Action**: Remove `scripts/run_preprocessing.py` or consolidate

---

## 🟡 Code Quality Improvements

### 6. Add Type Hints
**Files**: All `.py` files
**Example**:
```python
def add_kmeans(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 4,
    cols: tuple = ("GrLivArea", "TotalBsmtSF", ...),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

### 7. Remove Duplicate Imports
**File**: `utils/checks.py`
**Issue**: Lines 6, 10-11 have duplicate imports

### 8. Add Error Handling
**Files**: All notebooks
**Add**: Try-except blocks, validation checks, informative error messages

### 9. Add Logging
**Create**: `utils/logger.py`
```python
import logging
from pathlib import Path

def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # ... setup handlers
    return logger
```

### 10. Add Docstrings
**Files**: All functions in `.py` files
**Format**: Google or NumPy style

---

## 🟢 Project Structure Improvements

### 11. Create Comprehensive README.md
**Should include**:
- Project description
- Installation instructions
- Usage examples
- Project structure
- How to run preprocessing pipeline
- How to train models
- Contributing guidelines

### 12. Update .gitignore
**Add**:
```
# Data files
data/interim/*.csv
data/submissions/*.csv
*.csv
!data/raw/*.csv  # Keep raw data

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.env
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Kaggle
.kaggle/kaggle.json
```

### 13. Create Setup Script
**File**: `setup.py` or `setup.sh`
**Purpose**: Automated environment setup

### 14. Create Model Utilities Module
**File**: `utils/models.py`
**Functions**:
- `save_model(model, path)`
- `load_model(path)`
- `evaluate_model(model, X, y)`
- `create_submission(predictions, test_ids, filename)`

### 15. Create Data Utilities Module
**File**: `utils/data.py`
**Functions**:
- `load_data(stage: int)` - Load processed data at any stage
- `save_data(df, stage: int, train: bool)`
- `validate_data(df, expected_columns)`

---

## 🔵 Quality of Life Improvements

### 16. Configuration Validation
**File**: `config_local/local_config.py`
**Add**: Function to validate all paths exist

### 17. Make Blending Weights Configurable
**File**: `notebooks/Models/8blendingModel.py`
**Change**: Load weights from config file or command-line args

### 18. Add Progress Bars
**Files**: Long-running operations
**Use**: `tqdm` for progress tracking

### 19. Create Common Functions Module
**File**: `utils/common.py`
**Functions**:
- `read_csv_safe(path)` - With error handling
- `save_csv_safe(df, path)` - With validation
- `get_cv_splits(n_splits, random_state)` - Consistent CV

### 20. Add Experiment Tracking
**Options**:
- MLflow
- Weights & Biases
- Simple CSV logger
**Track**: Model params, CV scores, Kaggle scores, timestamps

### 21. Create Model Training Template
**File**: `templates/train_model_template.py`
**Purpose**: Standardized model training script

### 22. Add Data Validation Checks
**File**: `utils/validation.py`
**Functions**:
- Check for missing values
- Check data types
- Check column consistency
- Check for data leakage

### 23. Create Preprocessing Utilities
**File**: `utils/preprocessing.py`
**Functions**:
- `fill_missing_values(df, strategy)`
- `detect_outliers(df, method)`
- `transform_skewed_features(df, threshold)`
- `encode_categorical(df, method)`

### 24. Add Unit Tests
**Files**: `tests/test_*.py`
**Coverage**:
- Data loading functions
- Preprocessing functions
- Model utilities
- Validation functions

### 25. Create Makefile or Scripts
**File**: `Makefile` or `scripts/`
**Commands**:
- `make preprocess` - Run preprocessing pipeline
- `make train` - Train all models
- `make blend` - Create blended submission
- `make clean` - Clean interim files
- `make test` - Run tests

---

## 📊 Performance Optimizations

### 26. Optimize Data Types
**Action**: Use appropriate dtypes (int8, int16, float32) to reduce memory

### 27. Parallel Processing
**Files**: Model training notebooks
**Add**: Parallel CV folds where possible

### 28. Caching
**Add**: Cache expensive computations (feature engineering, transformations)

### 29. Optimize CSV Reading
**Use**: `pd.read_csv(..., dtype=...)` with specified dtypes

---

## 🎯 Quick Wins (Start Here)

1. ✅ Fix hardcoded path in `run_preprocessing.py`
2. ✅ Add missing dependencies to `requirements.txt`
3. ✅ Update `.gitignore` to exclude data files
4. ✅ Remove duplicate imports in `checks.py`
5. ✅ Create basic `README.md`
6. ✅ Fix broken `checks.py` references
7. ✅ Remove duplicate `scripts/run_preprocessing.py`
8. ✅ Add type hints to `8blendingModel.py`
9. ✅ Make blending weights configurable
10. ✅ Add basic error handling to notebooks

---

## 📝 Implementation Priority

**High Priority** (Do First):
- Critical issues (#1-5)
- Quick wins (#1-7)

**Medium Priority** (Do Next):
- Code quality (#6-10)
- Project structure (#11-15)

**Low Priority** (Nice to Have):
- Quality of life (#16-25)
- Performance (#26-29)

---

## 🔧 Tools to Add

- **Black** - Code formatting
- **flake8** or **pylint** - Linting
- **mypy** - Type checking
- **pytest** - Testing (already have)
- **pre-commit** - Git hooks
- **tqdm** - Progress bars
- **python-dotenv** - Environment variables

