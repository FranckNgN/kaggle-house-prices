# Technical Log - House Prices Prediction Project

**Detailed technical documentation, implementation details, error logs, and optimization history.**

> **Note**: For project overview and showcase, see `FLAGSHIP_LOG.md`

---

## Table of Contents

1. [Implementation Details](#implementation-details)
2. [Optimization History](#optimization-history)
3. [Error Logs & Fixes](#error-logs--fixes)
4. [Configuration Details](#configuration-details)
5. [Performance Tracking](#performance-tracking)
6. [Technical Decisions](#technical-decisions)

---

## Implementation Details

### Preprocessing Pipeline Stages

#### Stage 1: Cleaning
- **Missing Value Strategy**:
  - Numeric columns: Fill with `0`
  - Categorical columns: Replace `"NA"` and empty strings with `pd.NA`, then fill with `"<None>"`
- **Output**: `train_process1.csv`, `test_process1.csv`
- **Feature Summary**: Generated `feature_summary.csv` categorizing features

#### Stage 2: Data Engineering
- **Target Transformation**: `logSP = log1p(SalePrice)`
- **Outlier Removal**: Removed rows where `GrLivArea > 4000` AND `SalePrice < 300000` (2 outliers)
- **Output**: `train_process2.csv`, `test_process2.csv`

#### Stage 3: Skew/Kurtosis Normalization
- **Method**: PowerTransformer with Yeo-Johnson method
- **Threshold**: Applied to numeric columns with `|skew| > 0.75`
- **Features Transformed**: LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, 1stFlrSF, 2ndFlrSF, GrLivArea, GarageYrBlt, WoodDeckSF, OpenPorchSF, EnclosedPorch, ScreenPorch
- **Output**: `train_process3.csv`, `test_process3.csv`

#### Stage 4: Feature Engineering
- **Age Features**: `Age = YrSold - YearBuilt`, `Garage_Age = YrSold - GarageYrBlt`, `RemodAge = YrSold - YearRemodAdd`
- **Aggregate Features**: TotalSF, TotalBath, TotalPorchSF (log-normalized)
- **Group Benchmarks**: Neighborhood and MSSubClass ratios
- **Ordinal Scores**: Converted quality ratings to numeric (0-5)
- **Interaction Features**: Qual_x_TotalSF, Kitchen_x_TotalSF, Cond_x_Age
- **Binary Flags**: HasPool, Has2ndFlr, HasGarage, HasBsmt, HasFireplace, IsNormalCondition
- **K-Means Clustering**: k=4 on key features (GrLivArea, TotalBsmtSF, 1stFlrSF, GarageCars, YearBuilt, OverallQual)
- **Output**: `train_process4.csv`, `test_process4.csv`

#### Stage 5: Scaling
- **Method**: StandardScaler
- **Scope**: Only continuous numeric features (ratio of unique values > 5%)
- **Output**: `train_process5.csv`, `test_process5.csv`

#### Stage 6: Categorical Encoding
- **Method**: One-hot encoding with `drop_first=True`
- **Implementation**: Concatenate train and test sets before encoding
- **Data Type**: Encoded as `int8` for memory efficiency
- **Output**: `train_process6.csv`, `test_process6.csv` (264 features)

---

## Optimization History

See `CHANGES.md` for detailed optimization history and improvements.

### Hyperparameter Optimization Settings

#### Random Forest
- **n_trials**: 50
- **n_splits**: 5
- **n_estimators**: (200, 1000)
- **max_depth**: (3, 20)
- **Target Runtime**: ~20 minutes

#### SVR
- **n_trials**: 40
- **n_splits**: 5
- **C**: (0.1, 100)
- **gamma**: (1e-5, 1e-2)
- **epsilon**: (0.001, 0.1)

#### XGBoost
- **n_trials**: 40
- **n_splits**: 5
- **n_estimators**: (500, 2000)
- **max_depth**: (3, 9)
- **learning_rate**: (0.01, 0.1)
- **GPU**: CUDA acceleration

#### LightGBM
- **n_trials**: 40
- **n_splits**: 5
- **n_estimators**: (500, 2000)
- **num_leaves**: (20, 150)
- **max_depth**: (3, 12)
- **GPU**: GPU acceleration

#### CatBoost
- **n_trials**: 30
- **n_splits**: 5
- **iterations**: (800, 2500)
- **depth**: (4, 10)
- **learning_rate**: (0.01, 0.1)
- **GPU**: GPU acceleration

---

## Error Logs & Fixes

### Unicode Encoding Errors (Windows)
**Issue**: Emoji characters in print statements caused `UnicodeEncodeError` on Windows.

**Fix**: Replaced all emoji characters with plain text equivalents:
- `✅` → `[SUCCESS]`
- `❌` → `[FAILED]`
- `⚠️` → `[WARNING]`
- `📉` → `[IMPROVED]`
- `📈` → `[WORSE]`

**Files Affected**: `scripts/run_all_models_parallel.py`, `utils/checks.py`, `utils/metrics.py`, `scripts/submit_to_kaggle.py`

### Model Number Extraction Bug
**Issue**: `run_all_models_parallel.py` incorrectly included models 10 and 11 when filtering for models 0-4 (only checked first digit).

**Fix**: Updated `categorize_models` to use regex to extract full numerical prefix:
```python
import re
match = re.match(r'^(\d+)', script_name)
if match:
    model_num = int(match.group(1))
```

### Module Import Errors
**Issue**: `ModuleNotFoundError: No module named 'config_local'` when running scripts.

**Fix**: 
- Set `PYTHONPATH` environment variable to include project root
- Updated all scripts to add project root to `sys.path`

### Data Integrity Error (Model 1)
**Issue**: Linear Regression Updated model produced non-positive `SalePrice` values in submission.

**Status**: Identified but not yet fixed - requires investigation of model implementation.

### Kaggle Submission Limit
**Issue**: Daily submission limit (10/day) reached, preventing additional submissions.

**Solution**: Implemented duplicate submission detection using file hashing to prevent wasting submissions.

---

## Configuration Details

### Model Configuration (`config_local/model_config.py`)

Centralized hyperparameter settings for all models:
- Optuna optimization settings (n_trials, n_splits)
- Search spaces for each model
- Base parameters
- Runtime targets

### Local Configuration (`config_local/local_config.py`)

Path management for:
- Data directories (raw, interim, processed, submissions)
- Feature definitions
- Model performance logs
- Submission logs

---

## Performance Tracking

### Model Performance Log (`runs/model_performance.csv`)

Tracks for each model run:
- Timestamp
- Model name
- CV RMSE
- Hyperparameters (JSON)
- Feature hash
- Feature count
- Notes
- Runtime (human-readable)
- Kaggle score

### Submission Log (`data/submissions/submission_log.json`)

Tracks all Kaggle submissions:
- Timestamp
- File name
- Message
- Rank
- Score
- Total submissions
- File hash (for duplicate detection)

---

## Technical Decisions

### Why Log Transformation?

1. **Distribution Properties**: Sale prices are right-skewed, log transformation normalizes
2. **Outlier Robustness**: Reduces impact of extreme values
3. **Metric Alignment**: RMSLE uses log space, training in log space aligns objectives
4. **Model Convergence**: Better gradient behavior for optimization

### Why 5-Fold CV?

- Balance between bias and variance
- Sufficient data per fold (80% training, 20% validation)
- Standard practice in ML competitions
- Consistent across all models for fair comparison

### Why Optuna over GridSearch?

- **Efficiency**: Bayesian optimization finds good solutions faster
- **Adaptive**: TPE sampler focuses on promising regions
- **Scalable**: Handles large search spaces better
- **Runtime Control**: Can set time limits and trial limits

### Why GPU Acceleration?

- **Speed**: 10-50x faster training for tree-based models
- **Deeper Search**: Enables more thorough hyperparameter optimization
- **Practical**: Makes 20-minute optimization targets feasible

### Why File Hashing for Duplicate Detection?

- **Accuracy**: Detects if file content changed, not just filename
- **Efficiency**: Prevents wasting daily Kaggle submission limit
- **Reliability**: MD5 hash is fast and reliable

---

## Automation Features

### Parallel Model Execution
- Uses `ProcessPoolExecutor` for concurrent training
- Automatic error handling and progress tracking
- Selective execution by model category

### Automatic Score Retrieval
- Fetches Kaggle scores after submission
- Logs to performance tracking system
- Retry logic with exponential backoff

### Duplicate Submission Prevention
- File hash tracking (MD5)
- Checks submission history before submitting
- Prevents wasting daily submission quota

### Virtual Environment Management
- Automatic venv detection
- No manual activation required
- Setup scripts for one-time configuration

---

## Code Quality Improvements

### Type Hints
- All functions have type hints
- Improved IDE support and code clarity

### Docstrings
- Comprehensive function documentation
- Parameter and return type descriptions

### Error Handling
- Proper exception handling throughout
- Informative error messages
- Graceful degradation

### Code Organization
- Modular structure
- Separation of concerns
- Reusable utility functions

---

*For project overview and showcase, see `FLAGSHIP_LOG.md`*

