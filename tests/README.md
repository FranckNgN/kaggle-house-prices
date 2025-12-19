# Comprehensive Sanity Check Framework

This directory contains a comprehensive test framework that automatically validates each preprocessing stage and model for data integrity and leakage detection.

## Overview

The framework provides:
- **Automatic validation** after each preprocessing file runs
- **Immediate stopping** when errors are detected
- **Clear error highlighting** with colored output
- **Comprehensive checks** for data leakage, integrity, and correctness

## How It Works

### During Preprocessing

When you run `notebooks/preprocessing/run_preprocessing.py`, the system:

1. **Runs each preprocessing file** (1cleaning.py, 2dataEngineering.py, etc.)
2. **Immediately validates** the output after each file completes
3. **Stops the pipeline** if any critical error is detected
4. **Highlights problems** with clear error messages

### Validation Checks Performed

For each preprocessing stage, the system checks:

1. **Train/Test Leakage**: No shared IDs, column parity
2. **Missing Values**: Properly handled in train and test
3. **Shape Consistency**: Row counts match expectations
4. **Column Parity**: Train and test have same features (except target)
5. **Target Integrity**: Target column is valid and correctly transformed
6. **Feature Engineering Sanity**: Engineered features are logically correct
7. **Target Leakage**: No features created using target information

### Error Types

- **DataLeakageError**: Critical - data leakage detected (pipeline stops)
- **DataIntegrityError**: Critical - data integrity issue (pipeline stops)
- **Warnings**: Non-critical issues (pipeline continues but warns)

## Running Tests Manually

You can run tests manually using pytest:

```bash
# Run all preprocessing tests
pytest tests/test_preprocessing/ -v

# Run tests for a specific stage
pytest tests/test_preprocessing/test_stage1_cleaning.py -v

# Run all model tests
pytest tests/test_models/ -v

# Run all tests
pytest tests/ -v
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_preprocessing/
│   ├── test_stage1_cleaning.py
│   ├── test_stage2_data_engineering.py
│   ├── test_stage3_skew_kurtosis.py
│   ├── test_stage4_feature_engineering.py
│   ├── test_stage5_scaling.py
│   └── test_stage6_categorical_encoding.py
└── test_models/
    ├── test_model_base.py         # Basic model validation
    └── test_model_leakage.py     # Leakage detection
```

## Example Output

When an error is detected:

```
============================================================
VALIDATING Stage 2
============================================================

[1/7] Checking for train/test leakage...
✅ No train/test leakage detected

[2/7] Checking missing values...
✅ Missing values properly handled

[3/7] Checking shape consistency...
✅ Shapes consistent (train: 1458, test: 1459)

[4/7] Checking column parity...
✅ Column parity OK (78 shared features)

[5/7] Checking target integrity...
✅ Target 'logSP' is valid

[6/7] Checking feature engineering sanity...
✅ Feature engineering checks passed

[7/7] Checking for target leakage in features...

============================================================
ERROR: TARGET LEAKAGE: Feature 'SalePrice_ratio' has near-perfect correlation with target (r=0.9956). Possible leakage!
============================================================

============================================================
VALIDATION FAILED: 1 error(s) found
  1. TARGET LEAKAGE: Feature 'SalePrice_ratio' has near-perfect correlation with target (r=0.9956). Possible leakage!
============================================================

============================================================
CRITICAL ERROR: Data leakage issue in Stage 2
============================================================

============================================================
PIPELINE STOPPED
============================================================

⚠️  Fix the issue in 2dataEngineering.py before continuing.
```

## Key Functions

### `validate_preprocessing_stage(stage_num, stop_on_error=True)`

Comprehensive validation for a preprocessing stage. Raises exceptions if `stop_on_error=True`.

### `check_no_train_test_leakage(train_df, test_df, stage_name)`

Checks for leakage between train and test sets.

### `check_model_no_target_leakage(X_train, y_train, X_test, model_name)`

Checks that model features don't contain target information.

### `check_cv_properly_implemented(cv_splits, n_samples)`

Validates that cross-validation is properly implemented.

### `check_predictions_sanity(predictions, model_name, target_is_log=True)`

Validates that model predictions are reasonable.

## Integration

The framework is automatically integrated into:
- `notebooks/preprocessing/run_preprocessing.py` - Runs checks after each preprocessing stage
- All preprocessing scripts can be validated independently
- Model scripts can use the validation functions before training

## Requirements

- `pytest>=7.0.0` - Testing framework
- `colorama>=0.4.6` - Colored terminal output (optional, falls back gracefully)

## Notes

- Tests marked with `@pytest.mark.integration` require actual data files
- Tests will skip if `config_local` is not available
- All checks are designed to catch issues early and prevent bad data from propagating

