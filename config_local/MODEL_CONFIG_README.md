# Model Configuration Guide

## Overview

The `model_config.py` file is a centralized configuration system that allows you to adjust all model hyperparameters from a single location. This eliminates the need to modify individual model files when tuning parameters.

## Location

`config_local/model_config.py`

## Usage

### Viewing All Configurations

To see all available model configurations:

```python
from config_local import model_config
model_config.print_all_configs()
```

Or run directly:
```bash
python config_local/model_config.py
```

### Accessing Specific Model Config

```python
from config_local import model_config

# Get Ridge configuration
ridge_cfg = model_config.get_model_config('ridge')
print(ridge_cfg['alphas'])

# Or access directly
alphas = model_config.RIDGE['alphas']
```

## Model Configurations

### 1. Linear Regression (`LINEAR_REGRESSION`)
- `kfold_n_splits`: Number of CV folds (default: 5)
- `kfold_shuffle`: Whether to shuffle data (default: True)
- `kfold_random_state`: Random seed (default: 42)

### 2. Ridge Regression (`RIDGE`)
- `alphas`: List of alpha values to search (default: [0.01, 0.1, 1-30])
- `cv_n_splits`: Number of CV folds (default: 5)
- `cv_shuffle`: Whether to shuffle (default: True)
- `cv_random_state`: Random seed (default: 42)
- `n_jobs`: Number of parallel jobs (default: -1)
- `refit`: Whether to refit on full data (default: True)

### 3. Lasso Regression (`LASSO`)
- `alphas`: List of alpha values to search (default: [0.01, 0.1, 1-30])
- `max_iter`: Maximum iterations (default: 10000)
- `cv_n_splits`: Number of CV folds (default: 5)
- `cv_shuffle`: Whether to shuffle (default: True)
- `cv_random_state`: Random seed (default: 42)
- `n_jobs`: Number of parallel jobs (default: -1)
- `refit`: Whether to refit on full data (default: True)

### 4. Elastic Net (`ELASTIC_NET`)
- `alphas`: List of alpha values (default: [0.0001, 0.0005, ..., 0.01])
- `l1_ratios`: List of L1 ratio values (default: [0.1, 0.3, 0.5, 0.7, 0.9])
- `max_iter`: Maximum iterations (default: 20000)
- `test_size`: Train/test split size (default: 0.2)
- `random_state`: Random seed (default: 42)

### 5. XGBoost (`XGBOOST`)
- `base_params`: Base model parameters
  - `objective`: "reg:squarederror"
  - `random_state`: 42
  - `n_jobs`: -1
  - `eval_metric`: "rmse"
  - `tree_method`: "hist" (options: "hist", "exact", "approx")
  - `device`: "cuda" (options: "cuda", "cpu")
- `param_dist`: Hyperparameter search space
  - `n_estimators`: [800, 1000, 1200]
  - `learning_rate`: [0.05, 0.04, 0.03]
  - `max_depth`: [3, 4, 5]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.7, 0.9]
- `search`: RandomizedSearchCV parameters
  - `n_iter`: 30
  - `scoring`: "neg_mean_squared_error"
  - `cv`: 5
  - `n_jobs`: -1
  - `verbose`: 2
  - `random_state`: 42

### 6. LightGBM (`LIGHTGBM`)
- `base_params`: Base model parameters
  - `objective`: "regression"
  - `random_state`: 42
  - `n_jobs`: -1
  - `device_type`: "gpu" (options: "gpu", "cpu")
- `param_dist`: Hyperparameter search space
  - `num_leaves`: [31, 63]
  - `max_depth`: [5, 7, -1] (-1 means no limit)
  - `learning_rate`: [0.05, 0.04, 0.03]
  - `n_estimators`: [800, 1000, 1200]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.7, 0.9]
  - `min_child_samples`: [10, 20]
- `search`: RandomizedSearchCV parameters (same as XGBoost)

### 7. CatBoost (`CATBOOST`)
- `base_params`: Base model parameters
  - `loss_function`: "RMSE"
  - `learning_rate`: 0.03
  - `depth`: 6
  - `iterations`: 2000
  - `random_seed`: 42
  - `verbose`: 0
  - `thread_count`: -1
  - `task_type`: "GPU" (options: "GPU", "CPU")
  - `devices`: "0"
- `search_space`: Parameters to search over
  - `l2_leaf_reg`: [3.0, 5.0]
  - (Add more parameters here to expand search)
- `cv`: Cross-validation settings
  - `n_splits`: 5
  - `shuffle`: True
  - `random_state`: 42
- `fit_params`: Training parameters
  - `use_best_model`: True
  - `verbose`: 500
- `final_fit_verbose`: Verbosity for final model (default: 200)

## Examples

### Example 1: Change Ridge Alpha Range

```python
# In model_config.py, modify:
RIDGE = {
    "alphas": [0.1, 1, 10, 100, 1000],  # Custom range
    # ... rest of config
}
```

### Example 2: Switch XGBoost to CPU

```python
# In model_config.py, modify:
XGBOOST = {
    "base_params": {
        # ...
        "device": "cpu",  # Changed from "cuda"
    },
    # ... rest of config
}
```

### Example 3: Expand CatBoost Search Space

```python
# In model_config.py, modify:
CATBOOST = {
    # ...
    "search_space": {
        "l2_leaf_reg": [3.0, 5.0, 7.0],
        "learning_rate": [0.01, 0.03, 0.05],  # Added
        "depth": [4, 6, 8],  # Added
    },
    # ... rest of config
}
```

### Example 4: Increase CV Folds

```python
# In model_config.py, modify:
RIDGE = {
    # ...
    "cv_n_splits": 10,  # Changed from 5
    # ... rest of config
}
```

## Benefits

1. **Single Source of Truth**: All hyperparameters in one place
2. **Easy Experimentation**: Change parameters without touching model code
3. **Version Control**: Track parameter changes in git
4. **Documentation**: Self-documenting configuration
5. **Reproducibility**: Share exact configurations easily

## Notes

- All models automatically use these configurations
- Changes take effect immediately (no need to modify model files)
- Use `get_model_config()` helper function for programmatic access
- Run `python config_local/model_config.py` to view all configs

