"""
Centralized Model Configuration
===============================
This file contains all hyperparameters for all models in the project.
Modify parameters here to adjust model behavior without editing individual model files.
"""

# ============================================================================
# COMMON PARAMETERS
# ============================================================================
RANDOM_STATE = 42
CV_N_SPLITS = 5
CV_SHUFFLE = True
N_JOBS = -1  # Use all available CPUs (-1) or specify number

# Train/Test split for models that use it
TRAIN_TEST_SPLIT_SIZE = 0.2
TRAIN_TEST_SPLIT_RANDOM_STATE = 42

# ============================================================================
# LINEAR REGRESSION
# ============================================================================
LINEAR_REGRESSION = {
    "kfold_n_splits": 5,
    "kfold_shuffle": True,
    "kfold_random_state": 42,
}

# ============================================================================
# RIDGE REGRESSION
# ============================================================================
RIDGE = {
    "alphas": sorted({0.01, 0.1, 30} | set(range(1, 31))),  # 0.01, 0.1, 1..30
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 42,
    "n_jobs": -1,
    "refit": True,
}

# ============================================================================
# LASSO REGRESSION
# ============================================================================
LASSO = {
    "alphas": sorted({0.01, 0.1, 30} | set(range(1, 31))),  # 0.01, 0.1, 1..30
    "max_iter": 10000,
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 42,
    "n_jobs": -1,
    "refit": True,
}

# ============================================================================
# ELASTIC NET
# ============================================================================
ELASTIC_NET = {
    "alphas": [
        0.0001, 0.0005, 0.0007, 0.0008, 0.0009,
        0.0010, 0.0011, 0.0012, 0.0013, 0.0015,
        0.002, 0.005, 0.01
    ],
    "l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
    "max_iter": 20000,
    "test_size": 0.2,
    "random_state": 42,
}

# ============================================================================
# XGBOOST
# ============================================================================
XGBOOST = {
    # Base model parameters
    "base_params": {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "rmse",
        "tree_method": "hist",  # Options: "hist", "exact", "approx"
        "device": "cuda",  # Options: "cuda", "cpu"
    },
    # Hyperparameter search space
    "param_dist": {
        "n_estimators": [800, 1000, 1200],
        "learning_rate": [0.05, 0.04, 0.03],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
    },
    # RandomizedSearchCV parameters
    "search": {
        "n_iter": 30,
        "scoring": "neg_mean_squared_error",
        "cv": 5,
        "n_jobs": -1,
        "verbose": 2,
        "random_state": 42,
    },
}

# ============================================================================
# LIGHTGBM
# ============================================================================
LIGHTGBM = {
    # Base model parameters
    "base_params": {
        "objective": "regression",
        "random_state": 42,
        "n_jobs": -1,
        "device_type": "gpu",  # Options: "gpu", "cpu"
    },
    # Hyperparameter search space
    "param_dist": {
        "num_leaves": [31, 63],
        "max_depth": [5, 7, -1],  # -1 means no limit
        "learning_rate": [0.05, 0.04, 0.03],
        "n_estimators": [800, 1000, 1200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "min_child_samples": [10, 20],
    },
    # RandomizedSearchCV parameters
    "search": {
        "n_iter": 30,
        "scoring": "neg_mean_squared_error",
        "cv": 5,
        "n_jobs": -1,
        "verbose": 2,
        "random_state": 42,
    },
}

# ============================================================================
# CATBOOST
# ============================================================================
CATBOOST = {
    # Base parameters (fixed)
    "base_params": {
        "loss_function": "RMSE",
        "learning_rate": 0.03,
        "depth": 6,
        "iterations": 2000,
        "random_seed": 42,
        "verbose": 0,  # Set >0 for training logs
        "thread_count": -1,
        "task_type": "GPU",  # Options: "GPU", "CPU"
        "devices": "0",  # GPU device ID
    },
    # Hyperparameter search space
    "search_space": {
        "l2_leaf_reg": [3.0, 5.0],
        # Add more parameters here if you want to search over them
        # "learning_rate": [0.01, 0.03, 0.05],
        # "depth": [4, 6, 8],
        # "iterations": [1500, 2000, 2500],
    },
    # Cross-validation settings
    "cv": {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
    # Training settings
    "fit_params": {
        "use_best_model": True,
        "verbose": 500,  # Print progress every N iterations
    },
    "final_fit_verbose": 200,  # Verbosity for final model training
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'ridge', 'xgboost', 'lightgbm')
        
    Returns:
        Dictionary containing model configuration
        
    Example:
        >>> config = get_model_config('ridge')
        >>> print(config['alphas'])
    """
    configs = {
        "linear_regression": LINEAR_REGRESSION,
        "ridge": RIDGE,
        "lasso": LASSO,
        "elastic_net": ELASTIC_NET,
        "xgboost": XGBOOST,
        "lightgbm": LIGHTGBM,
        "catboost": CATBOOST,
    }
    
    model_name_lower = model_name.lower().replace(" ", "_")
    if model_name_lower not in configs:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(configs.keys())}"
        )
    
    return configs[model_name_lower]


def print_all_configs():
    """Print all model configurations for reference."""
    import json
    
    configs = {
        "LINEAR_REGRESSION": LINEAR_REGRESSION,
        "RIDGE": RIDGE,
        "LASSO": LASSO,
        "ELASTIC_NET": ELASTIC_NET,
        "XGBOOST": XGBOOST,
        "LIGHTGBM": LIGHTGBM,
        "CATBOOST": CATBOOST,
    }
    
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    # Print all configurations when run directly
    print_all_configs()

