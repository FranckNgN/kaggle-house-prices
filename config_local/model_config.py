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
# LINEAR REGRESSION (Level 0)
# ============================================================================
LINEAR_REGRESSION = {
    "submission_name": "0_linear_regression",
    "submission_filename": "naive_lr.csv",
    "kfold_n_splits": 5,
    "kfold_shuffle": True,
    "kfold_random_state": 42,
}

# ============================================================================
# LINEAR REGRESSION UPDATED (Level 1)
# ============================================================================
LINEAR_REGRESSION_UPDATED = {
    "submission_name": "1_linear_regression_updated",
    "submission_filename": "linearModel_KFold.csv",
    "kfold_n_splits": 5,
    "kfold_shuffle": True,
    "kfold_random_state": 42,
}

# ============================================================================
# RIDGE REGRESSION (Level 2)
# ============================================================================
RIDGE = {
    "submission_name": "2_ridge",
    "submission_filename": "ridgeModel_KFold.csv",
    "alphas": sorted({0.01, 0.1, 30} | set(range(1, 31))),  # 0.01, 0.1, 1..30
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 42,
    "n_jobs": -1,
    "refit": True,
}

# ============================================================================
# LASSO REGRESSION (Level 3)
# ============================================================================
LASSO = {
    "submission_name": "3_lasso",
    "submission_filename": "lassoModel_KFold.csv",
    "alphas": sorted({0.01, 0.1, 30} | set(range(1, 31))),  # 0.01, 0.1, 1..30
    "max_iter": 10000,
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 42,
    "n_jobs": -1,
    "refit": True,
}

# ============================================================================
# ELASTIC NET (Level 4)
# ============================================================================
ELASTIC_NET = {
    "submission_name": "4_elastic_net",
    "submission_filename": "elasticNetModel.csv",
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
# RANDOM FOREST (Level 5)
# ============================================================================
RANDOM_FOREST = {
    "submission_name": "5_random_forest",
    "submission_filename": "randomForest_Model.csv",
    "base_params": {
        "random_state": 42,
        "n_jobs": -1,
        "criterion": "mse",
    },
    "optuna_settings": {
        "n_trials": 5,
        "n_splits": 3,  # Reduced from 5 to 3 for faster training
        "random_state": 42,
    },
    "optuna_space": {
        "n_estimators": (50, 300),  # Reduced from (100, 1000) for faster training
        "max_depth": (3, 15),  # Reduced from (3, 20)
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": (0.1, 1.0),
    },
}

# ============================================================================
# SUPPORT VECTOR REGRESSION (SVR) (Level 6)
# ============================================================================
SVR = {
    "submission_name": "6_svr",
    "submission_filename": "svr_Model.csv",
    "base_params": {
        "kernel": "rbf",
    },
    "optuna_settings": {
        "n_trials": 5,
        "n_splits": 3,  # Reduced from 5 to 3 for faster training
        "random_state": 42,
    },
    "optuna_space": {
        "C": (0.1, 100.0, "log"),
        "epsilon": (0.001, 1.0, "log"),
        "gamma": (0.0001, 1.0, "log"),
    },
}

# ============================================================================
# XGBOOST (Level 7)
# ============================================================================
XGBOOST = {
    "submission_name": "7_xgboost",
    "submission_filename": "xgboost_Model.csv",
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
        "n_estimators": [300, 400, 500],  # Reduced from [800, 1000, 1200] for faster training
        "learning_rate": [0.05, 0.04, 0.03],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
    },
    # RandomizedSearchCV parameters
    "search": {
        "n_iter": 10,  # Reduced from 30 to 10 for faster training
        "scoring": "neg_mean_squared_error",
        "cv": 3,  # Reduced from 5 to 3 for faster training
        "n_jobs": -1,
        "verbose": 2,
        "random_state": 42,
    },
    # Optuna search settings
    "optuna_settings": {
        "n_trials": 5,
        "n_splits": 3,  # Reduced from 5 to 3 for faster training
        "random_state": 42,
    },
    # Optuna search space
    "optuna_space": {
        "n_estimators": (200, 600),  # Reduced from (500, 2000) for faster training
        "learning_rate": (0.01, 0.1, "log"),
        "max_depth": (3, 7),  # Reduced from (3, 9)
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "min_child_weight": (1, 10),
        "gamma": (0, 5),
    },
}

# ============================================================================
# LIGHTGBM (Level 8)
# ============================================================================
LIGHTGBM = {
    "submission_name": "8_lightgbm",
    "submission_filename": "lightGBM_Model.csv",
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
        "n_estimators": [300, 400, 500],  # Reduced from [800, 1000, 1200] for faster training
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "min_child_samples": [10, 20],
    },
    # RandomizedSearchCV parameters
    "search": {
        "n_iter": 10,  # Reduced from 30 to 10 for faster training
        "scoring": "neg_mean_squared_error",
        "cv": 3,  # Reduced from 5 to 3 for faster training
        "n_jobs": -1,
        "verbose": 2,
        "random_state": 42,
    },
    # Optuna search settings
    "optuna_settings": {
        "n_trials": 5,
        "n_splits": 3,  # Reduced from 5 to 3 for faster training
        "random_state": 42,
    },
    # Optuna search space
    "optuna_space": {
        "n_estimators": (200, 600),  # Reduced from (500, 2000) for faster training
        "learning_rate": (0.01, 0.1, "log"),
        "num_leaves": (20, 100),  # Reduced from (20, 150)
        "max_depth": (3, 8),  # Reduced from (3, 12)
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "min_child_samples": (5, 50),
        "reg_alpha": (0, 10),
        "reg_lambda": (0, 10),
    },
}

# ============================================================================
# CATBOOST (Level 9)
# ============================================================================
CATBOOST = {
    "submission_name": "9_catboost",
    "submission_filename": "catboost_Model.csv",
    # Base parameters (fixed)
    "base_params": {
        "loss_function": "RMSE",
        "learning_rate": 0.03,
        "depth": 6,
        "iterations": 500,  # Reduced from 2000 to 500 for faster training
        "random_seed": 42,
        "verbose": 0,  # Set >0 for training logs
        "thread_count": -1,
        "task_type": "GPU",  # Options: "GPU", "CPU"
        "devices": "0",  # GPU device ID
    },
    # Hyperparameter search space
    "search_space": {
        "l2_leaf_reg": [3.0, 5.0],
    },
    # Optuna search settings
    "optuna_settings": {
        "n_trials": 5,
        "n_splits": 3,  # Reduced from 5 to 3 for faster training
        "random_state": 42,
    },
    # Optuna search space
    "optuna_space": {
        "iterations": (300, 800),  # Reduced from (1000, 3000) for faster training
        "learning_rate": (0.01, 0.1, "log"),
        "depth": (4, 8),  # Reduced from (4, 10)
        "l2_leaf_reg": (1, 10),
        "bagging_temperature": (0, 1),
        "random_strength": (0, 1),
    },
    # Cross-validation settings
    "cv": {
        "n_splits": 3,  # Reduced from 5 to 3 for faster training
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
# BLENDING MODEL (Level 10)
# ============================================================================
BLENDING = {
    "submission_name": "10_blending",
    "submission_filename": "blend_xgb_lgb_cat_Model.csv",
    # Dictionary mapping model name to its CSV filename
    "models": {
        "xgb": XGBOOST["submission_filename"],
        "lgb": LIGHTGBM["submission_filename"],
        "cat": CATBOOST["submission_filename"],
        "ridge": RIDGE["submission_filename"],
        "lasso": LASSO["submission_filename"],
        "elasticNet": ELASTIC_NET["submission_filename"],
        "rf": RANDOM_FOREST["submission_filename"],
        "svr": SVR["submission_filename"],
    },
    # Weights for the weighted average blend
    "weights": {
        "xgb": 2.0,
        "lgb": 0.5,
        "cat": 1.0,
        "ridge": 0.0,
        "lasso": 0.0,
        "elasticNet": 0.0,
        "rf": 0.0,
        "svr": 0.0,
    },
}

# ============================================================================
# STACKING MODEL (Level 11)
# ============================================================================
STACKING = {
    "submission_name": "11_stacking",
    "submission_filename": "stacking_submission.csv",
    # Base models to use for stacking (must match names in configs)
    "base_models": [
        "xgboost", 
        "lightgbm", 
        "catboost", 
        "ridge", 
        "lasso", 
        "elastic_net",
        "random_forest",
        "svr"
    ],
    # Meta-model to combine predictions
    "meta_model": "lasso", 
    "meta_model_params": {
        "alpha": 0.0005,
        "random_state": 42,
    },
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 42,
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
        "random_forest": RANDOM_FOREST,
        "svr": SVR,
        "stacking": STACKING,
        "blending": BLENDING,
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
        "RANDOM_FOREST": RANDOM_FOREST,
        "SVR": SVR,
        "STACKING": STACKING,
        "BLENDING": BLENDING,
    }
    
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    # Print all configurations when run directly
    print_all_configs()


