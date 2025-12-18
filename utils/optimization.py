"""Hyperparameter optimization utilities using Optuna."""
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from typing import Dict, Any, Callable


def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    base_params: Dict[str, Any],
    optuna_space: Dict[str, Any],
    n_trials: int = 100,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run an Optuna study to find best hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: 'xgboost', 'lightgbm', or 'catboost'
        base_params: Fixed parameters for the model
        optuna_space: Search space for Optuna
        n_trials: Number of trials
        n_splits: Number of CV folds
        random_state: Random state for reproducibility
        
    Returns:
        Best hyperparameters found
    """
    
    def objective(trial):
        params = base_params.copy()
        
        # Suggest parameters based on the search space
        for param_name, space in optuna_space.items():
            if isinstance(space, tuple):
                low, high = space[0], space[1]
                log = len(space) > 2 and space[2] == "log"
                
                if isinstance(low, int) and isinstance(high, int) and not log:
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high, log=log)
        
        # Initialize model
        if model_type == "xgboost":
            model = XGBRegressor(**params)
        elif model_type == "lightgbm":
            model = LGBMRegressor(**params)
        elif model_type == "catboost":
            # For CatBoost, we might need special handling for GPU/CPU
            model = CatBoostRegressor(**params)
        elif model_type == "random_forest":
            model = RandomForestRegressor(**params)
        elif model_type == "svr":
            model = SVR(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # 5-Fold Cross Validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        
        return -np.mean(scores)

    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n{'='*60}")
    print(f"Best hyperparameters for {model_type}:")
    print(study.best_params)
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print(f"{'='*60}\n")
    
    return study.best_params, study.best_value

