"""Hyperparameter optimization utilities using Optuna."""
import optuna
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from typing import Dict, Any, Callable
from utils.cv_strategy import get_cv_strategy


class ProgressCallback:
    """Callback to show Optuna study progress."""
    def __init__(self, n_trials: int, n_splits: int):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.start_time = time.time()
        self.trial_times = []
        
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        trial_start = time.time()
        trial_num = trial.number + 1
        
        # Calculate progress
        progress_pct = (trial_num / self.n_trials) * 100
        elapsed = time.time() - self.start_time
        
        # Estimate remaining time
        if trial_num > 1:
            avg_trial_time = elapsed / trial_num
            remaining_trials = self.n_trials - trial_num
            estimated_remaining = avg_trial_time * remaining_trials
            remaining_str = f"{estimated_remaining/60:.1f} min"
        else:
            remaining_str = "calculating..."
        
        # Get current best
        if len(study.trials) > 0:
            best_value = study.best_value
            best_trial = study.best_trial.number + 1
        else:
            best_value = None
            best_trial = None
        
        print(f"\n{'='*70}")
        print(f"Trial {trial_num}/{self.n_trials} ({progress_pct:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min | Estimated remaining: {remaining_str}")
        if best_value is not None:
            print(f"  Current best: RMSE = {best_value:.6f} (from trial {best_trial})")
        print(f"  Running {self.n_splits}-fold CV...")
        print(f"{'='*70}")


def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    base_params: Dict[str, Any],
    optuna_space: Dict[str, Any],
    n_trials: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
    show_progress: bool = True,
    cv_strategy: str = "stratified"
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
            if isinstance(space, list):
                # Categorical parameter (e.g., loss functions)
                params[param_name] = trial.suggest_categorical(param_name, space)
            elif isinstance(space, tuple):
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
            
        # Cross Validation with specified strategy
        if cv_strategy == "stratified":
            cv_splits = get_cv_strategy(
                strategy="stratified",
                y=y,
                n_splits=n_splits,
                random_state=random_state
            )
            # cross_val_score doesn't work directly with list of splits, so we compute manually
            scores = []
            for train_idx, val_idx in cv_splits:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
                scores.append(-rmse)  # Negative for consistency with cross_val_score
            scores = np.array(scores)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        
        return -np.mean(scores)

    # Create and run study
    study = optuna.create_study(direction="minimize")
    
    # Add progress callback if requested
    callbacks = []
    if show_progress:
        progress_callback = ProgressCallback(n_trials, n_splits)
        callbacks.append(progress_callback)
    
    print(f"\n{'='*70}")
    print(f"Starting Optuna optimization for {model_type}")
    print(f"  Trials: {n_trials}")
    print(f"  CV Folds: {n_splits}")
    print(f"  CV Strategy: {cv_strategy}")
    print(f"  Total model fits: {n_trials * n_splits}")
    print(f"{'='*70}")
    
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
    
    total_time = time.time() - (callbacks[0].start_time if callbacks else time.time())
    
    print(f"\n{'='*70}")
    print(f"Optuna optimization complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best hyperparameters for {model_type}:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"  Best CV RMSE: {study.best_value:.6f}")
    print(f"{'='*70}\n")
    
    return study.best_params, study.best_value

