#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from config_local import local_config
from config_local import model_config
from utils.optimization import run_optuna_study
from utils.data import load_sample_submission
from utils.metrics import log_model_result
from utils.model_wrapper import (
    validate_predictions_wrapper,
    validate_submission_wrapper
)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)

    y = train["logSP"]
    X = train.drop(columns=["logSP"])
    
    # Drop categorical columns - use target-encoded versions instead (more informative)
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if categorical_cols:
        print(f"Dropping {len(categorical_cols)} categorical columns (target-encoded versions available): {categorical_cols}")
        X = X.select_dtypes(include=['number'])
        test = test.select_dtypes(include=['number'])

    cfg = model_config.CATBOOST
    opt_cfg = cfg["optuna_settings"]
    
    # Start timing
    start_time = time.time()
    
    # Use Optuna for hyperparameter optimization
    print("\n" + "="*70)
    print("CATBOOST MODEL TRAINING")
    print("="*70)
    print(f"Training samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Optuna trials: {opt_cfg['n_trials']}")
    print(f"CV folds: {opt_cfg['n_splits']}")
    print("="*70)
    
    best_params, best_rmse = run_optuna_study(
        X.values, 
        y.values, 
        model_type="catboost",
        base_params=cfg["base_params"],
        optuna_space=cfg["optuna_space"],
        n_trials=opt_cfg["n_trials"],
        n_splits=opt_cfg["n_splits"],
        random_state=opt_cfg["random_state"],
        show_progress=True
    )
    
    # Calculate runtime
    runtime = time.time() - start_time

    # Log results
    log_model_result(
        model_name="catboost",
        rmse=best_rmse,
        hyperparams={**cfg["base_params"], **best_params},
        features=X.columns.tolist(),
        notes="Optuna Optimized",
        runtime=runtime
    )

    # Train final model with best params
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*70)
    print("Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best CV RMSE: {best_rmse:.6f}")
    print("="*70)
    
    final_params = cfg["base_params"].copy()
    final_params.update(best_params)
    
    # Apply final training verbosity if set
    if "final_fit_verbose" in cfg:
        final_params["verbose"] = cfg["final_fit_verbose"]
    else:
        # Default to showing progress every 100 iterations
        final_params["verbose"] = 100
    
    print(f"\nTraining final model on full dataset...")
    print(f"  Iterations: {final_params.get('iterations', 'auto')}")
    print(f"  Learning rate: {final_params.get('learning_rate', 'auto')}")
    print(f"  Depth: {final_params.get('depth', 'auto')}")
    print(f"  Device: {final_params.get('task_type', 'CPU')}")
    
    final_start_time = time.time()
    best_model = CatBoostRegressor(**final_params)
    
    # Use fit_params from config but remove use_best_model if no eval_set is provided
    fit_params = cfg.get("fit_params", {}).copy()
    if "use_best_model" in fit_params:
        # use_best_model=True requires eval_set; disable for final fit on full data
        fit_params["use_best_model"] = False
        
    best_model.fit(X.values, y.values, **fit_params)
    
    final_time = time.time() - final_start_time
    print(f"\nFinal model training complete! ({final_time/60:.1f} minutes)")

    # Predict on Kaggle test set
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    print(f"Predicting on {len(test)} test samples...")
    
    test_pred_log = best_model.predict(test.values)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "CatBoost", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)
    print(f"  Predictions range: ${test_pred_real.min():,.0f} - ${test_pred_real.max():,.0f}")
    print(f"  Mean prediction: ${test_pred_real.mean():,.0f}")

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "CatBoost", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    
    total_runtime = time.time() - start_time
    
    print("\n" + "="*70)
    print("CATBOOST MODEL COMPLETE!")
    print("="*70)
    print(f"Total runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)")
    print(f"Submission saved: {out_path}")
    print(f"Best CV RMSE: {best_rmse:.6f}")
    print("="*70)
