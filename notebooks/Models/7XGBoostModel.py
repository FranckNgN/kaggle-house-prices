#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from config_local import local_config
from config_local import model_config
from kaggle.remote.gpu_runner import get_gpu_params_for_model
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

    cfg = model_config.XGBOOST
    opt_cfg = cfg["optuna_settings"]
    
    # Detect GPU and adjust parameters
    gpu_params = get_gpu_params_for_model("xgboost")
    print("\n" + "="*70)
    print("XGBOOST MODEL TRAINING")
    print("="*70)
    if gpu_params.get("tree_method") == "gpu_hist":
        print("[INFO] GPU detected - will use GPU acceleration")
    else:
        print("[INFO] GPU not available - will use CPU")
    print("="*70)
    
    # Update base_params with GPU settings
    base_params = cfg["base_params"].copy()
    base_params.update(gpu_params)
    
    # Start timing
    start_time = time.time()
    
    # Use Optuna for hyperparameter optimization
    print("\nRunning Optuna optimization for XGBoost...")
    best_params, best_rmse = run_optuna_study(
        X.values, 
        y.values, 
        model_type="xgboost",
        base_params=base_params,
        optuna_space=cfg["optuna_space"],
        n_trials=opt_cfg["n_trials"],
        n_splits=opt_cfg["n_splits"],
        random_state=opt_cfg["random_state"]
    )
    
    # Calculate runtime
    runtime = time.time() - start_time

    # Log results
    log_model_result(
        model_name="xgboost",
        rmse=best_rmse,
        hyperparams={**cfg["base_params"], **best_params},
        features=X.columns.tolist(),
        notes="Optuna Optimized",
        runtime=runtime
    )

    # Train final model with best params
    print("\nTraining final model with best hyperparameters...")
    final_params = base_params.copy()
    final_params.update(best_params)
    print(f"  Device: {final_params.get('tree_method', 'hist')} ({'GPU' if final_params.get('tree_method') == 'gpu_hist' else 'CPU'})")
    
    best_model = XGBRegressor(**final_params)
    best_model.fit(X.values, y.values)

    # Predict on Kaggle test set
    test_pred_log = best_model.predict(test.values)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "XGBoost", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)

    # Create submission using enhanced utility
    from utils.models import create_submission
    submission = create_submission(
        predictions=test_pred_log,
        test_ids=load_sample_submission()["Id"],
        filename=cfg["submission_filename"],
        log_space=True,
        model_name=cfg["submission_name"],
        validate=True
    )

    # Additional validation for compatibility
    validate_submission_wrapper(submission, len(test), "XGBoost", test_ids=submission["Id"])
    
    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    print(f"Submission saved: {out_path}")
