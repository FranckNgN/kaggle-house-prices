#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
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
    

    cfg = model_config.CATBOOST
    opt_cfg = cfg["optuna_settings"]
    
    # Detect GPU and adjust parameters
    gpu_params = get_gpu_params_for_model("catboost")
    
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
    if gpu_params.get("task_type") == "GPU":
        print(f"Device: GPU (accelerated)")
    else:
        print(f"Device: CPU")
    print("="*70)
    
    # Update base_params with GPU settings
    base_params = cfg["base_params"].copy()
    base_params.update(gpu_params)
    
    best_params, best_rmse = run_optuna_study(
        X.values, 
        y.values, 
        model_type="catboost",
        base_params=base_params,
        optuna_space=cfg["optuna_space"],
        n_trials=opt_cfg["n_trials"],
        n_splits=opt_cfg["n_splits"],
        random_state=opt_cfg["random_state"],
        show_progress=True,
        cv_strategy="stratified"
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
    
    final_params = base_params.copy()
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
    device_type = final_params.get('task_type', 'CPU')
    print(f"  Device: {device_type} {'(GPU accelerated)' if device_type == 'GPU' else '(CPU)'}")
    
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
    validate_submission_wrapper(submission, len(test), "CatBoost", test_ids=submission["Id"])
    
    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    
    total_runtime = time.time() - start_time
    
    print("\n" + "="*70)
    print("CATBOOST MODEL COMPLETE!")
    print("="*70)
    print(f"Total runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)")
    print(f"Submission saved: {out_path}")
    print(f"Best CV RMSE: {best_rmse:.6f}")
    print("="*70)
