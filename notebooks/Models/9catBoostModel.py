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
    train = pd.read_csv(local_config.TRAIN_PROCESS7_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS7_CSV)

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    cfg = model_config.CATBOOST
    opt_cfg = cfg["optuna_settings"]
    
    # Start timing
    start_time = time.time()
    
    # Use Optuna for hyperparameter optimization
    print("Running Optuna optimization for CatBoost...")
    best_params, best_rmse = run_optuna_study(
        X.values, 
        y.values, 
        model_type="catboost",
        base_params=cfg["base_params"],
        optuna_space=cfg["optuna_space"],
        n_trials=opt_cfg["n_trials"],
        n_splits=opt_cfg["n_splits"],
        random_state=opt_cfg["random_state"]
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
    print("Training final model with best hyperparameters...")
    final_params = cfg["base_params"].copy()
    final_params.update(best_params)
    
    # Apply final training verbosity if set
    if "final_fit_verbose" in cfg:
        final_params["verbose"] = cfg["final_fit_verbose"]
    
    best_model = CatBoostRegressor(**final_params)
    
    # Use fit_params from config but remove use_best_model if no eval_set is provided
    fit_params = cfg.get("fit_params", {}).copy()
    if "use_best_model" in fit_params:
        # use_best_model=True requires eval_set; disable for final fit on full data
        fit_params["use_best_model"] = False
        
    best_model.fit(X.values, y.values, **fit_params)

    # Predict on Kaggle test set
    test_pred_log = best_model.predict(test.values)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "CatBoost", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "CatBoost", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
