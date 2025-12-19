#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVR
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
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    cfg = model_config.SVR
    opt_cfg = cfg["optuna_settings"]
    
    # Start timing
    start_time = time.time()
    
    # Use Optuna for hyperparameter optimization
    print("Running Optuna optimization for SVR...")
    best_params, best_rmse = run_optuna_study(
        X.values, 
        y.values, 
        model_type="svr",
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
        model_name="svr",
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
    
    best_model = SVR(**final_params)
    best_model.fit(X.values, y.values)

    # Predict on Kaggle test set
    test_pred_log = best_model.predict(test.values)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "SVR", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "SVR", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")

