#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from config_local import local_config
from config_local import model_config
from utils.optimization import run_optuna_study
from utils.data import load_sample_submission


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    cfg = model_config.LIGHTGBM
    opt_cfg = cfg["optuna_settings"]
    
    # Use Optuna for hyperparameter optimization
    print("Running Optuna optimization for LightGBM...")
    best_params = run_optuna_study(
        X.values, 
        y.values, 
        model_type="lightgbm",
        base_params=cfg["base_params"],
        optuna_space=cfg["optuna_space"],
        n_trials=opt_cfg["n_trials"],
        n_splits=opt_cfg["n_splits"],
        random_state=opt_cfg["random_state"]
    )

    # Train final model with best params
    print("Training final model with best hyperparameters...")
    final_params = cfg["base_params"].copy()
    final_params.update(best_params)
    
    best_model = LGBMRegressor(**final_params)
    best_model.fit(X.values, y.values)

    # Predict on Kaggle test set
    test_pred_log = best_model.predict(test.values)
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    out_path = os.path.join(local_config.SUBMISSIONS_DIR, "lightGBM_Model.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
