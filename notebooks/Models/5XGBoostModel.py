#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from config_local import local_config
from config_local import model_config
from utils.optimization import run_optuna_study


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)
    testRaw = pd.read_csv(local_config.TEST_CSV, index_col="Id")

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    cfg = model_config.XGBOOST
    
    # Use Optuna for hyperparameter optimization
    print("Running Optuna optimization for XGBoost...")
    best_params = run_optuna_study(
        X.values, 
        y.values, 
        model_type="xgboost",
        base_params=cfg["base_params"],
        optuna_space=cfg["optuna_space"],
        n_trials=50  # Adjust based on time/resources
    )

    # Train final model with best params
    print("Training final model with best hyperparameters...")
    final_params = cfg["base_params"].copy()
    final_params.update(best_params)
    
    best_model = XGBRegressor(**final_params)
    best_model.fit(X.values, y.values)

    # Predict on Kaggle test set
    test_pred_log = best_model.predict(test.values)
    test_pred_real = np.expm1(test_pred_log)

    submission = pd.DataFrame({
        "Id": testRaw.index,
        "SalePrice": test_pred_real
    })

    out_path = os.path.join(local_config.SUBMISSIONS_DIR, "xgboost_Model.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
