#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission
from utils.model_wrapper import (
    validate_predictions_wrapper,
    validate_submission_wrapper
)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS7_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS7_CSV)

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)

    cfg = model_config.LINEAR_REGRESSION_UPDATED
    kf = KFold(
        n_splits=cfg["kfold_n_splits"],
        shuffle=cfg["kfold_shuffle"],
        random_state=cfg["kfold_random_state"]
    )
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_val_pred_log = model.predict(X_val)
        
        # Clip log predictions to avoid overflow in exp
        y_val_pred_log = np.clip(y_val_pred_log, 0, 15) 
        
        y_val_pred = np.expm1(y_val_pred_log)
        y_val_real = np.expm1(y_val)

        mse = mean_squared_error(y_val_real, y_val_pred)
        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)

        print(f"Fold {fold}: RMSE = {rmse:.4f}")

    print("\n==== K-Fold CV with LinearRegression ====")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Std  RMSE: {np.std(rmse_scores):.4f}")

    final_model = LinearRegression()
    final_model.fit(X, y)

    test_pred_log = final_model.predict(test)
    test_pred_log = np.clip(test_pred_log, 0, 15)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "LinearRegressionUpdated", target_is_log=True)
    
    test_pred = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "LinearRegressionUpdated", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print("Saved:", out_path)
