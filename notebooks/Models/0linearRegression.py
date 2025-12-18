#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from config_local import local_config
from utils.data import load_sample_submission


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

    print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")

    y = train['logSP']
    X = train.drop(columns=['logSP'])
    X_test = test

    print(f"Features: {X.shape[1]} features")
    print(f"Target: logSP (log1p-transformed SalePrice)")

    assert not X.isna().any().any(), "Train data has missing values"
    assert not X_test.isna().any().any(), "Test data has missing values"
    print("Data validation passed - no missing values")

    model = LinearRegression()
    model.fit(X, y)

    pred_train_log = model.predict(X)
    rmse_log = mean_squared_error(y, pred_train_log)

    pred_train_real = np.expm1(pred_train_log)
    y_real = np.expm1(y)
    rmse_real = np.sqrt(mean_squared_error(y_real, pred_train_real))

    print("Intercept:", model.intercept_)
    print(f"Number of features: {len(model.coef_)}")
    print(f"Train RMSE (log scale): {rmse_log:.6f}")
    print(f"Train RMSE (real scale): {rmse_real:,.0f}")

    pred_test_log = model.predict(X_test)
    pred_test_real = np.expm1(pred_test_log)

    cfg = model_config.LINEAR_REGRESSION
    submission = load_sample_submission()
    submission["SalePrice"] = pred_test_real

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)

    print(f"Submission saved -> {out_path}")
