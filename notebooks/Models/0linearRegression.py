#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission
from utils.model_wrapper import (
    validate_predictions_wrapper,
    validate_submission_wrapper
)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)

    print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")

    y = train['logSP']
    X = train.drop(columns=['logSP'])
    X_test = test

    # Drop categorical columns for linear models (tree models can handle them)
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if categorical_cols:
        print(f"Dropping {len(categorical_cols)} categorical columns: {categorical_cols}")
        X = X.select_dtypes(include=['number'])
        X_test = X_test.select_dtypes(include=['number'])

    print(f"Features: {X.shape[1]} features")
    print(f"Target: logSP (log1p-transformed SalePrice)")

    assert not X.isna().any().any(), "Train data has missing values"
    assert not X_test.isna().any().any(), "Test data has missing values"
    print("Data validation passed - no missing values")

    cfg = model_config.LINEAR_REGRESSION
    # Use alpha from config
    alpha = cfg.get("alpha", 100.0)
    model = Ridge(alpha=alpha)
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
    
    # Validate predictions
    validate_predictions_wrapper(pred_test_log, "LinearRegression", target_is_log=True)
    
    pred_test_real = np.expm1(pred_test_log)

    cfg = model_config.LINEAR_REGRESSION
    submission = load_sample_submission()
    submission["SalePrice"] = pred_test_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(X_test), "LinearRegression", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)

    print(f"Submission saved -> {out_path}")
