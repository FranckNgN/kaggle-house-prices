#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from config_local import local_config


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)
    testRaw = pd.read_csv(local_config.TEST_CSV, index_col="Id")

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        tree_method="hist",
        device="cuda"
    )

    param_dist = {
        "n_estimators": [800, 1000, 1200],
        "learning_rate": [0.05, 0.04, 0.03],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    print("Running RandomizedSearchCV for XGBoost...")
    random_search.fit(X.values, y.values)

    print("\nBest params found:")
    print(random_search.best_params_)

    best_mse = -random_search.best_score_
    best_rmse = best_mse ** 0.5
    print(f"\nBest CV RMSE from RandomizedSearch: {best_rmse:.4f}")

    best_model = random_search.best_estimator_
    test_pred_log = best_model.predict(test)
    test_pred_real = np.expm1(test_pred_log)

    submission = pd.DataFrame({
        "Id": testRaw.index,
        "SalePrice": test_pred_real
    })

    out_path = os.path.join(local_config.SUBMISSIONS_DIR, "xgboost_Model.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
