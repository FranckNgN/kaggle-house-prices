#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from config_local import local_config
from config_local import model_config


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)
    testRaw = pd.read_csv(local_config.TEST_CSV, index_col="Id")

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    cfg = model_config.LIGHTGBM
    lgbm = LGBMRegressor(**cfg["base_params"])

    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=cfg["param_dist"],
        **cfg["search"]
    )

    print("Running RandomizedSearchCV for LightGBM...")
    random_search.fit(X, y)

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

    out_path = os.path.join(local_config.SUBMISSIONS_DIR, "lightGBM_Model.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
