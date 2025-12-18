#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)

    cfg = model_config.ELASTIC_NET
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"]
    )

    alphas = cfg["alphas"]
    l1_ratios = cfg["l1_ratios"]
    max_iter = cfg["max_iter"]

    best_alpha = None
    best_l1_ratio = None
    best_rmse = float("inf")

    print("Searching best hyperparameters for ElasticNet...\n")

    for a in alphas:
        for l1 in l1_ratios:
            model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=max_iter)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_real = np.exp(y_pred)
            y_test_real = np.exp(y_test)

            mse = mean_squared_error(y_test_real, y_pred_real)
            rmse = mse ** 0.5

            print(f"alpha={a:.6f}, l1_ratio={l1:.1f} -> RMSE={rmse:.2f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = a
                best_l1_ratio = l1

    print("\n===============================")
    print(f"Best ElasticNet -> alpha={best_alpha}, l1_ratio={best_l1_ratio}, RMSE={best_rmse:.2f}")
    print("===============================\n")

    best_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=max_iter)
    best_model.fit(X_train, y_train)

    test_pred_log = best_model.predict(test)
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    out_path = os.path.join(local_config.SUBMISSIONS_DIR, "elasticNetModel.csv")
    submission.to_csv(out_path, index=False)
