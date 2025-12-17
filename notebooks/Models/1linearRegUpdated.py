#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from config_local import local_config


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)
    testRaw = pd.read_csv(local_config.TEST_CSV, index_col="Id")

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_val_pred_log = model.predict(X_val)
        y_val_pred = np.exp(y_val_pred_log)
        y_val_real = np.exp(y_val)

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
    test_pred = np.expm1(test_pred_log)

    submission = pd.DataFrame({
        "Id": testRaw.index,
        "SalePrice": test_pred
    })

    out_path = os.path.join(local_config.SUBMISSIONS_DIR, "linearModel_KFold.csv")
    submission.to_csv(out_path, index=False)
    print("Saved:", out_path)
