#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from config_local import local_config
from config_local import model_config


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)
    testRaw = pd.read_csv(local_config.TEST_CSV, index_col="Id")

    y = train["logSP"]
    X = train.drop(columns=["logSP"])

    cfg = model_config.CATBOOST
    kf = KFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"]
    )

    best_params = None
    best_cv_rmse = float("inf")

    print("Searching best CatBoost hyperparameters with 5-Fold CV...\n")

    # Generate all parameter combinations from search space
    base_params = cfg["base_params"].copy()
    search_space = cfg["search_space"]
    
    # Generate all combinations using itertools.product
    param_names = list(search_space.keys())
    param_value_lists = list(search_space.values())
    
    for param_combination in product(*param_value_lists):
        params = base_params.copy()
        for param_name, param_value in zip(param_names, param_combination):
            params[param_name] = param_value

        fold_rmses = []
        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k in ["learning_rate", "depth", "l2_leaf_reg", "iterations"]])
        print(f"Testing params: {param_str}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            print(f"    Starting fold {fold}...")
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                **cfg["fit_params"]
            )

            y_pred_log = model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred_log) ** 0.5
            fold_rmses.append(rmse)
            print(f"  Fold {fold} log-RMSE: {rmse:.5f}")

        mean_rmse = np.mean(fold_rmses)
        print(f"--> Mean CV log-RMSE for these params: {mean_rmse:.5f}\n")

        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_params = params.copy()

    print("\n=====================================")
    print("Best CatBoost params found:")
    print(best_params)
    print(f"Best mean CV log-RMSE: {best_cv_rmse:.5f}")
    print("=====================================\n")

    final_cat = CatBoostRegressor(**best_params)
    print("Training final CatBoost model on all data with best hyperparameters...")
    final_cat.fit(X, y, verbose=cfg["final_fit_verbose"])

    test_pred_log_cat = final_cat.predict(test)
    test_pred_real_cat = np.expm1(test_pred_log_cat)

    submission_cat = pd.DataFrame({
        "Id": testRaw.index,
        "SalePrice": test_pred_real_cat
    })

    out_path_cat = os.path.join(local_config.SUBMISSIONS_DIR, "catboost_Model.csv")
    submission_cat.to_csv(out_path_cat, index=False)
    print(f"CatBoost submission saved to: {out_path_cat}")
