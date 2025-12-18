#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission


def rmse_real(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)

    cfg = model_config.RIDGE
    rmse_scorer = make_scorer(rmse_real, greater_is_better=False)
    param_grid = {"alpha": cfg["alphas"]}

    cv = KFold(
        n_splits=cfg["cv_n_splits"],
        shuffle=cfg["cv_shuffle"],
        random_state=cfg["cv_random_state"]
    )
    ridge = Ridge()

    grid = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=cfg["n_jobs"],
        refit=cfg["refit"]
    )

    grid.fit(X, y)

    best_alpha = grid.best_params_["alpha"]
    best_rmse_cv = -grid.best_score_

    print(f"Best alpha from CV: {best_alpha}")
    print(f"CV RMSE (real scale) with best alpha: {best_rmse_cv:.4f}")

    best_model = grid.best_estimator_
    test_pred_log = best_model.predict(test)
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
