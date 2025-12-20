#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission
from utils.metrics import log_model_result
from utils.model_wrapper import (
    validate_predictions_wrapper,
    validate_submission_wrapper
)


def rmse_real(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    # Clip to avoid extreme values
    y_pred = np.clip(y_pred, 0, 1e7)
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)

    cfg = model_config.LASSO
    rmse_scorer = make_scorer(rmse_real, greater_is_better=False)
    param_grid = {"alpha": cfg["alphas"]}

    cv = KFold(
        n_splits=cfg["cv_n_splits"],
        shuffle=cfg["cv_shuffle"],
        random_state=cfg["cv_random_state"]
    )
    lasso = Lasso(max_iter=cfg["max_iter"])

    grid = GridSearchCV(
        estimator=lasso,
        param_grid=param_grid,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=cfg["n_jobs"],
        refit=cfg["refit"]
    )

    # Start timing
    start_time = time.time()
    
    grid.fit(X, y)
    
    # Calculate runtime
    runtime = time.time() - start_time

    best_alpha = grid.best_params_["alpha"]
    best_rmse_cv = -grid.best_score_

    print(f"Best alpha from CV (Lasso): {best_alpha}")
    print(f"CV RMSE (real scale) with best alpha (Lasso): {best_rmse_cv:.4f}")

    # Log results
    log_model_result(
        model_name="lasso",
        rmse=best_rmse_cv,
        hyperparams={"alpha": best_alpha},
        features=X.columns.tolist(),
        notes="GridSearchCV optimized",
        runtime=runtime
    )

    best_model = grid.best_estimator_
    test_pred_log = best_model.predict(test)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "Lasso", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "Lasso", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
