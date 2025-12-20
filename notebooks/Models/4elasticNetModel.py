#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission
from utils.model_wrapper import (
    validate_predictions_wrapper,
    validate_submission_wrapper
)


def rmse_real(y_true_log, y_pred_log):
    # Use expm1 because logSP was created with log1p
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    # Clip to avoid extreme values if model is unstable
    y_pred = np.clip(y_pred, 0, 1e7)
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)

    cfg = model_config.ELASTIC_NET
    
    # Setup CV
    rmse_scorer = make_scorer(rmse_real, greater_is_better=False)
    param_grid = {
        "alpha": cfg["alphas"],
        "l1_ratio": cfg["l1_ratios"]
    }
    
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg["random_state"]
    )
    
    print("Searching best hyperparameters for ElasticNet using GridSearchCV...")
    
    enet = ElasticNet(max_iter=cfg["max_iter"], random_state=cfg["random_state"])
    
    grid = GridSearchCV(
        estimator=enet,
        param_grid=param_grid,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    best_params = grid.best_params_
    best_rmse_cv = -grid.best_score_
    
    print("\n===============================")
    print(f"Best ElasticNet -> alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
    print(f"CV RMSE (real scale): {best_rmse_cv:.2f}")
    print("===============================\n")

    # Final model trained on all data
    best_model = grid.best_estimator_
    
    test_pred_log = best_model.predict(test)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "ElasticNet", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "ElasticNet", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
