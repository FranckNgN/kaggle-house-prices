#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission
from utils.metrics import log_model_result, rmse_real
from utils.model_wrapper import (
    validate_predictions_wrapper,
    validate_submission_wrapper
)


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)

    y = train['logSP']
    X = train.drop(['logSP'], axis=1)
    
    # Drop categorical columns (Ridge needs numeric only)
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if categorical_cols:
        print(f"Dropping {len(categorical_cols)} categorical columns: {categorical_cols}")
        X = X.select_dtypes(include=['number'])
        test = test.select_dtypes(include=['number'])

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

    # Start timing
    start_time = time.time()
    
    grid.fit(X, y)
    
    # Calculate runtime
    runtime = time.time() - start_time

    best_alpha = grid.best_params_["alpha"]
    best_rmse_cv_real = -grid.best_score_
    
    # Also calculate log-scale RMSE for comparison with previous results
    best_model_cv = grid.best_estimator_
    y_pred_log = best_model_cv.predict(X)
    best_rmse_cv_log = np.sqrt(mean_squared_error(y, y_pred_log))

    print(f"Best alpha from CV: {best_alpha}")
    print(f"CV RMSE (real scale): {best_rmse_cv_real:.4f}")
    print(f"CV RMSE (log scale): {best_rmse_cv_log:.6f}")

    # Log results
    log_model_result(
        model_name="ridge",
        rmse=best_rmse_cv_log,  # Use log-scale RMSE for consistency
        hyperparams={"alpha": best_alpha},
        features=X.columns.tolist(),
        notes=f"GridSearchCV optimized (process8, {len(X.columns)} features)",
        runtime=runtime
    )

    best_model = grid.best_estimator_
    test_pred_log = best_model.predict(test)
    
    # Validate predictions
    validate_predictions_wrapper(test_pred_log, "Ridge", target_is_log=True)
    
    test_pred_real = np.expm1(test_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = test_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(test), "Ridge", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
