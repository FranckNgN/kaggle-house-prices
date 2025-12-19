#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from config_local import local_config
from config_local import model_config
from utils.data import load_sample_submission
from utils.metrics import log_model_result

def get_base_model(model_name, params):
    """Factory for creating base models."""
    if model_name == "ridge":
        return Ridge(**params)
    elif model_name == "lasso":
        return Lasso(**params)
    elif model_name == "elastic_net":
        return ElasticNet(**params)
    elif model_name == "xgboost":
        return XGBRegressor(**params)
    elif model_name == "lightgbm":
        return LGBMRegressor(**params)
    elif model_name == "catboost":
        return CatBoostRegressor(**params)
    elif model_name == "random_forest":
        return RandomForestRegressor(**params)
    elif model_name == "svr":
        return SVR(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

    y = train["logSP"].values
    feature_names = train.drop(columns=["logSP"]).columns.tolist()
    X = train.drop(columns=["logSP"]).values
    X_test = test.values

    cfg = model_config.STACKING
    base_model_names = cfg["base_models"]
    
    # Get modification time of input data for cache validation
    train_mtime = os.path.getmtime(local_config.TRAIN_PROCESS6_CSV)
    test_mtime = os.path.getmtime(local_config.TEST_PROCESS6_CSV)
    latest_input_mtime = max(train_mtime, test_mtime)

    kf = KFold(
        n_splits=cfg["cv_n_splits"],
        shuffle=cfg["cv_shuffle"],
        random_state=cfg["cv_random_state"]
    )

    # To store OOF predictions (features for meta-model)
    oof_train = np.zeros((X.shape[0], len(base_model_names)))
    oof_test = np.zeros((X_test.shape[0], len(base_model_names)))

    print(f"Data Shapes: Train={X.shape}, Test={X_test.shape}")
    print(f"Starting stacking with {len(base_model_names)} base models: {', '.join(base_model_names)}")

    for i, model_name in enumerate(base_model_names):
        print(f"\n[{i+1}/{len(base_model_names)}] Model: {model_name}")
        oof_train_path = local_config.OOF_DIR / f"{model_name}_oof_train.npy"
        oof_test_path = local_config.OOF_DIR / f"{model_name}_oof_test.npy"

        # Check if cache exists and is newer than the input data
        cache_valid = False
        if oof_train_path.exists() and oof_test_path.exists():
            cache_mtime = min(os.path.getmtime(oof_train_path), os.path.getmtime(oof_test_path))
            if cache_mtime > latest_input_mtime:
                cache_valid = True
            else:
                print(f"  Cache stale (input data updated). Re-training...")

        if cache_valid:
            print(f"  Loading valid cached predictions...")
            oof_train[:, i] = np.load(oof_train_path)
            oof_test[:, i] = np.load(oof_test_path)
        else:
            # Get best params for this model if they exist in config
            model_cfg = model_config.get_model_config(model_name)
            params = model_cfg.get("base_params", {}).copy()
            
            # Temporary storage for test predictions across folds
            test_preds_fold = np.zeros((X_test.shape[0], cfg["cv_n_splits"]))

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f"    - Processing fold {fold+1}/{cfg['cv_n_splits']}...", end="\r")
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = get_base_model(model_name, params)
                model.fit(X_train, y_train)

                oof_train[val_idx, i] = model.predict(X_val)
                test_preds_fold[:, fold] = model.predict(X_test)
            print() # Newline after folds are done
            
            # Average test predictions across folds
            oof_test[:, i] = test_preds_fold.mean(axis=1)
            
            # Cache the results
            np.save(oof_train_path, oof_train[:, i])
            np.save(oof_test_path, oof_test[:, i])
            
        rmse = np.sqrt(mean_squared_error(y, oof_train[:, i]))
        print(f"    {model_name} OOF RMSE: {rmse:.4f}")

        # Get params for logging
        model_cfg = model_config.get_model_config(model_name)
        log_params = model_cfg.get("base_params", {})
        if not log_params:
            # Fallback for models without base_params (like linear models in some configs)
            log_params = {k: v for k, v in model_cfg.items() if k not in ["submission_name", "submission_filename"]}

        log_model_result(
            model_name=model_name,
            rmse=rmse,
            hyperparams=log_params,
            features=feature_names,
            notes="Base model for Stacking"
        )

    # Meta-model training
    print(f"\nTraining meta-model: {cfg['meta_model'].upper()}")
    if cfg["meta_model"] == "lasso":
        meta_model = Lasso(**cfg["meta_model_params"])
    elif cfg["meta_model"] == "ridge":
        meta_model = Ridge(**cfg["meta_model_params"])
    else:
        meta_model = Ridge() # Default

    meta_model.fit(oof_train, y)
    
    # Calculate Stacking CV/OOF RMSE
    y_stack_pred = meta_model.predict(oof_train)
    stack_rmse = np.sqrt(mean_squared_error(y, y_stack_pred))
    print(f"\nStacking Meta-model OOF RMSE: {stack_rmse:.4f}")

    log_model_result(
        model_name="STACKING_META",
        rmse=stack_rmse,
        hyperparams={
            "meta_model": cfg["meta_model"],
            "meta_params": cfg.get("meta_model_params", {}),
            "base_models": base_model_names
        },
        features=base_model_names,
        notes=f"Full Stacking with {len(base_model_names)} models"
    )
    
    # Final predictions
    print("Generating final predictions...")
    final_pred_log = meta_model.predict(oof_test)
    
    # Validate predictions
    from utils.model_wrapper import (
        validate_predictions_wrapper,
        validate_submission_wrapper
    )
    validate_predictions_wrapper(final_pred_log, "Stacking", target_is_log=True)
    
    final_pred_real = np.expm1(final_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = final_pred_real

    # Validate submission format and ID matching
    validate_submission_wrapper(submission, len(oof_test), "Stacking", test_ids=submission["Id"])

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print(f"\nStacking submission saved: {out_path}")
    
    # Print meta-model weights/coefficients
    if hasattr(meta_model, "coef_"):
        print("\nMeta-model coefficients:")
        for name, coef in zip(base_model_names, meta_model.coef_):
            print(f"  {name:15s}: {coef:.4f}")

