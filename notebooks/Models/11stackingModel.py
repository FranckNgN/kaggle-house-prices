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
    X = train.drop(columns=["logSP"]).values
    X_test = test.values

    cfg = model_config.STACKING
    base_model_names = cfg["base_models"]
    
    kf = KFold(
        n_splits=cfg["cv_n_splits"],
        shuffle=cfg["cv_shuffle"],
        random_state=cfg["cv_random_state"]
    )

    # To store OOF predictions (features for meta-model)
    oof_train = np.zeros((X.shape[0], len(base_model_names)))
    oof_test = np.zeros((X_test.shape[0], len(base_model_names)))

    print(f"Starting stacking with {len(base_model_names)} base models...")

    for i, model_name in enumerate(base_model_names):
        oof_train_path = local_config.OOF_DIR / f"{model_name}_oof_train.npy"
        oof_test_path = local_config.OOF_DIR / f"{model_name}_oof_test.npy"

        if oof_train_path.exists() and oof_test_path.exists():
            print(f"  Loading cached predictions for: {model_name}")
            oof_train[:, i] = np.load(oof_train_path)
            oof_test[:, i] = np.load(oof_test_path)
        else:
            print(f"  Training base model: {model_name}")
            
            # Get best params for this model if they exist in config
            model_cfg = model_config.get_model_config(model_name)
            params = model_cfg.get("base_params", {}).copy()
            
            # Temporary storage for test predictions across folds
            test_preds_fold = np.zeros((X_test.shape[0], cfg["cv_n_splits"]))

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = get_base_model(model_name, params)
                model.fit(X_train, y_train)

                oof_train[val_idx, i] = model.predict(X_val)
                test_preds_fold[:, fold] = model.predict(X_test)
            
            # Average test predictions across folds
            oof_test[:, i] = test_preds_fold.mean(axis=1)
            
            # Cache the results
            np.save(oof_train_path, oof_train[:, i])
            np.save(oof_test_path, oof_test[:, i])
            
        rmse = np.sqrt(mean_squared_error(y, oof_train[:, i]))
        print(f"    {model_name} OOF RMSE: {rmse:.4f}")

    # Meta-model training
    print(f"Training meta-model: {cfg['meta_model']}")
    if cfg["meta_model"] == "lasso":
        meta_model = Lasso(**cfg["meta_model_params"])
    elif cfg["meta_model"] == "ridge":
        meta_model = Ridge(**cfg["meta_model_params"])
    else:
        meta_model = Ridge() # Default

    meta_model.fit(oof_train, y)
    
    # Final predictions
    final_pred_log = meta_model.predict(oof_test)
    final_pred_real = np.expm1(final_pred_log)

    submission = load_sample_submission()
    submission["SalePrice"] = final_pred_real

    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    submission.to_csv(out_path, index=False)
    print(f"\nStacking submission saved: {out_path}")
    
    # Print meta-model weights/coefficients
    if hasattr(meta_model, "coef_"):
        print("\nMeta-model coefficients:")
        for name, coef in zip(base_model_names, meta_model.coef_):
            print(f"  {name:15s}: {coef:.4f}")

