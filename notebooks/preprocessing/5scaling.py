#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import config_local.local_config as local_config


def is_continuous(series, threshold=0.05):
    if series.dtype not in ['int64', 'float64']:
        return False
    ratio = series.nunique() / len(series)
    return ratio > threshold


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS4_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS4_CSV)

    X_train = train.drop(columns=["logSP"]).copy()
    y = train["logSP"]
    X_test = test.copy()

    train_numeric = X_train.select_dtypes(include='number')
    # Scale ONLY continuous numeric columns
    # Binary flags and small-range ordinals are often better left unscaled or handled separately
    cols_to_scale = [col for col in train_numeric.columns if is_continuous(X_train[col], 0.05)]

    print(f"Scaling {len(cols_to_scale)} continuous columns...")
    scaler = StandardScaler()

    for col in X_train.columns:
        if col in cols_to_scale:
            X_train[col] = scaler.fit_transform(X_train[[col]])
            X_test[col] = scaler.transform(X_test[[col]])

    X_train = pd.concat([X_train, y], axis=1)

    Path(local_config.TRAIN_PROCESS5_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS5_CSV).parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(local_config.TRAIN_PROCESS5_CSV, index=False)
    X_test.to_csv(local_config.TEST_PROCESS5_CSV, index=False)
