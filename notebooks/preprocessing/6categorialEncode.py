#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import config_local.local_config as local_config


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS5_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS5_CSV)

    y = train["logSP"].copy()
    X_train = train.drop(columns=["logSP"]).copy()
    X_test = test.copy()

    all_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    all_X_enc = pd.get_dummies(all_X, drop_first=True, dtype="int8")

    # DO NOT scale binary dummies. They are already on a 0-1 scale which is ideal.
    # The continuous features were already scaled in stage 5.
    
    X_train_enc = all_X_enc.iloc[:len(X_train), :].copy()
    X_test_enc = all_X_enc.iloc[len(X_train):, :].copy()

    train_enc = pd.concat([y.reset_index(drop=True), X_train_enc.reset_index(drop=True)], axis=1)

    Path(local_config.TRAIN_PROCESS6_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS6_CSV).parent.mkdir(parents=True, exist_ok=True)

    train_enc.to_csv(local_config.TRAIN_PROCESS6_CSV, index=False)
    X_test_enc.to_csv(local_config.TEST_PROCESS6_CSV, index=False)
