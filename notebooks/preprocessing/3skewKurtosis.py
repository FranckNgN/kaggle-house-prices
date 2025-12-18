#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Monkey-patch numpy.warnings for compatibility with older scikit-learn and numpy 2.0+
if not hasattr(np, "warnings"):
    import warnings
    np.warnings = warnings

from pathlib import Path
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from config_local import local_config


def is_continuous(series, threshold=0.05):
    if series.dtype not in ['int64', 'float64']:
        return False
    ratio = series.nunique() / len(series)
    return ratio > threshold


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS2_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS2_CSV)

    continuous_cols = [col for col in train.columns if is_continuous(train[col], 0.05)]

    skew_before = train[continuous_cols].apply(lambda x: skew(x.dropna()))
    skewed_cols = skew_before[skew_before.abs() > 0.75].index

    print("Skewed columns (|skew| > 0.75):")
    print(list(skewed_cols))

    # Using a try-except block to handle potential numpy/sklearn version conflicts
    try:
        pt = PowerTransformer(method="yeo-johnson")
        train[skewed_cols] = pt.fit_transform(train[skewed_cols])
        test[skewed_cols] = pt.transform(test[skewed_cols])
        print(f"Applied Yeo-Johnson transform to {len(skewed_cols)} columns.")
    except Exception as e:
        print(f"Warning: Yeo-Johnson transform failed ({e}). Falling back to log1p.")
        for col in skewed_cols:
            train[col] = np.log1p(train[col].clip(lower=0))
            test[col] = np.log1p(test[col].clip(lower=0))


    Path(local_config.TRAIN_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(local_config.TRAIN_PROCESS3_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS3_CSV, index=False)
