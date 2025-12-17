#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
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

    num_cols = lambda train, threshold=0.05: [col for col in train.columns if is_continuous(train[col], threshold)]

    skew_before = train[num_cols].apply(lambda x: skew(x.dropna()))
    skewed_cols = skew_before[skew_before.abs() > 0.75].index

    print("Skewed columns (|skew| > 0.75):")
    print(list(skewed_cols))

    pt = PowerTransformer(method="yeo-johnson")
    train[skewed_cols] = pt.fit_transform(train[skewed_cols])
    test[skewed_cols] = pt.transform(test[skewed_cols])

    Path(local_config.TRAIN_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(local_config.TRAIN_PROCESS3_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS3_CSV, index=False)
