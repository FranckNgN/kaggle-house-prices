#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy.stats import boxcox
from config_local import local_config


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS1_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS1_CSV)

    salePriceReshape = train["SalePrice"].values.reshape(-1, 1)

    transforms = {
        "SalePrice": lambda x: x.ravel(),
        "logSP": lambda x: np.log1p(x).ravel(),
        "SquareRootSP": lambda x: np.sqrt(x).ravel(),
        "BoxCoxSP": lambda x: boxcox(x.ravel())[0],
        "YeoJohnsonSP": lambda x: PowerTransformer(method="yeo-johnson").fit_transform(x).ravel(),
        "QuantileSP": lambda x: QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(x).ravel()
    }
    salePrice_df = pd.DataFrame({name: func(salePriceReshape) for name, func in transforms.items()})

    outlier_mask = (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
    train = train.loc[~outlier_mask]

    train["logSP"] = salePrice_df['logSP']

    if "SalePrice" in train.columns:
        train = train.drop(columns=["SalePrice"])

    Path(local_config.SALEPRICE_TRANSFORMS_CSV).parent.mkdir(parents=True, exist_ok=True)
    salePrice_df.to_csv(local_config.SALEPRICE_TRANSFORMS_CSV, index=False)

    Path(local_config.TRAIN_PROCESS2_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS2_CSV).parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(local_config.TRAIN_PROCESS2_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS2_CSV, index=False)
