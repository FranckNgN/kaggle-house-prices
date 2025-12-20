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

    # Transform columns one by one to handle failures gracefully
    # Some columns with extreme skewness may cause PowerTransformer to fail
    transformed_cols = []
    failed_cols = []
    
    for col in skewed_cols:
        try:
            pt = PowerTransformer(method="yeo-johnson")
            train_transformed = pt.fit_transform(train[[col]])
            test_transformed = pt.transform(test[[col]])
            
            # Check if transformation produced valid results (not all zeros or all same)
            if (train_transformed.std() > 1e-6 and 
                len(np.unique(train_transformed)) > 1 and
                not np.isnan(train_transformed).any()):
                train[col] = train_transformed.ravel()
                test[col] = test_transformed.ravel()
                transformed_cols.append(col)
            else:
                # Transformation produced invalid results, use log1p instead
                print(f"  Warning: {col} transformation produced invalid results, using log1p instead.")
                train[col] = np.log1p(train[col].clip(lower=0))
                test[col] = np.log1p(test[col].clip(lower=0))
                failed_cols.append(col)
        except Exception as e:
            # PowerTransformer failed, use log1p as fallback
            print(f"  Warning: {col} Yeo-Johnson transform failed ({e}), using log1p instead.")
            train[col] = np.log1p(train[col].clip(lower=0))
            test[col] = np.log1p(test[col].clip(lower=0))
            failed_cols.append(col)
    
    print(f"Applied Yeo-Johnson transform to {len(transformed_cols)} columns.")
    if failed_cols:
        print(f"Used log1p fallback for {len(failed_cols)} columns: {failed_cols}")


    Path(local_config.TRAIN_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(local_config.TRAIN_PROCESS3_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS3_CSV, index=False)
