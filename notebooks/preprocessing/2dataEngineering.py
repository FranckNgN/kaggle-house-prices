#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd

# Monkey-patch numpy.warnings for compatibility with older scikit-learn and numpy 2.0+
if not hasattr(np, "warnings"):
    import warnings
    np.warnings = warnings

from pathlib import Path
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy.stats import boxcox
from config_local import local_config
from utils.engineering import update_engineering_summary


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS1_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS1_CSV)

    salePriceReshape = train["SalePrice"].values.reshape(-1, 1)

    transforms = {
        "SalePrice": lambda x: x.ravel(),
        "logSP": lambda x: np.log1p(x).ravel(),
        "SquareRootSP": lambda x: np.sqrt(x).ravel(),
        "BoxCoxSP": lambda x: boxcox(x.ravel())[0],
        # Yeo-Johnson can fail due to numpy/sklearn version conflicts in some envs
        # "YeoJohnsonSP": lambda x: PowerTransformer(method="yeo-johnson").fit_transform(x).ravel(),
        # "QuantileSP": lambda x: QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(x).ravel()
    }
    
    # Try to add Yeo-Johnson and Quantile if they don't crash
    try:
        salePrice_df = pd.DataFrame({name: func(salePriceReshape) for name, func in transforms.items()})
        # Try individual ones separately
        try:
            salePrice_df["YeoJohnsonSP"] = PowerTransformer(method="yeo-johnson").fit_transform(salePriceReshape).ravel()
        except Exception:
            print("Warning: YeoJohnsonSP failed")
        
        try:
            salePrice_df["QuantileSP"] = QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(salePriceReshape).ravel()
        except Exception:
            print("Warning: QuantileSP failed")
            
    except Exception as e:
        print(f"Warning: Basic transforms failed: {e}")
        # Fallback to absolute minimum
        salePrice_df = pd.DataFrame({
            "SalePrice": salePriceReshape.ravel(),
            "logSP": np.log1p(salePriceReshape).ravel()
        })



    # Outlier removal: Remove houses with very large living area but low price
    # This is a domain-specific outlier (likely data error)
    outlier_mask = (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
    n_outliers_removed = outlier_mask.sum()
    train = train.loc[~outlier_mask]
    
    if n_outliers_removed > 0:
        print(f"Removed {n_outliers_removed} outlier(s): GrLivArea > 4000 & SalePrice < 300000")
    else:
        print("No outliers removed")

    # --- Feature Creation (Pre-Skewness) ---
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        # Age Features
        df["Age"] = df["YrSold"] - df["YearBuilt"]
        df["Garage_Age"] = df["YrSold"] - df["GarageYrBlt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        
        # Aggregate Surface Area
        df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
        
        # Total Bathrooms
        df["TotalBath"] = (df["FullBath"].fillna(0) + (0.5 * df["HalfBath"].fillna(0)) + 
                           df["BsmtFullBath"].fillna(0) + (0.5 * df["BsmtHalfBath"].fillna(0)))
        
        # Total Porch Area
        df["TotalPorchSF"] = (df["OpenPorchSF"].fillna(0) + df["3SsnPorch"].fillna(0) + 
                              df["EnclosedPorch"].fillna(0) + df["ScreenPorch"].fillna(0) + 
                              df["WoodDeckSF"].fillna(0))
        return df

    train = add_basic_features(train)
    test = add_basic_features(test)
    
    # Log engineering details
    update_engineering_summary("Data Engineering", {
        "target_transform": "log1p",
        "outlier_removal": "GrLivArea > 4000 & SalePrice < 300000",
        "new_features": ["Age", "Garage_Age", "RemodAge", "TotalSF", "TotalBath", "TotalPorchSF"]
    })

    # --------------------------------------

    train["logSP"] = salePrice_df['logSP']

    if "SalePrice" in train.columns:
        train = train.drop(columns=["SalePrice"])

    Path(local_config.SALEPRICE_TRANSFORMS_CSV).parent.mkdir(parents=True, exist_ok=True)
    salePrice_df.to_csv(local_config.SALEPRICE_TRANSFORMS_CSV, index=False)

    Path(local_config.TRAIN_PROCESS2_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS2_CSV).parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(local_config.TRAIN_PROCESS2_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS2_CSV, index=False)
