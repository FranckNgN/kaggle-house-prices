#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config_local.local_config as local_config


def add_kmeans(train, test, k=4, cols=("GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"), seed=42):
    cols = [c for c in cols if c in train.columns and c in test.columns]
    X = pd.concat([train[cols], test[cols]]).to_numpy()
    labels = KMeans(k, n_init=20, random_state=seed).fit_predict(StandardScaler().fit_transform(X))
    
    train["KMeansCluster"] = labels[:len(train)].astype("int16")
    test["KMeansCluster"] = labels[len(train):].astype("int16")
    return train, test


if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS3_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS3_CSV)

    for df in (train, test):
        # --- 1. Age Features ---
        df["Age"] = df["YrSold"] - df["YearBuilt"]
        df["Garage_Age"] = df["YrSold"] - df["GarageYrBlt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

        # --- 2. Aggregate Features (The "Magic" Features) ---
        # Total Surface Area
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        
        # Total Bathrooms
        df["TotalBath"] = (df["FullBath"] + (0.5 * df["HalfBath"]) + 
                           df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]))
        
        # Total Porch Area
        df["TotalPorchSF"] = (df["OpenPorchSF"] + df["3SsnPorch"] + 
                              df["EnclosedPorch"] + df["ScreenPorch"] + 
                              df["WoodDeckSF"])

        # --- 3. Binary Flags (Luxury presence) ---
        df["HasPool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
        df["Has2ndFlr"] = df["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
        df["HasGarage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
        df["HasBsmt"] = df["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
        df["HasFireplace"] = df["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

        # --- 4. Skew Correction for New Areas ---
        # Since Process 3 happened already, we log-transform these manually
        for col in ["TotalSF", "TotalPorchSF"]:
            df[col] = np.log1p(df[col])

    # --- 5. Clustering ---
    kmeans_cols = ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"]
    train, test = add_kmeans(train, test, cols=kmeans_cols)

    # --- 6. Save ---
    Path(local_config.TRAIN_PROCESS4_CSV).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(local_config.TRAIN_PROCESS4_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS4_CSV, index=False)
    print(f"✅ Feature Engineering complete. Added TotalSF, TotalBath, Porch, Flags, and Clustering.")
