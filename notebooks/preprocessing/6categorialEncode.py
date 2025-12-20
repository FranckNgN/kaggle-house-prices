#!/usr/bin/env python
# coding: utf-8

"""
Categorical Encoding (Stage 6)
===============================
One-hot encodes categorical features, but keeps high-cardinality categoricals
for target encoding in stage 8.

Best Practice: High-cardinality categoricals (like Neighborhood) are better
target-encoded than one-hot encoded to avoid curse of dimensionality.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import config_local.local_config as local_config
from utils.engineering import update_engineering_summary


def identify_categorical_columns(df: pd.DataFrame) -> tuple:
    """
    Identify categorical columns and separate by cardinality.
    
    Returns:
        (low_cardinality_cols, high_cardinality_cols)
    """
    categorical_cols = []
    
    for col in df.columns:
        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Skip already encoded columns (like KMeansCluster which is a string)
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
    
    # Separate by cardinality
    # High-cardinality: > 10 unique values (better for target encoding)
    # Low-cardinality: <= 10 unique values (better for one-hot encoding)
    low_cardinality = []
    high_cardinality = []
    
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > 10:
            high_cardinality.append(col)
        else:
            low_cardinality.append(col)
    
    return low_cardinality, high_cardinality


if __name__ == "__main__":
    print("="*70)
    print("CATEGORICAL ENCODING (Stage 6)")
    print("="*70)
    
    train = pd.read_csv(local_config.TRAIN_PROCESS5_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS5_CSV)

    print(f"\nInitial shape - Train: {train.shape}, Test: {test.shape}")

    y = train["logSP"].copy()
    X_train = train.drop(columns=["logSP"]).copy()
    X_test = test.copy()

    # Identify categorical columns
    print("\nIdentifying categorical columns...")
    low_card_cols, high_card_cols = identify_categorical_columns(X_train)
    
    print(f"  Low-cardinality (one-hot encode): {len(low_card_cols)} columns")
    if low_card_cols:
        print(f"    Examples: {low_card_cols[:5]}")
    print(f"  High-cardinality (keep for target encoding): {len(high_card_cols)} columns")
    if high_card_cols:
        print(f"    Examples: {high_card_cols[:5]}")
    
    # Separate numeric, low-cardinality categorical, and high-cardinality categorical
    numeric_cols = [col for col in X_train.columns 
                   if pd.api.types.is_numeric_dtype(X_train[col])]
    
    # One-hot encode only low-cardinality categoricals
    # Keep high-cardinality categoricals as-is for target encoding in stage 8
    all_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    
    # Extract columns to encode vs keep
    cols_to_encode = numeric_cols + low_card_cols
    cols_to_keep = high_card_cols
    
    # One-hot encode low-cardinality categoricals
    if low_card_cols:
        X_to_encode = all_X[cols_to_encode].copy()
        X_encoded = pd.get_dummies(X_to_encode, drop_first=True, dtype="int8", 
                                  columns=low_card_cols)
    else:
        X_encoded = all_X[numeric_cols].copy()
    
    # Keep high-cardinality categoricals as-is
    if cols_to_keep:
        X_high_card = all_X[cols_to_keep].copy()
        # Combine encoded + high-cardinality
        all_X_enc = pd.concat([X_encoded.reset_index(drop=True), 
                               X_high_card.reset_index(drop=True)], axis=1)
    else:
        all_X_enc = X_encoded

    # DO NOT scale binary dummies. They are already on a 0-1 scale which is ideal.
    # The continuous features were already scaled in stage 5.
    
    X_train_enc = all_X_enc.iloc[:len(X_train), :].copy()
    X_test_enc = all_X_enc.iloc[len(X_train):, :].copy()

    train_enc = pd.concat([y.reset_index(drop=True), X_train_enc.reset_index(drop=True)], axis=1)

    # Log engineering details
    update_engineering_summary("Categorical Encoding", {
        "one_hot_encoded": len(low_card_cols),
        "kept_for_target_encoding": len(high_card_cols),
        "low_cardinality_cols": low_card_cols[:10] if low_card_cols else [],
        "high_cardinality_cols": high_card_cols[:10] if high_card_cols else []
    })

    Path(local_config.TRAIN_PROCESS6_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.TEST_PROCESS6_CSV).parent.mkdir(parents=True, exist_ok=True)

    train_enc.to_csv(local_config.TRAIN_PROCESS6_CSV, index=False)
    X_test_enc.to_csv(local_config.TEST_PROCESS6_CSV, index=False)
    
    print(f"\nFinal shape - Train: {train_enc.shape}, Test: {X_test_enc.shape}")
    print(f"  One-hot encoded: {len(low_card_cols)} categorical columns")
    print(f"  Kept for target encoding: {len(high_card_cols)} categorical columns")
    print("\n" + "="*70)
    print("Categorical Encoding complete!")
    print("="*70)
