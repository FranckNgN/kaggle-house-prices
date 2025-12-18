"""Data validation utilities for the preprocessing pipeline."""
import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import List, Optional

def validate_dataframe(df: pd.DataFrame, name: str, expected_no_nas: bool = True) -> bool:
    """
    Check for missing values in a dataframe.
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        cols_with_nulls = null_counts[null_counts > 0]
        print(f"  ❌ [{name}] Found {total_nulls} missing values in {len(cols_with_nulls)} columns:")
        for col, count in cols_with_nulls.items():
            print(f"    - {col}: {count}")
        if expected_no_nas:
            return False
    else:
        print(f"  ✅ [{name}] No missing values found.")
    
    return True

def validate_column_parity(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "logSP") -> bool:
    """
    Check if train and test dataframes have the same features.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Remove target column from train comparison
    if target_col in train_cols:
        train_cols.remove(target_col)
    
    mismatch_train = train_cols - test_cols
    mismatch_test = test_cols - train_cols
    
    if mismatch_train or mismatch_test:
        print(f"  ❌ Column mismatch detected!")
        if mismatch_train:
            print(f"    - Only in Train: {mismatch_train}")
        if mismatch_test:
            print(f"    - Only in Test: {mismatch_test}")
        return False
    
    print(f"  ✅ Column parity check passed ({len(train_cols)} shared features).")
    return True

def check_skewness(df: pd.DataFrame, name: str, threshold: float = 0.75) -> None:
    """
    Check for features with high skewness.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skews = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    high_skew = skews[abs(skews) > threshold]
    
    if not high_skew.empty:
        print(f"  ⚠️  [{name}] {len(high_skew)} features have skewness > {threshold}:")
        # Print top 5 highest skews
        for col, val in high_skew.sort_values(ascending=False).head(5).items():
            print(f"    - {col}: {val:.4f}")
    else:
        print(f"  ✅ [{name}] No highly skewed features found (threshold={threshold}).")



