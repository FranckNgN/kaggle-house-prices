"""
Target Encoding for Categorical Features
========================================
Implements cross-validated target encoding to capture category-target relationships
without overfitting.

This is a high-impact feature engineering technique that often improves model performance
by 0.005-0.015 RMSE.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import config_local.local_config as local_config
from utils.engineering import update_engineering_summary


def target_encode_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_col: str,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 1.0,
    noise_level: float = 0.01,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series]:
    """
    Perform cross-validated target encoding with smoothing and noise.
    
    Args:
        train: Training dataframe
        test: Test dataframe
        cat_col: Categorical column to encode
        target_col: Target column name
        n_splits: Number of CV folds
        smoothing: Smoothing factor (higher = more global mean influence)
        noise_level: Standard deviation of noise to add during training
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (train_encoded, test_encoded) Series
    """
    # Global mean
    global_mean = train[target_col].mean()
    
    # Initialize encoded columns
    train_encoded = pd.Series(index=train.index, dtype=float)
    test_encoded = pd.Series(index=test.index, dtype=float)
    
    # Cross-validation for training set
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        # Calculate mean target for each category in training fold
        category_means = train_fold.groupby(cat_col)[target_col].mean()
        category_counts = train_fold.groupby(cat_col)[target_col].count()
        
        # Smoothing: blend category mean with global mean
        # Higher smoothing = more weight on global mean
        smoothed_means = (
            (category_means * category_counts + global_mean * smoothing) /
            (category_counts + smoothing)
        )
        
        # Map to validation fold
        val_encoded = val_fold[cat_col].map(smoothed_means).fillna(global_mean)
        
        # Add noise during training to prevent overfitting
        np.random.seed(random_state + fold)
        noise = np.random.normal(0, noise_level * val_encoded.std(), len(val_encoded))
        train_encoded.iloc[val_idx] = val_encoded + noise
    
    # For test set: use full training data
    category_means = train.groupby(cat_col)[target_col].mean()
    category_counts = train.groupby(cat_col)[target_col].count()
    
    smoothed_means = (
        (category_means * category_counts + global_mean * smoothing) /
        (category_counts + smoothing)
    )
    
    test_encoded = test[cat_col].map(smoothed_means).fillna(global_mean)
    
    return train_encoded, test_encoded


def apply_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: List[str],
    target_col: str = "logSP",
    n_splits: int = 5,
    smoothing: float = 1.0,
    noise_level: float = 0.01
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply target encoding to multiple categorical columns.
    
    Args:
        train: Training dataframe
        test: Test dataframe
        cat_cols: List of categorical column names to encode
        target_col: Target column name
        n_splits: Number of CV folds
        smoothing: Smoothing factor
        noise_level: Noise level for training set
    
    Returns:
        Tuple of (train_df, test_df) with new encoded columns
    """
    train = train.copy()
    test = test.copy()
    
    encoded_cols = []
    
    for col in cat_cols:
        if col not in train.columns:
            print(f"  Warning: {col} not found in dataframe. Skipping.")
            continue
        
        print(f"  Encoding {col}...")
        train_encoded, test_encoded = target_encode_cv(
            train, test, col, target_col, n_splits, smoothing, noise_level
        )
        
        # Create new column
        new_col_name = f"{col}_TargetEnc"
        train[new_col_name] = train_encoded
        test[new_col_name] = test_encoded
        encoded_cols.append(new_col_name)
        
        # Print statistics
        print(f"    Train range: [{train[new_col_name].min():.4f}, {train[new_col_name].max():.4f}]")
        print(f"    Test range: [{test[new_col_name].min():.4f}, {test[new_col_name].max():.4f}]")
        print(f"    Unique values: {train[new_col_name].nunique()}")
    
    print(f"\n  Created {len(encoded_cols)} target-encoded features")
    return train, test, encoded_cols


def add_neighborhood_price_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "logSP",
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add neighborhood price statistics using cross-validated target encoding.
    
    This creates multiple statistics (mean, median, std, min, max) for each
    neighborhood based on the target variable, using proper CV to prevent leakage.
    """
    if "Neighborhood" not in train.columns or target_col not in train.columns:
        return train, test
    
    print("  Adding Neighborhood price statistics (cross-validated)...")
    
    global_mean = train[target_col].mean()
    global_std = train[target_col].std()
    global_median = train[target_col].median()
    
    # Initialize columns
    for stat in ["mean", "median", "std", "min", "max"]:
        train[f"Neighborhood_{stat}_logSP"] = np.nan
        test[f"Neighborhood_{stat}_logSP"] = np.nan
    
    # Cross-validated encoding for training set
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        neighborhood_stats = train_fold.groupby("Neighborhood")[target_col].agg([
            "mean", "median", "std", "min", "max"
        ]).fillna({
            "mean": global_mean,
            "median": global_median,
            "std": global_std,
            "min": train_fold[target_col].min(),
            "max": train_fold[target_col].max()
        })
        
        for stat in ["mean", "median", "std", "min", "max"]:
            val_encoded = val_fold["Neighborhood"].map(neighborhood_stats[stat]).fillna(
                global_mean if stat == "mean" else (global_median if stat == "median" else global_std)
            )
            train.loc[val_fold.index, f"Neighborhood_{stat}_logSP"] = val_encoded
    
    # For test set: use full training data
    neighborhood_stats = train.groupby("Neighborhood")[target_col].agg([
        "mean", "median", "std", "min", "max"
    ]).fillna({
        "mean": global_mean,
        "median": global_median,
        "std": global_std,
        "min": train[target_col].min(),
        "max": train[target_col].max()
    })
    
    for stat in ["mean", "median", "std", "min", "max"]:
        test[f"Neighborhood_{stat}_logSP"] = test["Neighborhood"].map(neighborhood_stats[stat]).fillna(
            global_mean if stat == "mean" else (global_median if stat == "median" else global_std)
        )
    
    print(f"    Created 5 Neighborhood statistics features")
    return train, test


def main() -> None:
    """Main execution entry point."""
    print("="*70)
    print("TARGET ENCODING FOR CATEGORICAL FEATURES")
    print("="*70)
    
    print("\nLoading Processed 7 data (after feature selection)...")
    train = pd.read_csv(local_config.TRAIN_PROCESS7_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS7_CSV)
    
    print(f"Initial shape - Train: {train.shape}, Test: {test.shape}")
    
    # Step 1: Add Neighborhood price statistics (special case - creates multiple features)
    if "Neighborhood" in train.columns:
        train, test = add_neighborhood_price_stats(train, test, target_col="logSP")
        
        # Step 1b: Add Neighborhood Price per Square Foot (target-encoded)
        if "TotalSF" in train.columns:
            print("  Adding Neighborhood Price per Square Foot (cross-validated)...")
            # Create price per SF in log space (logSP / TotalSF)
            train["PricePerSF_log"] = train["logSP"] / (train["TotalSF"] + 1)
            # Test set doesn't have logSP, so set to NaN (will be replaced with target encoding)
            test["PricePerSF_log"] = np.nan
            
            # Target encode PricePerSF by Neighborhood
            train_encoded, test_encoded = target_encode_cv(
                train, test, "Neighborhood", "PricePerSF_log", 
                n_splits=5, smoothing=1.0, noise_level=0.01, random_state=42
            )
            train["Neighborhood_PricePerSF_TargetEnc"] = train_encoded
            test["Neighborhood_PricePerSF_TargetEnc"] = test_encoded
            
            # Drop temporary column
            train = train.drop(columns=["PricePerSF_log"])
            test = test.drop(columns=["PricePerSF_log"])
            print(f"    Created Neighborhood_PricePerSF_TargetEnc feature")
    
    # Step 2: Identify other categorical columns that would benefit from target encoding
    # Focus on high-cardinality or important categoricals that weren't one-hot encoded
    target_encode_cols = [
        "MSZoning",          # Important zoning information
        "MSSubClass",        # Building class
        "HouseStyle",        # House style
        "RoofStyle",         # Roof style
        "Exterior1st",       # Exterior covering
        "Foundation",        # Foundation type
        "Heating",           # Heating type
        "SaleType",          # Sale type
        "SaleCondition",     # Sale condition
    ]
    
    # Filter to only columns that exist and weren't already one-hot encoded
    # (check if column exists and is categorical, not numeric)
    available_cols = []
    for col in target_encode_cols:
        if col in train.columns:
            # Check if it's still categorical (not one-hot encoded)
            if train[col].dtype == 'object' or train[col].dtype.name == 'category':
                available_cols.append(col)
    
    if available_cols:
        print(f"\nTarget encoding {len(available_cols)} additional categorical features:")
        print(f"  {', '.join(available_cols)}")
        
        # Apply target encoding
        train, test, encoded_cols = apply_target_encoding(
            train,
            test,
            available_cols,
            target_col="logSP",
            n_splits=5,
            smoothing=1.0,  # Can tune this (higher = more conservative)
            noise_level=0.01  # Small noise to prevent overfitting
        )
    else:
        encoded_cols = []
        print("\nNo additional categorical features to target encode.")
    
    # Count all target-encoded features (including neighborhood stats)
    neighborhood_stats_cols = [col for col in train.columns if "Neighborhood_" in col and "_logSP" in col]
    all_encoded_cols = encoded_cols + neighborhood_stats_cols
    
    # ================================================================
    # DROP ALL ORIGINAL CATEGORICAL COLUMNS (replaced by target-encoded versions)
    # ================================================================
    # After target encoding, ALL original categorical columns are redundant
    # and cause issues for tree models that can't handle object/string columns
    # We should drop ALL categorical columns, not just the ones we target-encoded
    print(f"\nDropping ALL original categorical columns (replaced by target-encoded versions)...")
    
    # Find ALL categorical columns (object/string types) in the dataframe
    all_categorical_cols = train.select_dtypes(exclude=['number']).columns.tolist()
    
    # Exclude the target column if it's somehow categorical (shouldn't be)
    if "logSP" in all_categorical_cols:
        all_categorical_cols.remove("logSP")
    
    if all_categorical_cols:
        print(f"  Dropping {len(all_categorical_cols)} original categorical columns: {all_categorical_cols}")
        train = train.drop(columns=all_categorical_cols, errors='ignore')
        test = test.drop(columns=all_categorical_cols, errors='ignore')
        print(f"  [SUCCESS] Removed all redundant categorical columns")
        
        # Verify no categorical columns remain
        remaining_cat = train.select_dtypes(exclude=['number']).columns.tolist()
        if remaining_cat:
            print(f"  [WARNING] {len(remaining_cat)} categorical columns still remain: {remaining_cat}")
        else:
            print(f"  [VERIFIED] No categorical columns remain in final dataset")
    else:
        print(f"  No categorical columns found to drop")
    
    # Log engineering details
    update_engineering_summary("Target Encoding", {
        "encoded_features": all_encoded_cols,
        "neighborhood_stats": len(neighborhood_stats_cols),
        "other_target_encoded": len(encoded_cols),
        "dropped_original_cols": all_categorical_cols if 'all_categorical_cols' in locals() else [],
        "n_splits": 5,
        "smoothing": 1.0,
        "noise_level": 0.01
    })
    
    # ================================================================
    # CRITICAL: Scale newly created target-encoded features
    # ================================================================
    # Target-encoded features are NEW numeric features created AFTER stage 5 scaling
    # They need to be scaled for linear models (ElasticNet, Ridge, Lasso) to converge properly
    print(f"\nScaling newly created target-encoded features...")
    
    # Identify all target-encoded numeric features that need scaling
    # These are continuous numeric features (not binary 0/1 from one-hot encoding)
    target_encoded_numeric = all_encoded_cols  # All target-encoded features are numeric
    
    if target_encoded_numeric:
        # Filter to only columns that exist and are numeric
        cols_to_scale = [col for col in target_encoded_numeric 
                        if col in train.columns and pd.api.types.is_numeric_dtype(train[col])]
        
        if cols_to_scale:
            print(f"  Scaling {len(cols_to_scale)} target-encoded features: {cols_to_scale[:5]}{'...' if len(cols_to_scale) > 5 else ''}")
            
            # Fit scaler on training data only, then transform both train and test
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train[cols_to_scale])
            test_scaled = scaler.transform(test[cols_to_scale])
            
            # Update scaled columns
            for i, col in enumerate(cols_to_scale):
                train[col] = train_scaled[:, i]
                test[col] = test_scaled[:, i]
            
            print(f"  [SUCCESS] Scaled target-encoded features for proper linear model convergence")
        else:
            print(f"  No target-encoded numeric features found to scale")
    else:
        print(f"  No target-encoded features to scale")
    
    # ================================================================
    # CREATE NEIGHBORHOOD INTERACTION FEATURES (after target encoding)
    # ================================================================
    # These features depend on Neighborhood_mean_logSP which is created above
    print(f"\nCreating Neighborhood interaction features...")
    interaction_features_created = []
    
    if "Neighborhood_mean_logSP" in train.columns:
        # Neighborhood × Quality interactions
        if "OverallQual" in train.columns:
            train["Neighborhood_x_Qual"] = train["Neighborhood_mean_logSP"] * train["OverallQual"]
            test["Neighborhood_x_Qual"] = test["Neighborhood_mean_logSP"] * test["OverallQual"]
            interaction_features_created.append("Neighborhood_x_Qual")
        
        # Neighborhood × Size interactions
        if "TotalSF" in train.columns:
            train["Neighborhood_x_Size"] = train["Neighborhood_mean_logSP"] * train["TotalSF"]
            test["Neighborhood_x_Size"] = test["Neighborhood_mean_logSP"] * test["TotalSF"]
            interaction_features_created.append("Neighborhood_x_Size")
        
        # Neighborhood × Age interactions
        if "Age" in train.columns:
            train["Neighborhood_x_Age"] = train["Neighborhood_mean_logSP"] * train["Age"]
            test["Neighborhood_x_Age"] = test["Neighborhood_mean_logSP"] * test["Age"]
            interaction_features_created.append("Neighborhood_x_Age")
        
        if interaction_features_created:
            print(f"  Created {len(interaction_features_created)} Neighborhood interaction features: {interaction_features_created}")
            # Scale these new interaction features (they're continuous numeric)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train[interaction_features_created])
            test_scaled = scaler.transform(test[interaction_features_created])
            
            for i, col in enumerate(interaction_features_created):
                train[col] = train_scaled[:, i]
                test[col] = test_scaled[:, i]
            
            print(f"  [SUCCESS] Scaled Neighborhood interaction features")
        else:
            print(f"  No Neighborhood interaction features created (missing dependencies)")
    else:
        print(f"  Skipping Neighborhood interactions (Neighborhood_mean_logSP not found)")
    
    # Save to new stage (process 8)
    print(f"\nSaving Processed 8 data...")
    Path(local_config.INTERIM_TRAIN_DIR / "train_process8.csv").parent.mkdir(parents=True, exist_ok=True)
    Path(local_config.INTERIM_TEST_DIR / "test_process8.csv").parent.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(local_config.INTERIM_TRAIN_DIR / "train_process8.csv", index=False)
    test.to_csv(local_config.INTERIM_TEST_DIR / "test_process8.csv", index=False)
    
    print(f"Final shape - Train: {train.shape}, Test: {test.shape}")
    print(f"Added {len(all_encoded_cols)} target-encoded features:")
    print(f"  - Neighborhood statistics: {len(neighborhood_stats_cols)} features")
    if encoded_cols:
        print(f"  - Other categoricals: {len(encoded_cols)} features")
    print("\n" + "="*70)
    print("Target Encoding complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Update model scripts to use process8 data (final processed data)")
    print("2. Retrain models and compare performance")


if __name__ == "__main__":
    main()

