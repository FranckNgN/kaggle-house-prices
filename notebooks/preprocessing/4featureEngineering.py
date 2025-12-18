import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config_local.local_config as local_config
from utils.engineering import update_engineering_summary


from typing import Tuple, Dict


def add_era_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group years into decades to capture architectural eras."""
    df["Decade"] = (df["YearBuilt"] // 10) * 10
    return df


def add_luxury_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flags for the presence of luxury/essential features."""
    features = {
        "HasPool": "PoolArea",
        "Has2ndFlr": "2ndFlrSF",
        "HasGarage": "GarageArea",
        "HasBsmt": "TotalBsmtSF",
        "HasFireplace": "Fireplaces"
    }
    for flag, col in features.items():
        if col in df.columns:
            df[flag] = df[col].apply(lambda x: 1 if x > 0 else 0)
        else:
            df[flag] = 0
    
    # 1. Noise Reduction: Simplify conditions
    if "Condition1" in df.columns:
        df["IsNormalCondition"] = df["Condition1"].apply(lambda x: 1 if x == "Norm" else 0)
    else:
        df["IsNormalCondition"] = 1
    
    return df


def add_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Convert qualitative ratings into ordered numerical values and drop original columns."""
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "<None>": 0}
    qual_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    
    for col in qual_cols:
        if col in df.columns:
            df[f"{col}_Score"] = df[col].map(qual_map).fillna(0).astype("int8")
            # Drop original categorical column to avoid redundant OHE later
            df.drop(columns=[col], inplace=True)
            
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create multiplicative features between quality and quantity."""
    # 2. Qualitative x Quantitative interactions
    # Luxury Area: High quality extra square footage is worth more
    df["Qual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    
    # Heart of the Home: Kitchen quality impact on total size value
    if "KitchenQual_Score" in df.columns:
        df["Kitchen_x_TotalSF"] = df["KitchenQual_Score"] * df["TotalSF"]
    else:
        df["Kitchen_x_TotalSF"] = 3 * df["TotalSF"] # Default to TA
    
    # Maintenance impact on Age
    df["Cond_x_Age"] = df["OverallCond"] * df["Age"]
    
    return df


def add_group_benchmarks(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate relative performance features based on Neighborhood, SubClass, Zoning, Quality, and Era."""
    # Define benchmarks: (Group_Col, target_col, aggregation_type)
    benchmarks = [
        ("Neighborhood", "TotalSF", "median"),
        ("Neighborhood", "OverallQual", "mean"),
        ("Neighborhood", "Age", "median"),
        ("MSSubClass", "TotalSF", "median"),
        ("MSZoning", "LotArea", "median"),
        ("OverallQual", "TotalSF", "median"),
        ("Decade", "TotalSF", "median")
    ]

    for group_col, target_col, agg_type in benchmarks:
        if group_col not in train.columns or target_col not in train.columns:
            continue
            
        # 1. Compute benchmark using ONLY training data to prevent leakage
        stats_map = train.groupby(group_col)[target_col].agg(agg_type)
        global_fallback = train[target_col].agg(agg_type)
        
        # Ensure fallback is not zero to avoid division by zero
        if global_fallback == 0:
            global_fallback = 1.0

        for df in [train, test]:
            # 2. Map benchmarks back to data
            benchmark_col = f"Temp_{group_col}_{target_col}_Benchmark"
            df[benchmark_col] = df[group_col].map(stats_map).fillna(global_fallback)
            
            # Ensure benchmark is not zero
            df[benchmark_col] = df[benchmark_col].replace(0, global_fallback)

            # 3. Create relative features
            feature_name = f"{target_col}_to_{group_col}_Ratio"
            if target_col == "Age":
                df[f"Age_vs_{group_col}_Avg"] = df[target_col] - df[benchmark_col]
            else:
                df[feature_name] = df[target_col] / df[benchmark_col]
                # Fill any remaining NaNs/Infs
                df[feature_name] = df[feature_name].fillna(1.0).replace([np.inf, -np.inf], 1.0)

            # 4. Cleanup
            df.drop(columns=[benchmark_col], inplace=True)

    return train, test


def add_kmeans_clusters(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    k: int = 4, 
    cols: tuple = ("GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"), 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform K-Means clustering on both datasets to capture non-linear patterns."""
    cols = [c for c in cols if c in train.columns and c in test.columns]
    X = pd.concat([train[cols], test[cols]]).to_numpy()
    
    # Fit and predict labels
    scaler = StandardScaler()
    labels = KMeans(k, n_init=20, random_state=seed).fit_predict(scaler.fit_transform(X))
    
    train["KMeansCluster"] = [f"Cluster_{l}" for l in labels[:len(train)]]
    test["KMeansCluster"] = [f"Cluster_{l}" for l in labels[len(train):]]
    return train, test


def main() -> None:
    """Main execution entry point."""
    print("Loading Processed 3 data...")
    train = pd.read_csv(local_config.TRAIN_PROCESS3_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS3_CSV)

    print(f"Initial NaNs in train: {train.isna().sum().sum()}")

    print("Applying basic feature engineering...")
    # Process train
    train = add_era_features(train)
    train = add_luxury_flags(train)
    train = add_ordinal_encoding(train)
    train = add_interaction_features(train)
    
    # Process test
    test = add_era_features(test)
    test = add_luxury_flags(test)
    test = add_ordinal_encoding(test)
    test = add_interaction_features(test)

    print("Adding group-based relative benchmarks...")
    train, test = add_group_benchmarks(train, test)

    print("Running K-Means clustering...")
    train, test = add_kmeans_clusters(train, test)

    # Log engineering details
    update_engineering_summary("Feature Engineering", {
        "kmeans_k": 4,
        "kmeans_cols": ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"],
        "luxury_flags": ["HasPool", "Has2ndFlr", "HasGarage", "HasBsmt", "HasFireplace"],
        "interactions": ["Qual_x_TotalSF", "Kitchen_x_TotalSF", "Cond_x_Age"],
        "benchmarks": ["Neighborhood", "MSSubClass", "MSZoning", "OverallQual", "Decade"]
    })

    print("Saving Processed 4 data...")
    Path(local_config.TRAIN_PROCESS4_CSV).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(local_config.TRAIN_PROCESS4_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS4_CSV, index=False)
    
    print("Feature Engineering complete!")


if __name__ == "__main__":
    main()
