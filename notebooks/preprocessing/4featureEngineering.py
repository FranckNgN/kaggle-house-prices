import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import config_local.local_config as local_config
from utils.engineering import update_engineering_summary


from typing import Tuple, Dict, List


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
    cols: tuple = ("GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual", "TotalSF", "LotArea", "TotalBath", "Age"), 
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


def add_neighborhood_price_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "logSP",
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add neighborhood price statistics using cross-validated target encoding."""
    if "Neighborhood" not in train.columns or target_col not in train.columns:
        return train, test
    
    global_mean = train[target_col].mean()
    global_std = train[target_col].std()
    global_median = train[target_col].median()
    
    for stat in ["mean", "median", "std", "min", "max"]:
        train[f"Neighborhood_{stat}_logSP"] = np.nan
        test[f"Neighborhood_{stat}_logSP"] = np.nan
    
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
    
    return train, test


def add_polynomial_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add squared terms for key features."""
    key_features = ["TotalSF", "OverallQual", "GrLivArea", "GarageArea", "Age", "LotArea", "TotalBath"]
    created_features = []
    for feat in key_features:
        if feat in df.columns:
            squared_feat = f"{feat}_squared"
            df[squared_feat] = df[feat] ** 2
            created_features.append(squared_feat)
    return df, created_features


def add_ratio_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add ratio features for efficiency and density metrics."""
    created_features = []
    
    if "LotArea" in df.columns and "TotalSF" in df.columns:
        df["LotArea_to_TotalSF_Ratio"] = df["LotArea"] / (df["TotalSF"] + 1)
        created_features.append("LotArea_to_TotalSF_Ratio")
    
    if "GarageArea" in df.columns and "TotalSF" in df.columns:
        df["GarageArea_to_TotalSF_Ratio"] = df["GarageArea"] / (df["TotalSF"] + 1)
        created_features.append("GarageArea_to_TotalSF_Ratio")
    
    if "TotalPorchSF" in df.columns and "TotalSF" in df.columns:
        df["Porch_to_TotalSF_Ratio"] = df["TotalPorchSF"] / (df["TotalSF"] + 1)
        created_features.append("Porch_to_TotalSF_Ratio")
    
    if "GrLivArea" in df.columns and "TotalSF" in df.columns:
        df["Living_to_Total_Ratio"] = df["GrLivArea"] / (df["TotalSF"] + 1)
        created_features.append("Living_to_Total_Ratio")
    
    if "TotalBsmtSF" in df.columns and "TotalSF" in df.columns:
        df["Bsmt_to_Total_Ratio"] = df["TotalBsmtSF"] / (df["TotalSF"] + 1)
        created_features.append("Bsmt_to_Total_Ratio")
    
    if "TotRmsAbvGrd" in df.columns and "GrLivArea" in df.columns:
        df["Rooms_per_SF"] = df["TotRmsAbvGrd"] / (df["GrLivArea"] + 1)
        created_features.append("Rooms_per_SF")
    
    if "BedroomAbvGr" in df.columns and "GrLivArea" in df.columns:
        df["Bedrooms_per_SF"] = df["BedroomAbvGr"] / (df["GrLivArea"] + 1)
        created_features.append("Bedrooms_per_SF")
    
    if "TotalBath" in df.columns and "TotalSF" in df.columns:
        df["Bath_per_SF"] = df["TotalBath"] / (df["TotalSF"] + 1)
        created_features.append("Bath_per_SF")
    
    for feat in created_features:
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df, created_features


def add_temporal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add temporal features for year/month effects."""
    created_features = []
    
    if "YrSold" in df.columns:
        df["YearsSince2006"] = df["YrSold"] - 2006
        created_features.append("YearsSince2006")
        df["MarketCycle"] = (df["YrSold"] - 2006) % 4
        created_features.append("MarketCycle")
    
    if "MoSold" in df.columns:
        df["Quarter"] = ((df["MoSold"] - 1) // 3) + 1
        created_features.append("Quarter")
        df["PeakSeason"] = ((df["MoSold"] >= 3) & (df["MoSold"] <= 8)).astype(int)
        created_features.append("PeakSeason")
        df["EndOfYear"] = ((df["MoSold"] >= 11) | (df["MoSold"] <= 1)).astype(int)
        created_features.append("EndOfYear")
    
    if "Age" in df.columns:
        df["AgeAtSale"] = df["Age"]
        created_features.append("AgeAtSale")
    
    return df, created_features


def add_quality_aggregates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add aggregate quality metrics across all quality features."""
    created_features = []
    quality_cols = [col for col in df.columns if "_Score" in col or "Qual" in col or "Cond" in col]
    quality_scores = []
    for col in quality_cols:
        if col in ["OverallQual", "OverallCond"]:
            quality_scores.append(col)
        elif "_Score" in col:
            quality_scores.append(col)
    
    if len(quality_scores) > 0:
        df["AvgQuality"] = df[quality_scores].mean(axis=1)
        created_features.append("AvgQuality")
        df["MaxQuality"] = df[quality_scores].max(axis=1)
        created_features.append("MaxQuality")
        df["MinQuality"] = df[quality_scores].min(axis=1)
        created_features.append("MinQuality")
        df["QualityRange"] = df[quality_scores].max(axis=1) - df[quality_scores].min(axis=1)
        created_features.append("QualityRange")
        df["ExcellentFeatures"] = (df[quality_scores] >= 4).sum(axis=1)
        created_features.append("ExcellentFeatures")
        df["PoorFeatures"] = (df[quality_scores] <= 2).sum(axis=1)
        created_features.append("PoorFeatures")
    
    return df, created_features


def add_advanced_clustering(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_values: List[int] = [6, 8],
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Add multiple K-Means clusters with different k values and feature sets."""
    created_features = []
    
    cluster_configs = [
        {
            "name": "Size",
            "features": ["GrLivArea", "TotalBsmtSF", "LotArea", "GarageArea", "TotalSF"],
            "k_values": k_values
        },
        {
            "name": "Quality",
            "features": ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"],
            "k_values": k_values
        },
        {
            "name": "Location",
            "features": ["LotArea"],  # Don't use target-encoded features in clustering
            "k_values": [4, 6]
        },
        {
            "name": "Efficiency",
            "features": ["LotArea_to_TotalSF_Ratio", "Rooms_per_SF", "Bath_per_SF"] if "LotArea_to_TotalSF_Ratio" in train.columns else [],
            "k_values": [4, 6]
        },
        {
            "name": "Age_Quality",
            "features": ["Age", "OverallQual", "OverallCond", "RemodAge"] if "RemodAge" in train.columns else ["Age", "OverallQual", "OverallCond"],
            "k_values": [4, 6]
        },
        {
            "name": "Luxury",
            "features": ["HasPool", "HasGarage", "HasFireplace", "TotalBath", "GarageCars"],
            "k_values": [3, 5]
        },
        {
            "name": "Comprehensive",
            "features": ["GrLivArea", "OverallQual", "Age", "TotalBath", "GarageArea", "LotArea"],
            "k_values": [8, 10]
        }
    ]
    
    for config in cluster_configs:
        name = config["name"]
        features = [f for f in config["features"] if f in train.columns and f in test.columns]
        
        if len(features) < 2:
            continue
        
        for k in config["k_values"]:
            try:
                # Fit on train only to prevent leakage
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(train[features].values)
                kmeans = KMeans(n_clusters=k, n_init=20, random_state=random_state)
                train_labels = kmeans.fit_predict(X_train_scaled)
                
                # Transform test using train-fitted scaler and predict
                X_test_scaled = scaler.transform(test[features].values)
                test_labels = kmeans.predict(X_test_scaled)
                
                cluster_col = f"Cluster_{name}_k{k}"
                train[cluster_col] = train_labels
                test[cluster_col] = test_labels
                created_features.append(cluster_col)
            except Exception:
                pass
    
    return train, test, created_features


def add_interaction_features_advanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add more sophisticated interaction features."""
    created_features = []
    
    # Quality × Size interactions (more combinations)
    if "OverallQual" in df.columns:
        if "LotArea" in df.columns:
            df["Qual_x_LotArea"] = df["OverallQual"] * df["LotArea"]
            created_features.append("Qual_x_LotArea")
        if "GarageArea" in df.columns:
            df["Qual_x_GarageArea"] = df["OverallQual"] * df["GarageArea"]
            created_features.append("Qual_x_GarageArea")
        if "TotalBath" in df.columns:
            df["Qual_x_Bath"] = df["OverallQual"] * df["TotalBath"]
            created_features.append("Qual_x_Bath")
        if "GrLivArea" in df.columns:
            df["Qual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
            created_features.append("Qual_x_GrLivArea")
        if "TotalBsmtSF" in df.columns:
            df["Qual_x_BsmtSF"] = df["OverallQual"] * df["TotalBsmtSF"]
            created_features.append("Qual_x_BsmtSF")
    
    # Age × Quality interactions (maintenance effect)
    if "Age" in df.columns:
        if "OverallCond" in df.columns:
            df["Age_x_Condition"] = df["Age"] * df["OverallCond"]
            created_features.append("Age_x_Condition")
        if "OverallQual" in df.columns:
            df["Age_x_Quality"] = df["Age"] * df["OverallQual"]
            created_features.append("Age_x_Quality")
        if "TotalSF" in df.columns:
            df["Age_x_Size"] = df["Age"] * df["TotalSF"]
            created_features.append("Age_x_Size")
    
    # Location × Features (neighborhood premium)
    if "Neighborhood_mean_logSP" in df.columns:
        if "OverallQual" in df.columns:
            df["Neighborhood_x_Qual"] = df["Neighborhood_mean_logSP"] * df["OverallQual"]
            created_features.append("Neighborhood_x_Qual")
        if "TotalSF" in df.columns:
            df["Neighborhood_x_Size"] = df["Neighborhood_mean_logSP"] * df["TotalSF"]
            created_features.append("Neighborhood_x_Size")
        if "Age" in df.columns:
            df["Neighborhood_x_Age"] = df["Neighborhood_mean_logSP"] * df["Age"]
            created_features.append("Neighborhood_x_Age")
    
    # Size × Efficiency interactions
    if "TotalSF" in df.columns and "OverallQual" in df.columns:
        df["SF_x_Qual"] = df["TotalSF"] * df["OverallQual"]
        created_features.append("SF_x_Qual")
    
    if "GrLivArea" in df.columns and "TotalBath" in df.columns:
        df["Living_x_Bath"] = df["GrLivArea"] * df["TotalBath"]
        created_features.append("Living_x_Bath")
    
    # Room × Quality interactions
    if "TotRmsAbvGrd" in df.columns:
        if "OverallQual" in df.columns:
            df["Rooms_x_Qual"] = df["TotRmsAbvGrd"] * df["OverallQual"]
            created_features.append("Rooms_x_Qual")
        if "KitchenQual_Score" in df.columns:
            df["Rooms_x_KitchenQual"] = df["TotRmsAbvGrd"] * df["KitchenQual_Score"]
            created_features.append("Rooms_x_KitchenQual")
    
    # Garage × Quality interactions
    if "GarageArea" in df.columns and "GarageQual_Score" in df.columns:
        df["GarageArea_x_Qual"] = df["GarageArea"] * df["GarageQual_Score"]
        created_features.append("GarageArea_x_Qual")
    
    if "GarageCars" in df.columns and "OverallQual" in df.columns:
        df["GarageCars_x_Qual"] = df["GarageCars"] * df["OverallQual"]
        created_features.append("GarageCars_x_Qual")
    
    # Basement × Quality interactions
    if "TotalBsmtSF" in df.columns:
        if "BsmtQual_Score" in df.columns:
            df["BsmtSF_x_Qual"] = df["TotalBsmtSF"] * df["BsmtQual_Score"]
            created_features.append("BsmtSF_x_Qual")
        if "OverallQual" in df.columns:
            df["BsmtSF_x_OverallQual"] = df["TotalBsmtSF"] * df["OverallQual"]
            created_features.append("BsmtSF_x_OverallQual")
    
    # Year × Quality (newer high-quality vs older high-quality)
    if "YearBuilt" in df.columns and "OverallQual" in df.columns:
        df["YearBuilt_x_Qual"] = df["YearBuilt"] * df["OverallQual"]
        created_features.append("YearBuilt_x_Qual")
    
    if "YearRemodAdd" in df.columns and "OverallQual" in df.columns:
        df["YearRemod_x_Qual"] = df["YearRemodAdd"] * df["OverallQual"]
        created_features.append("YearRemod_x_Qual")
    
    # Bath × Quality interactions
    if "TotalBath" in df.columns:
        if "OverallQual" in df.columns:
            df["Bath_x_Qual"] = df["TotalBath"] * df["OverallQual"]
            created_features.append("Bath_x_Qual")
        if "KitchenQual_Score" in df.columns:
            df["Bath_x_KitchenQual"] = df["TotalBath"] * df["KitchenQual_Score"]
            created_features.append("Bath_x_KitchenQual")
    
    # Quality aggregate interactions
    if "AvgQuality" in df.columns:
        if "TotalSF" in df.columns:
            df["AvgQual_x_Size"] = df["AvgQuality"] * df["TotalSF"]
            created_features.append("AvgQual_x_Size")
        if "Age" in df.columns:
            df["AvgQual_x_Age"] = df["AvgQuality"] * df["Age"]
            created_features.append("AvgQual_x_Age")
    
    return df, created_features


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

    print("Running basic K-Means clustering...")
    train, test = add_kmeans_clusters(train, test)

    print("\nAdding advanced features...")
    
    # Neighborhood price statistics (target-encoded)
    print("  - Neighborhood price statistics...")
    train, test = add_neighborhood_price_stats(train, test, target_col="logSP")
    
    # Polynomial features
    print("  - Polynomial features...")
    train, poly_features = add_polynomial_features(train)
    test, _ = add_polynomial_features(test)
    
    # Ratio features
    print("  - Ratio features...")
    train, ratio_features = add_ratio_features(train)
    test, _ = add_ratio_features(test)
    
    # Temporal features
    print("  - Temporal features...")
    train, temporal_features = add_temporal_features(train)
    test, _ = add_temporal_features(test)
    
    # Quality aggregates (after ordinal encoding creates _Score columns)
    print("  - Quality aggregates...")
    train, quality_features = add_quality_aggregates(train)
    test, _ = add_quality_aggregates(test)
    
    # Advanced clustering
    print("  - Advanced clustering...")
    train, test, cluster_features = add_advanced_clustering(train, test, k_values=[6, 8])
    
    # Advanced interaction features
    print("  - Advanced interactions...")
    train, interaction_features = add_interaction_features_advanced(train)
    test, _ = add_interaction_features_advanced(test)

    # Log engineering details
    update_engineering_summary("Feature Engineering", {
        "kmeans_k": 4,
        "kmeans_cols": ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"],
        "luxury_flags": ["HasPool", "Has2ndFlr", "HasGarage", "HasBsmt", "HasFireplace"],
        "interactions": ["Qual_x_TotalSF", "Kitchen_x_TotalSF", "Cond_x_Age"],
        "benchmarks": ["Neighborhood", "MSSubClass", "MSZoning", "OverallQual", "Decade"],
        "neighborhood_stats": 5,
        "polynomial_features": len(poly_features),
        "ratio_features": len(ratio_features),
        "temporal_features": len(temporal_features),
        "quality_aggregates": len(quality_features),
        "advanced_clustering": len(cluster_features),
        "advanced_interactions": len(interaction_features)
    })

    print("Saving Processed 4 data...")
    Path(local_config.TRAIN_PROCESS4_CSV).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(local_config.TRAIN_PROCESS4_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS4_CSV, index=False)
    
    print("Feature Engineering complete!")


if __name__ == "__main__":
    main()
