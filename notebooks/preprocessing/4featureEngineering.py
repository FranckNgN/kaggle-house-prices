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


# NOTE: add_neighborhood_price_stats moved to stage 8 (target encoding)
# to ensure proper order: feature selection -> target encoding
# This prevents data leakage and ensures target encoding happens on selected features


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
    """Add temporal features for year/month effects and market conditions."""
    created_features = []
    
    if "YrSold" in df.columns:
        df["YearsSince2006"] = df["YrSold"] - 2006
        created_features.append("YearsSince2006")
        df["MarketCycle"] = (df["YrSold"] - 2006) % 4
        created_features.append("MarketCycle")
        # Market trend indicator (pre/post 2010 housing crisis recovery)
        df["PostCrisis"] = (df["YrSold"] >= 2010).astype(int)
        created_features.append("PostCrisis")
    
    if "MoSold" in df.columns:
        df["Quarter"] = ((df["MoSold"] - 1) // 3) + 1
        created_features.append("Quarter")
        df["PeakSeason"] = ((df["MoSold"] >= 3) & (df["MoSold"] <= 8)).astype(int)
        created_features.append("PeakSeason")
        df["EndOfYear"] = ((df["MoSold"] >= 11) | (df["MoSold"] <= 1)).astype(int)
        created_features.append("EndOfYear")
        # Month as cyclical feature (sine/cosine encoding)
        df["MonthSin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
        df["MonthCos"] = np.cos(2 * np.pi * df["MoSold"] / 12)
        created_features.append("MonthSin")
        created_features.append("MonthCos")
    
    # Remodel timing interactions with temporal features
    if "RemodAge" in df.columns and "YrSold" in df.columns:
        # Years since remodel at time of sale
        df["RemodAge_AtSale"] = df["RemodAge"]
        # Interaction: Remodel age × Market cycle
        if "MarketCycle" in df.columns:
            df["RemodAge_x_MarketCycle"] = df["RemodAge"] * df["MarketCycle"]
            created_features.append("RemodAge_x_MarketCycle")
    
    # House age categories (Decade bins) - already created in add_era_features as "Decade"
    # But add interactions with temporal features
    if "Decade" in df.columns and "YrSold" in df.columns:
        # Age of house at time of sale (relative to decade built)
        if "YearBuilt" in df.columns:
            df["Age_AtSale"] = df["YrSold"] - df["YearBuilt"]
            # Interaction: Decade × Age at sale
            df["Decade_x_AgeAtSale"] = df["Decade"] * df["Age_AtSale"]
            created_features.append("Decade_x_AgeAtSale")
    
    # AgeAtSale removed - redundant duplicate of Age column
    
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
        # NOTE: Age_x_Condition removed - duplicate of Cond_x_Age (created earlier)
        # Both are OverallCond * Age (multiplication is commutative)
        if "OverallQual" in df.columns:
            df["Age_x_Quality"] = df["Age"] * df["OverallQual"]
            created_features.append("Age_x_Quality")
        if "TotalSF" in df.columns:
            df["Age_x_Size"] = df["Age"] * df["TotalSF"]
            created_features.append("Age_x_Size")
    
    # Location × Features (neighborhood premium)
    # NOTE: Neighborhood_mean_logSP doesn't exist until Stage 8 (target encoding)
    # These interaction features should be created in Stage 8 after target encoding
    # Removed to avoid dead code that never executes
    
    # Size × Efficiency interactions
    # NOTE: SF_x_Qual removed - duplicate of Qual_x_TotalSF (created earlier)
    # Both are OverallQual * TotalSF (multiplication is commutative)
    
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
    
    # ========================================================================
    # THREE-WAY INTERACTIONS (Phase 2.3: Advanced Interaction Features)
    # ========================================================================
    
    # Quality × Size × Age: Captures how quality value changes with size and age
    if "OverallQual" in df.columns and "TotalSF" in df.columns and "Age" in df.columns:
        df["Qual_x_Size_x_Age"] = df["OverallQual"] * df["TotalSF"] * df["Age"]
        created_features.append("Qual_x_Size_x_Age")
    
    # Location × Quality: Neighborhood × OverallQual (if Neighborhood is numeric/encoded)
    # Note: If Neighborhood is still categorical, this will be handled in target encoding stage
    if "Neighborhood_QualityScore" in df.columns and "OverallQual" in df.columns:
        df["Neighborhood_x_Quality"] = df["Neighborhood_QualityScore"] * df["OverallQual"]
        created_features.append("Neighborhood_x_Quality")
    
    # Condition × Age × Quality: Multi-factor interactions
    if "OverallCond" in df.columns and "Age" in df.columns and "OverallQual" in df.columns:
        df["Cond_x_Age_x_Qual"] = df["OverallCond"] * df["Age"] * df["OverallQual"]
        created_features.append("Cond_x_Age_x_Qual")
    
    # Bathroom × Bedroom Ratios: Efficiency metrics
    if "TotalBath" in df.columns and "BedroomAbvGr" in df.columns:
        df["Bath_per_Bedroom"] = df["TotalBath"] / (df["BedroomAbvGr"] + 1)
        created_features.append("Bath_per_Bedroom")
    if "GrLivArea" in df.columns and "BedroomAbvGr" in df.columns:
        df["LivingArea_per_Bedroom"] = df["GrLivArea"] / (df["BedroomAbvGr"] + 1)
        created_features.append("LivingArea_per_Bedroom")
    
    # Garage × Lot Interactions: Parking value by lot size
    if "GarageArea" in df.columns and "LotArea" in df.columns:
        df["Garage_x_Lot"] = df["GarageArea"] * df["LotArea"]
        df["Garage_to_Lot_Ratio"] = df["GarageArea"] / (df["LotArea"] + 1)
        created_features.append("Garage_x_Lot")
        created_features.append("Garage_to_Lot_Ratio")
    
    # Basement × Above-Ground: Ratio and interaction features
    if "TotalBsmtSF" in df.columns and "GrLivArea" in df.columns:
        df["Bsmt_x_AboveGround"] = df["TotalBsmtSF"] * df["GrLivArea"]
        df["Bsmt_to_AboveGround_Ratio"] = df["TotalBsmtSF"] / (df["GrLivArea"] + 1)
        created_features.append("Bsmt_x_AboveGround")
        created_features.append("Bsmt_to_AboveGround_Ratio")
    
    # ========================================================================
    # ERROR-DRIVEN FEATURES (Added 2025-12-20 based on error analysis)
    # These features address specific failure patterns identified in worst predictions
    # ========================================================================
    
    # 1. Qual_Age_Interaction: Addresses high errors in old houses (YearBuilt < 1960: 14.67% error)
    #    and new houses (YearBuilt > 2005: 9.69% error)
    if "OverallQual" in df.columns and "YearBuilt" in df.columns:
        # Use YrSold per row if available (more accurate), otherwise use 2024 as reference
        if "YrSold" in df.columns:
            df["Qual_Age_Interaction"] = df["OverallQual"] * (df["YrSold"] - df["YearBuilt"])
        else:
            df["Qual_Age_Interaction"] = df["OverallQual"] * (2024 - df["YearBuilt"])
        created_features.append("Qual_Age_Interaction")
    
    # 2. RemodAge: Age of remodel relative to build year (addresses remodel patterns)
    #    Note: RemodAge already exists as YrSold - YearRemodAdd, but this is different
    if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
        df["RemodAge_FromBuild"] = df["YearRemodAdd"] - df["YearBuilt"]
        created_features.append("RemodAge_FromBuild")
        # Also create binary flag
        df["Is_Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
        created_features.append("Is_Remodeled")
    
    # 3. OverallQual_Squared: Addresses non-linear quality effects (low quality: 9.88% error)
    if "OverallQual" in df.columns:
        df["OverallQual_Squared"] = df["OverallQual"] ** 2
        created_features.append("OverallQual_Squared")
    
    return df, created_features


def add_neighborhood_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Add neighborhood-level features that don't require target encoding.
    These features capture neighborhood characteristics without using SalePrice.
    """
    created_features = []
    
    # Neighborhood Quality Score: Average OverallQual by Neighborhood
    if "Neighborhood" in train.columns and "OverallQual" in train.columns:
        neighborhood_quality = train.groupby("Neighborhood")["OverallQual"].mean()
        train["Neighborhood_QualityScore"] = train["Neighborhood"].map(neighborhood_quality)
        test["Neighborhood_QualityScore"] = test["Neighborhood"].map(neighborhood_quality).fillna(
            train["OverallQual"].mean()
        )
        created_features.append("Neighborhood_QualityScore")
    
    # Neighborhood Age Profile: Average YearBuilt by Neighborhood
    if "Neighborhood" in train.columns and "YearBuilt" in train.columns:
        neighborhood_age = train.groupby("Neighborhood")["YearBuilt"].mean()
        train["Neighborhood_AvgYearBuilt"] = train["Neighborhood"].map(neighborhood_age)
        test["Neighborhood_AvgYearBuilt"] = test["Neighborhood"].map(neighborhood_age).fillna(
            train["YearBuilt"].mean()
        )
        created_features.append("Neighborhood_AvgYearBuilt")
    
    # Neighborhood-LotArea Interaction: Average LotArea by Neighborhood
    if "Neighborhood" in train.columns and "LotArea" in train.columns:
        neighborhood_lot = train.groupby("Neighborhood")["LotArea"].mean()
        train["Neighborhood_AvgLotArea"] = train["Neighborhood"].map(neighborhood_lot)
        test["Neighborhood_AvgLotArea"] = test["Neighborhood"].map(neighborhood_lot).fillna(
            train["LotArea"].mean()
        )
        # Create interaction: how does this house's lot compare to neighborhood average
        train["LotArea_vs_Neighborhood"] = train["LotArea"] / (train["Neighborhood_AvgLotArea"] + 1)
        test["LotArea_vs_Neighborhood"] = test["LotArea"] / (test["Neighborhood_AvgLotArea"] + 1)
        created_features.append("Neighborhood_AvgLotArea")
        created_features.append("LotArea_vs_Neighborhood")
    
    # Neighborhood-Garage Interaction: Garage presence/value by neighborhood
    if "Neighborhood" in train.columns and "GarageArea" in train.columns:
        neighborhood_garage = train.groupby("Neighborhood")["GarageArea"].mean()
        train["Neighborhood_AvgGarageArea"] = train["Neighborhood"].map(neighborhood_garage)
        test["Neighborhood_AvgGarageArea"] = test["Neighborhood"].map(neighborhood_garage).fillna(
            train["GarageArea"].mean()
        )
        # Create interaction
        train["GarageArea_vs_Neighborhood"] = train["GarageArea"] / (train["Neighborhood_AvgGarageArea"] + 1)
        test["GarageArea_vs_Neighborhood"] = test["GarageArea"] / (test["Neighborhood_AvgGarageArea"] + 1)
        created_features.append("Neighborhood_AvgGarageArea")
        created_features.append("GarageArea_vs_Neighborhood")
    
    return train, test, created_features


def add_domain_specific_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add domain-specific features based on real estate valuation principles.
    These features capture luxury indicators, efficiency metrics, and property characteristics.
    """
    created_features = []
    
    # Luxury Indicators: Combinations of premium features
    luxury_flags = []
    if "HasPool" in df.columns:
        luxury_flags.append("HasPool")
    if "HasFireplace" in df.columns:
        luxury_flags.append("HasFireplace")
    if "HasGarage" in df.columns:
        luxury_flags.append("HasGarage")
    if "Has2ndFlr" in df.columns:
        luxury_flags.append("Has2ndFlr")
    
    if len(luxury_flags) > 0:
        df["LuxuryFeatureCount"] = df[luxury_flags].sum(axis=1)
        created_features.append("LuxuryFeatureCount")
        # Premium combination: Pool + Fireplace + Garage + 2nd Floor
        if all(flag in df.columns for flag in ["HasPool", "HasFireplace", "HasGarage", "Has2ndFlr"]):
            df["PremiumLuxury"] = (
                df["HasPool"] * df["HasFireplace"] * df["HasGarage"] * df["Has2ndFlr"]
            ).astype(int)
            created_features.append("PremiumLuxury")
    
    # Efficiency Metrics: Living area per bedroom, lot utilization
    if "GrLivArea" in df.columns and "BedroomAbvGr" in df.columns:
        df["LivingArea_per_Bedroom"] = df["GrLivArea"] / (df["BedroomAbvGr"] + 1)
        created_features.append("LivingArea_per_Bedroom")
    
    if "TotalSF" in df.columns and "LotArea" in df.columns:
        df["LotUtilization"] = df["TotalSF"] / (df["LotArea"] + 1)
        created_features.append("LotUtilization")
    
    # Condition Scores: Weighted average of all condition/quality ratings
    quality_cols = [col for col in df.columns if "_Score" in col or col in ["OverallQual", "OverallCond"]]
    if len(quality_cols) > 0:
        # Weighted average (OverallQual and OverallCond get higher weight)
        weights = {}
        for col in quality_cols:
            if col == "OverallQual":
                weights[col] = 2.0
            elif col == "OverallCond":
                weights[col] = 1.5
            else:
                weights[col] = 1.0
        
        weighted_sum = sum(df[col] * weights.get(col, 1.0) for col in quality_cols if col in df.columns)
        weight_sum = sum(weights.get(col, 1.0) for col in quality_cols if col in df.columns)
        df["WeightedQualityScore"] = weighted_sum / (weight_sum + 1e-10)
        created_features.append("WeightedQualityScore")
    
    # Renovation Indicators: Remodeled + Quality improvements
    if "Is_Remodeled" in df.columns and "OverallQual" in df.columns and "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
        # Quality improvement: Did quality increase after remodel?
        # This is a proxy - we can't directly measure, but newer remodels might have higher quality
        df["RecentRemodel"] = (df["Is_Remodeled"] == 1) & (df["YearRemodAdd"] >= 2000).astype(int)
        created_features.append("RecentRemodel")
    
    # Architectural Style: MSSubClass interactions with other features
    if "MSSubClass" in df.columns:
        if "TotalSF" in df.columns:
            df["MSSubClass_x_TotalSF"] = df["MSSubClass"] * df["TotalSF"]
            created_features.append("MSSubClass_x_TotalSF")
        if "OverallQual" in df.columns:
            df["MSSubClass_x_Quality"] = df["MSSubClass"] * df["OverallQual"]
            created_features.append("MSSubClass_x_Quality")
    
    # Zoning Effects: MSZoning × Neighborhood interactions
    # Note: If MSZoning is still categorical, this will be handled after encoding
    # For now, we'll create numeric interactions if MSZoning has been encoded
    if "MSZoning_TargetEnc" in df.columns and "Neighborhood_QualityScore" in df.columns:
        df["MSZoning_x_Neighborhood"] = df["MSZoning_TargetEnc"] * df["Neighborhood_QualityScore"]
        created_features.append("MSZoning_x_Neighborhood")
    
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
    
    # NOTE: Neighborhood price statistics moved to stage 8 (target encoding)
    # to prevent data leakage and ensure proper order
    
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
    
    # Neighborhood-level features (non-target-encoded)
    print("  - Neighborhood-level features...")
    train, test, neighborhood_features = add_neighborhood_features(train, test)
    
    # Domain-specific features
    print("  - Domain-specific features...")
    train, domain_features = add_domain_specific_features(train)
    test, _ = add_domain_specific_features(test)

    # Log engineering details
    update_engineering_summary("Feature Engineering", {
        "kmeans_k": 4,
        "kmeans_cols": ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"],
        "luxury_flags": ["HasPool", "Has2ndFlr", "HasGarage", "HasBsmt", "HasFireplace"],
        "interactions": ["Qual_x_TotalSF", "Kitchen_x_TotalSF", "Cond_x_Age"],
        "benchmarks": ["Neighborhood", "MSSubClass", "MSZoning", "OverallQual", "Decade"],
        "note": "Neighborhood price stats moved to stage 8 (target encoding) to prevent leakage",
        "polynomial_features": len(poly_features),
        "ratio_features": len(ratio_features),
        "temporal_features": len(temporal_features),
        "quality_aggregates": len(quality_features),
        "advanced_clustering": len(cluster_features),
        "advanced_interactions": len(interaction_features),
        "neighborhood_features": len(neighborhood_features),
        "domain_specific_features": len(domain_features)
    })

    print("Saving Processed 4 data...")
    Path(local_config.TRAIN_PROCESS4_CSV).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(local_config.TRAIN_PROCESS4_CSV, index=False)
    test.to_csv(local_config.TEST_PROCESS4_CSV, index=False)
    
    print("Feature Engineering complete!")


if __name__ == "__main__":
    main()
