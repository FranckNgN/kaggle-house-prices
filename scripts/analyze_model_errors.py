#!/usr/bin/env python
"""
Error Analysis Script for Model Improvement

Analyzes prediction errors to identify patterns and guide feature engineering.
Focuses on:
- Worst predictions (top 5% errors)
- Errors by Neighborhood
- Errors by OverallQual
- Errors by YearBuilt buckets
- Errors by RemodAge
- Interaction patterns
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from config_local import local_config, model_config


def load_model_predictions(model_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load model predictions and true values.
    
    Returns:
        (train_data, predictions_log_space)
    """
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    
    # Try to load OOF predictions
    oof_path = local_config.OOF_DIR / f"{model_name}_oof_train.npy"
    if oof_path.exists():
        predictions_log = np.load(oof_path)
        print(f"Loaded OOF predictions for {model_name}")
    else:
        print(f"Warning: OOF predictions not found for {model_name}")
        print("  You may need to run the model first to generate OOF predictions")
        return None, None
    
    return train, predictions_log


def calculate_errors(y_true: np.ndarray, y_pred_log: np.ndarray) -> pd.DataFrame:
    """
    Calculate errors in both log and real space.
    
    Returns:
        DataFrame with error metrics
    """
    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred_log)
    
    # Log space errors
    error_log = y_true - y_pred_log
    error_log_abs = np.abs(error_log)
    
    # Real space errors
    error_real = y_true_real - y_pred_real
    error_real_abs = np.abs(error_real)
    error_real_pct = (error_real / y_true_real) * 100
    
    errors_df = pd.DataFrame({
        'error_log': error_log,
        'error_log_abs': error_log_abs,
        'error_real': error_real,
        'error_real_abs': error_real_abs,
        'error_real_pct': error_real_pct,
        'y_true_log': y_true,
        'y_pred_log': y_pred_log,
        'y_true_real': y_true_real,
        'y_pred_real': y_pred_real,
    })
    
    return errors_df


def analyze_worst_predictions(
    train: pd.DataFrame,
    errors_df: pd.DataFrame,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Analyze the worst N predictions.
    """
    worst_idx = errors_df.nlargest(top_n, 'error_log_abs').index
    worst_data = train.iloc[worst_idx].copy()
    worst_errors = errors_df.iloc[worst_idx].copy()
    
    worst_analysis = pd.concat([worst_data, worst_errors], axis=1)
    
    print(f"\n{'='*70}")
    print(f"WORST {top_n} PREDICTIONS")
    print(f"{'='*70}")
    print(f"Mean absolute error (log): {worst_errors['error_log_abs'].mean():.4f}")
    print(f"Mean absolute error (real): ${worst_errors['error_real_abs'].mean():,.0f}")
    print(f"Mean percentage error: {worst_errors['error_real_pct'].abs().mean():.2f}%")
    
    return worst_analysis


def analyze_by_feature(
    train: pd.DataFrame,
    errors_df: pd.DataFrame,
    feature_name: str,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Analyze errors grouped by a feature.
    """
    if feature_name not in train.columns:
        print(f"Warning: Feature '{feature_name}' not found in training data")
        return None
    
    feature_values = train[feature_name].values
    
    # Handle categorical vs numeric
    feature_series = train[feature_name] if isinstance(train[feature_name], pd.Series) else pd.Series(train[feature_name])
    
    if feature_series.dtype == 'object' or feature_series.dtype.name == 'category':
        # Categorical: group by unique values
        groups = feature_series.unique()
        analysis = []
        for group in groups:
            mask = feature_series == group
            if mask.sum() > 0:
                group_errors = errors_df[mask]
                analysis.append({
                    'feature_value': group,
                    'count': mask.sum(),
                    'mean_error_log_abs': group_errors['error_log_abs'].mean(),
                    'mean_error_real_abs': group_errors['error_real_abs'].mean(),
                    'mean_error_pct': group_errors['error_real_pct'].abs().mean(),
                    'std_error_log_abs': group_errors['error_log_abs'].std(),
                })
    else:
        # Numeric: create bins
        unique_count = len(pd.Series(feature_values).unique())
        if n_bins > unique_count:
            n_bins = unique_count
        
        bins = pd.qcut(feature_values, q=n_bins, duplicates='drop')
        analysis = []
        for bin_val in bins.unique():
            if pd.isna(bin_val):
                continue
            mask = bins == bin_val
            if mask.sum() > 0:
                group_errors = errors_df[mask]
                analysis.append({
                    'feature_value': str(bin_val),
                    'count': mask.sum(),
                    'mean_error_log_abs': group_errors['error_log_abs'].mean(),
                    'mean_error_real_abs': group_errors['error_real_abs'].mean(),
                    'mean_error_pct': group_errors['error_real_pct'].abs().mean(),
                    'std_error_log_abs': group_errors['error_log_abs'].std(),
                })
    
    analysis_df = pd.DataFrame(analysis)
    if len(analysis_df) > 0:
        analysis_df = analysis_df.sort_values('mean_error_log_abs', ascending=False)
    
    return analysis_df


def suggest_features(worst_analysis: pd.DataFrame, feature_analyses: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Suggest new features based on error patterns.
    """
    suggestions = []
    
    # Check for new house patterns
    if 'YearBuilt' in worst_analysis.columns:
        new_house_mask = worst_analysis['YearBuilt'] > 2000
        if new_house_mask.sum() > len(worst_analysis) * 0.3:
            suggestions.append("Is_NewHouse = YearBuilt > 2000")
    
    # Check for quality-age interactions
    if 'OverallQual' in worst_analysis.columns and 'YearBuilt' in worst_analysis.columns:
        suggestions.append("Qual_Age_Interaction = OverallQual * (2024 - YearBuilt)")
    
    # Check for neighborhood patterns
    if 'Neighborhood' in feature_analyses:
        neigh_analysis = feature_analyses['Neighborhood']
        if len(neigh_analysis) > 0:
            high_error_neighborhoods = neigh_analysis.head(5)['feature_value'].tolist()
            suggestions.append(f"Neighborhood_HighError = Neighborhood in {high_error_neighborhoods}")
    
    # Check for remod age patterns
    if 'YearRemodAdd' in worst_analysis.columns and 'YearBuilt' in worst_analysis.columns:
        suggestions.append("RemodAge = YearRemodAdd - YearBuilt")
        suggestions.append("Is_Remodeled = (YearRemodAdd != YearBuilt)")
    
    # Check for quality patterns
    if 'OverallQual' in feature_analyses:
        qual_analysis = feature_analyses['OverallQual']
        if len(qual_analysis) > 0:
            suggestions.append("OverallQual_Squared = OverallQual ** 2")
    
    return suggestions


def main(model_name: str = "catboost"):
    """
    Main error analysis function.
    """
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS FOR {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    train, predictions_log = load_model_predictions(model_name)
    if train is None:
        return
    
    y_true = train["logSP"].values
    
    # Calculate errors
    errors_df = calculate_errors(y_true, predictions_log)
    
    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL ERROR STATISTICS")
    print(f"{'='*70}")
    print(f"Mean absolute error (log): {errors_df['error_log_abs'].mean():.4f}")
    print(f"Mean absolute error (real): ${errors_df['error_real_abs'].mean():,.0f}")
    print(f"Mean percentage error: {errors_df['error_real_pct'].abs().mean():.2f}%")
    print(f"RMSE (log space): {np.sqrt(np.mean(errors_df['error_log']**2)):.4f}")
    
    # Analyze worst predictions
    worst_analysis = analyze_worst_predictions(train, errors_df, top_n=50)
    
    # Analyze by key features
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS BY FEATURE")
    print(f"{'='*70}")
    
    feature_analyses = {}
    key_features = ['Neighborhood', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 
                    'GrLivArea', 'TotalSF', 'GarageArea', 'LotArea']
    
    for feature in key_features:
        if feature in train.columns:
            analysis = analyze_by_feature(train, errors_df, feature)
            if analysis is not None and len(analysis) > 0:
                feature_analyses[feature] = analysis
                print(f"\n{feature}:")
                print(analysis.head(10).to_string(index=False))
    
    # Suggest new features
    print(f"\n{'='*70}")
    print("SUGGESTED NEW FEATURES")
    print(f"{'='*70}")
    suggestions = suggest_features(worst_analysis, feature_analyses)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    if not suggestions:
        print("No automatic suggestions. Review worst predictions manually.")
    
    # Save analysis
    output_dir = Path("runs") / "error_analysis"
    output_dir.mkdir(exist_ok=True)
    
    worst_analysis.to_csv(output_dir / f"{model_name}_worst_predictions.csv", index=False)
    
    for feature, analysis in feature_analyses.items():
        analysis.to_csv(output_dir / f"{model_name}_errors_by_{feature}.csv", index=False)
    
    print(f"\n{'='*70}")
    print(f"Analysis saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "catboost"
    main(model_name)

