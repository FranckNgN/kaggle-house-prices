#!/usr/bin/env python
"""
Quick model comparison script that generates a text report comparing all models.
Enhanced version with proper model name matching to performance metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_local import local_config as config
from scripts.compare_models import load_all_submissions


def normalize_model_name(submission_name):
    """Normalize submission folder/file names to match model names in performance CSV."""
    # Remove numbers and underscores, convert to lowercase
    name = submission_name.lower()
    
    # Handle numbered prefixes like "7_xgboost" -> "xgboost"
    if '_' in name:
        parts = name.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            name = '_'.join(parts[1:])
    
    # Map common variations
    name_mapping = {
        'xgboost': 'xgboost',
        'xgb': 'xgboost',
        'lightgbm': 'lightgbm',
        'lgb': 'lightgbm',
        'lightgb': 'lightgbm',
        'catboost': 'catboost',
        'cat': 'catboost',
        'ridge': 'ridge',
        'lasso': 'lasso',
        'elasticnet': 'elastic_net',
        'elastic_net': 'elastic_net',
        'randomforest': 'random_forest',
        'random_forest': 'random_forest',
        'rf': 'random_forest',
        'svr': 'svr',
        'blending': 'blending',
        'blend': 'blending',
        'stacking': 'STACKING_META',
        'stack': 'STACKING_META',
        'linearregression': 'linear_regression',
        'linear_regression': 'linear_regression',
    }
    
    # Try exact match first
    if name in name_mapping:
        return name_mapping[name]
    
    # Try partial match
    for key, value in name_mapping.items():
        if key in name:
            return value
    
    return name


def main():
    print("="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)
    print()
    
    # Load predictions
    print("Loading model predictions from submission files...")
    df_predictions = load_all_submissions(config.SUBMISSIONS_DIR)
    
    # Load performance metrics
    print("Loading performance metrics...")
    df_performance = pd.read_csv(config.MODEL_PERFORMANCE_CSV)
    df_performance['timestamp'] = pd.to_datetime(df_performance['timestamp'])
    
    # Get latest entry for each model
    latest_performance = df_performance.sort_values('timestamp').groupby('model').last().reset_index()
    
    # Create a lookup dictionary
    perf_lookup = {}
    for _, row in latest_performance.iterrows():
        perf_lookup[row['model'].lower()] = {
            'rmse': row['rmse'],
            'kaggle_score': row['kaggle_score'] if pd.notna(row['kaggle_score']) else None,
            'timestamp': row['timestamp'],
            'notes': row['notes']
        }
    
    # Filter valid models
    valid_models = [col for col in df_predictions.columns if df_predictions[col].mean() < 1e7]
    invalid_models = [col for col in df_predictions.columns if col not in valid_models]
    
    print(f"\nFound {len(df_predictions.columns)} total submission files")
    print(f"  Valid models: {len(valid_models)}")
    print(f"  Invalid models (exploded): {len(invalid_models)}")
    
    if invalid_models:
        print(f"\nWARNING: Invalid models (excluded from comparison):")
        for model in invalid_models:
            print(f"    - {model}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL STATISTICS")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for model_name in valid_models:
        preds = df_predictions[model_name]
        
        # Normalize model name and find matching performance data
        normalized_name = normalize_model_name(model_name)
        perf_data = perf_lookup.get(normalized_name, {})
        
        rmse = perf_data.get('rmse')
        kaggle_score = perf_data.get('kaggle_score')
        notes = perf_data.get('notes', '')
        
        comparison_data.append({
            'Model': model_name,
            'Mean ($)': f"${preds.mean():,.0f}",
            'Median ($)': f"${preds.median():,.0f}",
            'Std ($)': f"${preds.std():,.0f}",
            'Min ($)': f"${preds.min():,.0f}",
            'Max ($)': f"${preds.max():,.0f}",
            'CV RMSE': f"{rmse:.6f}" if rmse else "N/A",
            'Kaggle Score': f"{kaggle_score:.5f}" if kaggle_score else "N/A",
            'Notes': notes[:30] + '...' if len(notes) > 30 else notes
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Ranking by CV RMSE
    print("\n" + "="*80)
    print("MODEL RANKINGS (by CV RMSE - lower is better)")
    print("="*80)
    
    df_ranking_rmse = df_comparison.copy()
    df_ranking_rmse['RMSE_num'] = df_ranking_rmse['CV RMSE'].replace('N/A', np.nan).astype(float)
    df_ranking_rmse = df_ranking_rmse[df_ranking_rmse['RMSE_num'].notna()].sort_values('RMSE_num')
    
    if len(df_ranking_rmse) > 0:
        for idx, (_, row) in enumerate(df_ranking_rmse.iterrows(), 1):
            print(f"{idx:2d}. {row['Model']:30s} - CV RMSE: {row['CV RMSE']:>10s} | Kaggle: {row['Kaggle Score']:>10s}")
    else:
        print("  No CV RMSE data available")
    
    # Ranking by Kaggle Score
    print("\n" + "="*80)
    print("MODEL RANKINGS (by Kaggle Score - lower is better)")
    print("="*80)
    
    df_ranking_kaggle = df_comparison.copy()
    df_ranking_kaggle['Kaggle_num'] = df_ranking_kaggle['Kaggle Score'].replace('N/A', np.nan).astype(float)
    df_ranking_kaggle = df_ranking_kaggle[df_ranking_kaggle['Kaggle_num'].notna()].sort_values('Kaggle_num')
    
    if len(df_ranking_kaggle) > 0:
        for idx, (_, row) in enumerate(df_ranking_kaggle.iterrows(), 1):
            print(f"{idx:2d}. {row['Model']:30s} - Kaggle: {row['Kaggle Score']:>10s} | CV RMSE: {row['CV RMSE']:>10s}")
    else:
        print("  No Kaggle score data available")
    
    # Correlation analysis
    print("\n" + "="*80)
    print("MODEL PREDICTION CORRELATIONS")
    print("="*80)
    df_valid = df_predictions[valid_models]
    corr_matrix = df_valid.corr()
    
    # Find highly correlated pairs
    print("\nHighly correlated model pairs (correlation > 0.95):")
    high_corr_pairs = []
    for i, model1 in enumerate(valid_models):
        for model2 in valid_models[i+1:]:
            corr = corr_matrix.loc[model1, model2]
            if corr > 0.95:
                high_corr_pairs.append((model1, model2, corr))
    
    if high_corr_pairs:
        for model1, model2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"  {model1:25s} <-> {model2:25s}: {corr:.4f}")
    else:
        print("  None found")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total valid models: {len(valid_models)}")
    
    if len(df_ranking_rmse) > 0:
        best_rmse = df_ranking_rmse.iloc[0]
        print(f"Best model by CV RMSE: {best_rmse['Model']} (RMSE: {best_rmse['CV RMSE']})")
    
    if len(df_ranking_kaggle) > 0:
        best_kaggle = df_ranking_kaggle.iloc[0]
        print(f"Best model by Kaggle Score: {best_kaggle['Model']} (Score: {best_kaggle['Kaggle Score']})")
        print(f"\n*** WINNER: {best_kaggle['Model']} with Kaggle Score {best_kaggle['Kaggle Score']} ***")
    
    print("\nFor detailed visualizations, run: python scripts/compare_models.py")
    print("="*80)


if __name__ == "__main__":
    main()
