#!/usr/bin/env python
"""
Quick model comparison script that generates a text report comparing all models.
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

def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*80)
    print()
    
    # Load predictions
    print("Loading model predictions...")
    df_predictions = load_all_submissions(config.SUBMISSIONS_DIR)
    
    # Load performance metrics
    df_performance = pd.read_csv(config.MODEL_PERFORMANCE_CSV)
    df_performance['timestamp'] = pd.to_datetime(df_performance['timestamp'])
    latest_performance = df_performance.sort_values('timestamp').groupby('model').last().reset_index()
    
    # Filter valid models
    valid_models = [col for col in df_predictions.columns if df_predictions[col].mean() < 1e7]
    invalid_models = [col for col in df_predictions.columns if col not in valid_models]
    
    print(f"\nFound {len(df_predictions.columns)} total models")
    print(f"  Valid models: {len(valid_models)}")
    print(f"  Invalid models (exploded): {len(invalid_models)}")
    
    if invalid_models:
        print(f"\nWARNING: Invalid models (excluded from comparison):")
        for model in invalid_models:
            print(f"    - {model}")
    
    print("\n" + "="*80)
    print("MODEL STATISTICS")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for model_name in valid_models:
        preds = df_predictions[model_name]
        
        # Find matching performance data
        model_key = model_name.split('_')[0].lower()
        perf_match = latest_performance[
            latest_performance['model'].str.lower().str.contains(model_key, na=False)
        ]
        
        rmse = None
        kaggle_score = None
        if len(perf_match) > 0:
            rmse = perf_match.iloc[0]['rmse']
            kaggle_score = perf_match.iloc[0]['kaggle_score'] if pd.notna(perf_match.iloc[0]['kaggle_score']) else None
        
        comparison_data.append({
            'Model': model_name,
            'Mean ($)': f"${preds.mean():,.0f}",
            'Median ($)': f"${preds.median():,.0f}",
            'Std ($)': f"${preds.std():,.0f}",
            'Min ($)': f"${preds.min():,.0f}",
            'Max ($)': f"${preds.max():,.0f}",
            'RMSE': f"{rmse:.6f}" if rmse else "N/A",
            'Kaggle': f"{kaggle_score:.5f}" if kaggle_score else "N/A"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Correlation analysis
    print("\n" + "="*80)
    print("MODEL CORRELATIONS")
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
            print(f"  {model1} <-> {model2}: {corr:.4f}")
    else:
        print("  None found")
    
    # Ranking by RMSE
    print("\n" + "="*80)
    print("MODEL RANKINGS (by RMSE - lower is better)")
    print("="*80)
    
    df_ranking = df_comparison.copy()
    df_ranking['RMSE_num'] = df_ranking['RMSE'].replace('N/A', np.nan).astype(float)
    df_ranking = df_ranking[df_ranking['RMSE_num'].notna()].sort_values('RMSE_num')
    
    if len(df_ranking) > 0:
        for idx, row in df_ranking.iterrows():
            rank = list(df_ranking.index).index(idx) + 1
            print(f"{rank:2d}. {row['Model']:30s} - RMSE: {row['RMSE']}")
    else:
        print("  No RMSE data available")
    
    # Ranking by Kaggle score
    print("\n" + "="*80)
    print("MODEL RANKINGS (by Kaggle Score - lower is better)")
    print("="*80)
    
    df_ranking_kaggle = df_comparison.copy()
    df_ranking_kaggle['Kaggle_num'] = df_ranking_kaggle['Kaggle'].replace('N/A', np.nan).astype(float)
    df_ranking_kaggle = df_ranking_kaggle[df_ranking_kaggle['Kaggle_num'].notna()].sort_values('Kaggle_num')
    
    if len(df_ranking_kaggle) > 0:
        for idx, row in df_ranking_kaggle.iterrows():
            rank = list(df_ranking_kaggle.index).index(idx) + 1
            print(f"{rank:2d}. {row['Model']:30s} - Score: {row['Kaggle']}")
    else:
        print("  No Kaggle score data available")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total models compared: {len(valid_models)}")
    print(f"Best model by RMSE: {df_ranking.iloc[0]['Model'] if len(df_ranking) > 0 else 'N/A'}")
    print(f"Best model by Kaggle: {df_ranking_kaggle.iloc[0]['Model'] if len(df_ranking_kaggle) > 0 else 'N/A'}")
    print("\nFor detailed visualizations, run: python scripts/compare_models.py")
    print("Or open notebooks/Journal.ipynb and run the comparison cells")
    print("="*80)

if __name__ == "__main__":
    main()

