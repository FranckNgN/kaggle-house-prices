#!/usr/bin/env python
"""
Analyze model performance log to find the best models.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config_local.local_config as config


def analyze_models():
    """Analyze model performance and find best models."""
    log_file = config.MODEL_PERFORMANCE_CSV
    
    if not log_file.exists():
        print(f"[ERROR] Model performance log not found: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Filter out entries with invalid RMSE (too high, likely errors)
    df_valid = df[df['rmse'] < 1.0].copy()
    
    # Convert kaggle_score to numeric, handling empty strings
    df_valid['kaggle_score'] = pd.to_numeric(df_valid['kaggle_score'], errors='coerce')
    
    print(f"\nTotal model runs: {len(df)}")
    print(f"Valid model runs (RMSE < 1.0): {len(df_valid)}")
    print(f"Models with Kaggle scores: {df_valid['kaggle_score'].notna().sum()}")
    
    # Best by CV RMSE
    print("\n" + "=" * 80)
    print("BEST MODELS BY CV RMSE (Cross-Validation)")
    print("=" * 80)
    best_cv = df_valid.nsmallest(10, 'rmse')[
        ['timestamp', 'model', 'rmse', 'kaggle_score', 'notes', 'runtime']
    ]
    print(best_cv.to_string(index=False))
    
    # Best by Kaggle Score
    print("\n" + "=" * 80)
    print("BEST MODELS BY KAGGLE SCORE (Leaderboard)")
    print("=" * 80)
    df_with_kaggle = df_valid[df_valid['kaggle_score'].notna()].copy()
    if len(df_with_kaggle) > 0:
        best_kaggle = df_with_kaggle.nsmallest(10, 'kaggle_score')[
            ['timestamp', 'model', 'rmse', 'kaggle_score', 'notes', 'runtime']
        ]
        print(best_kaggle.to_string(index=False))
        
        # Overall best
        overall_best = df_with_kaggle.nsmallest(1, 'kaggle_score').iloc[0]
        print("\n" + "-" * 80)
        print("*** OVERALL BEST MODEL (Lowest Kaggle Score) ***")
        print("-" * 80)
        print(f"Model: {overall_best['model']}")
        print(f"Kaggle Score: {overall_best['kaggle_score']:.6f} (RMSLE)")
        print(f"CV RMSE: {overall_best['rmse']:.6f}")
        print(f"Timestamp: {overall_best['timestamp']}")
        print(f"Notes: {overall_best['notes']}")
        if overall_best['runtime']:
            print(f"Runtime: {overall_best['runtime']}")
    else:
        print("No models with Kaggle scores yet.")
    
    # Model comparison by model type
    print("\n" + "=" * 80)
    print("MODEL TYPE COMPARISON")
    print("=" * 80)
    
    # Get best score for each model type
    model_summary = []
    for model_name in df_valid['model'].unique():
        model_df = df_valid[df_valid['model'] == model_name]
        best_cv_entry = model_df.nsmallest(1, 'rmse').iloc[0]
        
        # Get best Kaggle score if available
        model_kaggle = model_df[model_df['kaggle_score'].notna()]
        best_kaggle_score = None
        if len(model_kaggle) > 0:
            best_kaggle_entry = model_kaggle.nsmallest(1, 'kaggle_score').iloc[0]
            best_kaggle_score = best_kaggle_entry['kaggle_score']
        
        model_summary.append({
            'model': model_name,
            'best_cv_rmse': best_cv_entry['rmse'],
            'best_kaggle_score': best_kaggle_score,
            'runs': len(model_df),
            'runs_with_kaggle': len(model_kaggle)
        })
    
    summary_df = pd.DataFrame(model_summary)
    summary_df = summary_df.sort_values('best_cv_rmse')
    print(summary_df.to_string(index=False))
    
    # Correlation between CV RMSE and Kaggle Score
    print("\n" + "=" * 80)
    print("CV RMSE vs KAGGLE SCORE CORRELATION")
    print("=" * 80)
    df_correlation = df_valid[df_valid['kaggle_score'].notna()].copy()
    if len(df_correlation) > 1:
        correlation = df_correlation['rmse'].corr(df_correlation['kaggle_score'])
        print(f"Correlation coefficient: {correlation:.4f}")
        print("\nModels with both CV RMSE and Kaggle scores:")
        comparison = df_correlation[['model', 'rmse', 'kaggle_score', 'timestamp']].sort_values('kaggle_score')
        print(comparison.to_string(index=False))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_models()

