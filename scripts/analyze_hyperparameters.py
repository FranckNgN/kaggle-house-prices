#!/usr/bin/env python
"""
Analyze hyperparameters and their relationship to model performance.
"""

import sys
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config_local.local_config as config

def analyze_hyperparameters():
    """Analyze hyperparameters for models with Kaggle scores."""
    log_file = config.MODEL_PERFORMANCE_CSV
    
    if not log_file.exists():
        print(f"[ERROR] Model performance log not found: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    df['kaggle_score'] = pd.to_numeric(df['kaggle_score'], errors='coerce')
    df_valid = df[(df['rmse'] < 1.0) & (df['kaggle_score'].notna())].copy()
    
    print("=" * 80)
    print("HYPERPARAMETER ANALYSIS FOR MODELS WITH KAGGLE SCORES")
    print("=" * 80)
    
    # Group by model type
    for model_name in df_valid['model'].unique():
        model_df = df_valid[df_valid['model'] == model_name].copy()
        if len(model_df) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Sort by Kaggle score
        model_df = model_df.sort_values('kaggle_score')
        
        for idx, row in model_df.iterrows():
            print(f"\n--- Run {idx} (Kaggle: {row['kaggle_score']:.5f}, CV RMSE: {row['rmse']:.6f}) ---")
            print(f"Timestamp: {row['timestamp']}")
            print(f"Notes: {row['notes']}")
            
            try:
                hyperparams = json.loads(row['hyperparams'])
                print("Hyperparameters:")
                for key, value in sorted(hyperparams.items()):
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    elif isinstance(value, list):
                        print(f"  {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
                    else:
                        print(f"  {key}: {value}")
            except:
                print(f"  Could not parse hyperparameters: {row['hyperparams']}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_hyperparameters()

