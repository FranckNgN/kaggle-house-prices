#!/usr/bin/env python
"""
Sync Kaggle scores from submission log to model performance CSV.
This ensures all submitted models have their Kaggle scores logged.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config_local.local_config as config
from kaggle import load_submission_log

def normalize_model_name(file_name: str) -> str:
    """Normalize submission file name to model name in performance CSV."""
    # Remove .csv extension
    name = file_name.replace(".csv", "").lower()
    
    # Map file names to model names
    name_mapping = {
        "catboost_model": "catboost",
        "xgboost_model": "xgboost",
        "lightgbm_model": "lightgbm",
        "ridge_model_kfold": "ridge",
        "lasso_model_kfold": "lasso",
        "elasticnet_model": "elastic_net",
        "randomforest_model": "random_forest",
        "svr_model": "svr",
        "blend_xgb_lgb_cat_model": "blending",
        "stacking_submission": "STACKING_META",
        "naive_lr": "linear_regression",
    }
    
    # Try exact match first
    if name in name_mapping:
        return name_mapping[name]
    
    # Try partial matches
    for key, value in name_mapping.items():
        if key in name:
            return value
    
    # Default: return as-is
    return name

def sync_kaggle_scores():
    """Sync Kaggle scores from submission log to performance CSV."""
    print("=" * 70)
    print("SYNCING KAGGLE SCORES TO MODEL PERFORMANCE CSV")
    print("=" * 70)
    
    # Load submission log
    submission_log = load_submission_log()
    if not submission_log:
        print("[WARNING] No submission log found.")
        return
    
    print(f"\nLoaded {len(submission_log)} submission entries")
    
    # Load performance CSV
    perf_csv = config.MODEL_PERFORMANCE_CSV
    if not perf_csv.exists():
        print(f"[ERROR] Performance CSV not found: {perf_csv}")
        return
    
    df = pd.read_csv(perf_csv)
    
    # Ensure kaggle_score column exists
    if "kaggle_score" not in df.columns:
        df["kaggle_score"] = ""
    
    # Convert timestamp to datetime for comparison
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Track updates
    updates = []
    
    # Process each submission
    for entry in submission_log:
        file_name = entry.get("file", "")
        score = entry.get("score")
        timestamp_str = entry.get("timestamp", "")
        
        if not score or not timestamp_str:
            continue
        
        # Normalize model name
        model_name = normalize_model_name(file_name)
        
        # Parse timestamp
        try:
            sub_timestamp = pd.to_datetime(timestamp_str)
        except:
            continue
        
        # Find matching entries in performance CSV
        # Match by model name and find the closest timestamp
        model_entries = df[df["model"] == model_name].copy()
        
        if len(model_entries) == 0:
            # Try alternative names
            alt_names = {
                "STACKING_META": "stacking",
                "blending": "blend",
            }
            if model_name in alt_names:
                model_entries = df[df["model"] == alt_names[model_name]].copy()
        
        if len(model_entries) == 0:
            print(f"[SKIP] No performance entry found for model: {model_name} (file: {file_name})")
            continue
        
        # Find the entry closest to submission timestamp
        model_entries["time_diff"] = (model_entries["timestamp"] - sub_timestamp).abs()
        closest_idx = model_entries["time_diff"].idxmin()
        
        # Check if score is already logged
        current_score = df.at[closest_idx, "kaggle_score"]
        if pd.notna(current_score) and float(current_score) == float(score):
            continue  # Already logged
        
        # Update the score
        df.at[closest_idx, "kaggle_score"] = round(float(score), 6)
        updates.append({
            "model": model_name,
            "file": file_name,
            "score": score,
            "timestamp": timestamp_str
        })
    
    # Save updated CSV
    if updates:
        df.to_csv(perf_csv, index=False)
        print(f"\n[SUCCESS] Updated {len(updates)} entries:")
        for update in updates:
            print(f"  - {update['model']}: {update['score']:.6f} (from {update['file']})")
    else:
        print("\n[INFO] No updates needed. All scores are already synced.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    sync_kaggle_scores()

