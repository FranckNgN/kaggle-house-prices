"""Utilities for logging model performance metrics."""
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

def log_model_result(
    model_name: str,
    rmse: float,
    hyperparams: Dict[str, Any],
    features: Optional[List[str]] = None,
    log_path: str = "runs/model_performance.csv",
    notes: Optional[str] = None
) -> None:
    """
    Log model performance, hyperparameters, and features to a CSV file.
    
    Args:
        model_name: Name of the model
        rmse: Root Mean Squared Error (OOF or CV)
        hyperparams: Dictionary of hyperparameters used
        features: List of feature names used
        log_path: Path to the CSV log file
        notes: Optional notes about the run
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort features to ensure consistent comparison
    features_str = json.dumps(sorted(features)) if features else "[]"
    
    # Prepare data
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "rmse": round(rmse, 6),
        "hyperparams": json.dumps(hyperparams, sort_keys=True),
        "features": features_str,
        "notes": notes if notes else ""
    }
    
    # Load existing log or create new one
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            
            # Check for redundancy (same model, same hyperparams, same RMSE, same features)
            is_redundant = not df[
                (df["model"] == model_name) & 
                (df["rmse"] == new_entry["rmse"]) & 
                (df["hyperparams"] == new_entry["hyperparams"]) &
                (df["features"] == new_entry["features"])
            ].empty
            
            if is_redundant:
                print(f"    [Skip Log] Identical run (params + features) already exists for {model_name}.")
                return

            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])
    
    # Save back to CSV
    df.to_csv(log_file, index=False)
    
    # Print comparison with last run of same model if exists
    if len(df) > 1:
        same_model = df[df["model"] == model_name]
        if len(same_model) > 1:
            prev_row = same_model.iloc[-2]
            
            # RMSE Comparison
            prev_rmse = prev_row["rmse"]
            diff = rmse - prev_rmse
            trend = "📉 IMPROVED" if diff < 0 else "📈 WORSE"
            if abs(diff) < 1e-6:
                trend = "➖ NO CHANGE"
            print(f"    RMSE Comparison: {trend} (diff: {diff:.6f})")
            
            # Feature Engineering Comparison
            if "features" in prev_row:
                prev_features = json.loads(prev_row["features"])
                curr_features = sorted(features) if features else []
                
                if prev_features != curr_features:
                    added = set(curr_features) - set(prev_features)
                    removed = set(prev_features) - set(curr_features)
                    print(f"    ✨ Feature Set Changed: {len(prev_features)} -> {len(curr_features)}")
                    if added: print(f"      + Added: {list(added)[:5]}{'...' if len(added) > 5 else ''}")
                    if removed: print(f"      - Removed: {list(removed)[:5]}{'...' if len(removed) > 5 else ''}")
                else:
                    print(f"    ✅ Feature Set: Identical ({len(curr_features)} features)")
        
    print(f"    Logged results to {log_path}")

def get_best_models(log_path: str = "runs/model_performance.csv") -> pd.DataFrame:
    """Get the best (lowest RMSE) run for each model."""
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    df = pd.read_csv(log_path)
    # Get index of minimum RMSE for each model
    idx = df.groupby("model")["rmse"].idxmin()
    return df.loc[idx].sort_values("rmse")

