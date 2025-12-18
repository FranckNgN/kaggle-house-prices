"""Utilities for logging model performance metrics."""
import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

def get_feature_hash(features: List[str]) -> str:
    """Generate a stable hash for a list of features."""
    feat_str = json.dumps(sorted(features))
    return hashlib.md5(feat_str.encode()).hexdigest()[:8]

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
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load current engineering summary if it exists
    engineering_summary = {}
    summary_path = log_file.parent / "current_engineering_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                engineering_summary = json.load(f)
        except Exception:
            pass

    # Handle features with hashing to keep CSV clean
    feat_hash = "none"
    feat_count = 0
    if features:
        feat_hash = get_feature_hash(features)
        feat_count = len(features)
        
        # Save full feature definition to a separate JSON file
        feat_dir = log_file.parent / "feature_definitions"
        feat_dir.mkdir(parents=True, exist_ok=True)
        feat_file = feat_dir / f"{feat_hash}.json"
        
        # We always save/update the definition to include current engineering summary
        definition = {
            "features": sorted(features),
            "engineering": engineering_summary,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(feat_file, "w") as f:
            json.dump(definition, f, indent=2)

    # Prepare data for CSV
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "rmse": round(rmse, 6),
        "hyperparams": json.dumps(hyperparams, sort_keys=True),
        "feat_hash": feat_hash,
        "feat_count": feat_count,
        "notes": notes if notes else ""
    }
    
    # Load existing log or create new one
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            
            # Check for redundancy (same model, same hyperparams, same RMSE, same feat_hash)
            is_redundant = not df[
                (df["model"] == model_name) & 
                (df["rmse"] == new_entry["rmse"]) & 
                (df["hyperparams"] == new_entry["hyperparams"]) &
                (df.get("feat_hash", "") == feat_hash)
            ].empty
            
            if is_redundant:
                print(f"    [Skip Log] Identical run already exists for {model_name}.")
                return

            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception as e:
            print(f"    [Warning] Log error: {e}. Starting fresh.")
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])
    
    # Save back to CSV
    df.to_csv(log_file, index=False)
    
    # Print comparison with last run of same model
    if len(df) > 1:
        same_model = df[df["model"] == model_name]
        if len(same_model) > 1:
            prev_row = same_model.iloc[-2]
            
            # RMSE Comparison
            prev_rmse = prev_row["rmse"]
            diff = rmse - prev_rmse
            trend = "📉 IMPROVED" if diff < 0 else "📈 WORSE"
            if abs(diff) < 1e-6: trend = "➖ NO CHANGE"
            print(f"    RMSE Comparison: {trend} (diff: {diff:.6f})")
            
            # Feature Comparison using Hashes
            prev_hash = str(prev_row.get("feat_hash", "none"))
            if prev_hash != feat_hash:
                print(f"    ✨ Feature Set Changed: {prev_hash} -> {feat_hash} ({feat_count} features)")
                # Optionally load files to show diff
                feat_dir = log_file.parent / "feature_definitions"
                prev_file = feat_dir / f"{prev_hash}.json"
                if prev_file.exists() and features:
                    with open(prev_file, "r") as f:
                        prev_features = json.load(f)
                    added = set(features) - set(prev_features)
                    removed = set(prev_features) - set(features)
                    if added: print(f"      + Added: {list(added)[:3]}{'...' if len(added) > 3 else ''}")
                    if removed: print(f"      - Removed: {list(removed)[:3]}{'...' if len(removed) > 3 else ''}")
            else:
                print(f"    ✅ Feature Set: Identical ({feat_count} features)")
        
    print(f"    Logged results to {log_path}")

def get_best_models(log_path: str = "runs/model_performance.csv") -> pd.DataFrame:
    """Get the best (lowest RMSE) run for each model."""
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    df = pd.read_csv(log_path)
    # Get index of minimum RMSE for each model
    idx = df.groupby("model")["rmse"].idxmin()
    return df.loc[idx].sort_values("rmse")

