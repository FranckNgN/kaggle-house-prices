"""Utilities for logging model performance metrics."""
import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import config_local.local_config as config

def get_feature_hash(features: List[str]) -> str:
    """Generate a stable hash for a list of features."""
    feat_str = json.dumps(sorted(features))
    return hashlib.md5(feat_str.encode()).hexdigest()[:8]

def format_runtime(seconds: float) -> str:
    """
    Format runtime in a human-readable format.
    
    Args:
        seconds: Runtime in seconds
        
    Returns:
        Formatted string like "2h 15m 30s", "45m 12s", or "30s"
    """
    if seconds is None or seconds == "":
        return ""
    
    seconds = float(seconds)
    
    if seconds >= 3600:  # >= 1 hour
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{hours}h {minutes}m {secs}s"
        else:
            return f"{hours}h {secs}s"
    elif seconds >= 60:  # >= 1 minute
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:  # < 1 minute
        return f"{int(seconds)}s"

def log_model_result(
    model_name: str,
    rmse: float,
    hyperparams: Dict[str, Any],
    features: Optional[List[str]] = None,
    log_path: Optional[str] = None,
    notes: Optional[str] = None,
    runtime: Optional[float] = None,
    kaggle_score: Optional[float] = None
) -> None:
    """
    Log model performance, hyperparameters, and features to a CSV file.
    
    Args:
        model_name: Name of the model
        rmse: Cross-validation RMSE score
        hyperparams: Dictionary of hyperparameters
        features: Optional list of feature names
        log_path: Path to the log CSV file (defaults to config.MODEL_PERFORMANCE_CSV)
        notes: Optional notes about the run
        runtime: Optional runtime in seconds (for tracking how long model took to train)
        kaggle_score: Optional Kaggle leaderboard score (RMSLE)
    """
    if log_path is None:
        log_path = str(config.MODEL_PERFORMANCE_CSV)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load current engineering summary if it exists
    engineering_summary = {}
    summary_path = config.RUNS_DIR / "current_engineering_summary.json"
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
        feat_dir = config.RUNS_DIR / "feature_definitions"
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
    # Format runtime in human-readable format
    runtime_formatted = format_runtime(runtime) if runtime is not None else ""
    
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "rmse": round(rmse, 6),
        "hyperparams": json.dumps(hyperparams, sort_keys=True),
        "feat_hash": feat_hash,
        "feat_count": feat_count,
        "notes": notes if notes else "",
        "runtime": runtime_formatted,
        "kaggle_score": round(kaggle_score, 6) if kaggle_score is not None else ""
    }
    
    # Load existing log or create new one
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            
            # Ensure runtime and kaggle_score columns exist for backward compatibility
            if "runtime" not in df.columns:
                df["runtime"] = ""
            if "kaggle_score" not in df.columns:
                df["kaggle_score"] = ""
            
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
            trend = "[IMPROVED]" if diff < 0 else "[WORSE]"
            if abs(diff) < 1e-6: trend = "[NO CHANGE]"
            print(f"    RMSE Comparison: {trend} (diff: {diff:.6f})")
            
            # Feature Comparison using Hashes
            prev_hash = str(prev_row.get("feat_hash", "none"))
            if prev_hash != feat_hash:
                print(f"    [CHANGED] Feature Set Changed: {prev_hash} -> {feat_hash} ({feat_count} features)")
                # Optionally load files to show diff
                prev_file = config.RUNS_DIR / "feature_definitions" / f"{prev_hash}.json"
                if prev_file.exists() and features:
                    with open(prev_file, "r") as f:
                        prev_features = json.load(f)
                    added = set(features) - set(prev_features)
                    removed = set(prev_features) - set(features)
                    if added: print(f"      + Added: {list(added)[:3]}{'...' if len(added) > 3 else ''}")
                    if removed: print(f"      - Removed: {list(removed)[:3]}{'...' if len(removed) > 3 else ''}")
            else:
                print(f"    [IDENTICAL] Feature Set: Identical ({feat_count} features)")
        
    print(f"    Logged results to {log_path}")

def log_kaggle_score(
    model_name: str,
    kaggle_score: float,
    log_path: Optional[str] = None
) -> None:
    """
    Update the most recent entry for a model with Kaggle submission score.
    
    Args:
        model_name: Name of the model
        kaggle_score: Kaggle leaderboard score (RMSLE)
        log_path: Path to the log CSV file (defaults to config.MODEL_PERFORMANCE_CSV)
    """
    if log_path is None:
        log_path = str(config.MODEL_PERFORMANCE_CSV)
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"    [Warning] Log file not found: {log_path}")
        return
    
    try:
        df = pd.read_csv(log_file)
        
        # Ensure kaggle_score column exists
        if "kaggle_score" not in df.columns:
            df["kaggle_score"] = ""
        
        # Find the most recent entry for this model
        model_entries = df[df["model"] == model_name]
        if len(model_entries) == 0:
            print(f"    [Warning] No entries found for model: {model_name}")
            return
        
        # Get the index of the most recent entry (last row for this model)
        most_recent_idx = model_entries.index[-1]
        
        # Update the kaggle_score
        df.at[most_recent_idx, "kaggle_score"] = round(kaggle_score, 6)
        
        # Save back to CSV
        df.to_csv(log_file, index=False)
        print(f"    [SUCCESS] Updated Kaggle score for {model_name}: {kaggle_score:.6f}")
        
    except Exception as e:
        print(f"    [Warning] Error updating Kaggle score: {e}")


def get_best_models(log_path: Optional[str] = None) -> pd.DataFrame:
    """Get the best (lowest RMSE) run for each model."""
    if log_path is None:
        log_path = str(config.MODEL_PERFORMANCE_CSV)
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    df = pd.read_csv(log_path)
    # Get index of minimum RMSE for each model
    idx = df.groupby("model")["rmse"].idxmin()
    return df.loc[idx].sort_values("rmse")

