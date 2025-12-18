"""Utilities for logging model performance metrics."""
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def log_model_result(
    model_name: str,
    rmse: float,
    hyperparams: Dict[str, Any],
    log_path: str = "runs/model_performance.csv",
    notes: Optional[str] = None
) -> None:
    """
    Log model performance and hyperparameters to a CSV file.
    
    Args:
        model_name: Name of the model
        rmse: Root Mean Squared Error (OOF or CV)
        hyperparams: Dictionary of hyperparameters used
        log_path: Path to the CSV log file
        notes: Optional notes about the run
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "rmse": round(rmse, 6),
        "hyperparams": json.dumps(hyperparams, sort_keys=True),
        "notes": notes if notes else ""
    }
    
    # Load existing log or create new one
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            
            # Check for redundancy (same model, same hyperparams, same RMSE)
            # We compare hyperparams as JSON strings
            is_redundant = not df[
                (df["model"] == model_name) & 
                (df["rmse"] == new_entry["rmse"]) & 
                (df["hyperparams"] == new_entry["hyperparams"])
            ].empty
            
            if is_redundant:
                print(f"    [Skip Log] Identical run already exists for {model_name} (RMSE: {rmse:.6f}).")
                return

            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception:
            # Fallback if CSV is corrupted
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])
    
    # Save back to CSV
    df.to_csv(log_file, index=False)
    
    # Print comparison with last run of same model if exists
    if len(df) > 1:
        same_model = df[df["model"] == model_name]
        if len(same_model) > 1:
            prev_rmse = same_model.iloc[-2]["rmse"]
            diff = rmse - prev_rmse
            trend = "📉 IMPROVED" if diff < 0 else "📈 WORSE"
            if abs(diff) < 1e-6:
                trend = "➖ NO CHANGE"
            print(f"    Comparison to previous {model_name}: {trend} (diff: {diff:.6f})")
        
    print(f"    Logged results to {log_path}")

def get_best_models(log_path: str = "runs/model_performance.csv") -> pd.DataFrame:
    """Get the best (lowest RMSE) run for each model."""
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    df = pd.read_csv(log_path)
    # Get index of minimum RMSE for each model
    idx = df.groupby("model")["rmse"].idxmin()
    return df.loc[idx].sort_values("rmse")

