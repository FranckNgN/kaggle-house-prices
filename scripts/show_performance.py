import pandas as pd
import json
import os
import argparse
from pathlib import Path
import config_local.local_config as config

def show_performance(log_path=None, show_details=False):
    """Show model performance logs."""
    if log_path is None:
        log_path = config.MODEL_PERFORMANCE_CSV
    if not os.path.exists(log_path):
        print(f"\nNo performance logs found at {log_path}")
        print("Run a model script (e.g., python notebooks/Models/11stackingModel.py) to generate logs.\n")
        return

    df = pd.read_csv(log_path)
    
    print("\n" + "="*80)
    print("      MODEL PERFORMANCE HISTORY")
    print("="*80)
    
    # Sort by timestamp to show latest first
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)
    
    # Format the display
    display_df = df.copy()
    display_df['rmse'] = display_df['rmse'].map('{:.6f}'.format)
    
    # Handle feature info from new hashing system
    if 'feat_hash' not in display_df.columns:
        display_df['feat_hash'] = "old_sys"
    if 'feat_count' not in display_df.columns:
        if 'features' in display_df.columns:
            display_df['feat_count'] = display_df['features'].apply(lambda x: len(json.loads(x)) if pd.notnull(x) else 0)
        else:
            display_df['feat_count'] = 0

    # Print the table
    cols_to_show = ['timestamp', 'model', 'rmse', 'feat_count', 'feat_hash', 'notes']
    # Filter to only show columns that actually exist
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    print(display_df[cols_to_show].to_string(index=False))
    
    if show_details:
        print("\n" + "="*80)
        print("      ENGINEERING DETAILS FOR LATEST RUNS")
        print("="*80)
        feat_dir = config.RUNS_DIR / "feature_definitions"
        
        # Show details for the top 3 unique feature hashes
        seen_hashes = set()
        count = 0
        for _, row in display_df.iterrows():
            f_hash = row['feat_hash']
            if f_hash != "none" and f_hash not in seen_hashes:
                feat_file = feat_dir / f"{f_hash}.json"
                if feat_file.exists():
                    with open(feat_file, "r") as f:
                        data = json.load(f)
                    print(f"\n[Hash: {f_hash}] (Used by {row['model']} at {row['timestamp']})")
                    eng = data.get("engineering", {})
                    if eng:
                        for stage, details in eng.items():
                            print(f"  * {stage}:")
                            for k, v in details.items():
                                print(f"    - {k}: {v}")
                    else:
                        print("  (No engineering metadata found for this hash)")
                seen_hashes.add(f_hash)
                count += 1
                if count >= 3: break

    print("\n" + "="*80)
    print("      BEST RUN FOR EACH MODEL")
    print("="*80)
    
    # Get best RMSE for each model
    best_df = df.loc[df.groupby('model')['rmse'].idxmin()].sort_values('rmse')
    best_df['rmse'] = best_df['rmse'].map('{:.6f}'.format)
    
    # Apply same logic for best_df
    if 'feat_hash' not in best_df.columns:
        best_df['feat_hash'] = "old_sys"
    if 'feat_count' not in best_df.columns:
        if 'features' in best_df.columns:
            best_df['feat_count'] = best_df['features'].apply(lambda x: len(json.loads(x)) if pd.notnull(x) else 0)
        else:
            best_df['feat_count'] = 0

    cols_best = ['model', 'rmse', 'feat_count', 'feat_hash', 'timestamp']
    cols_best = [c for c in cols_best if c in best_df.columns]
    print(best_df[cols_best].to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show model performance logs.")
    parser.add_argument("--details", action="store_true", help="Show engineering details for recent runs")
    args = parser.parse_args()
    
    show_performance(show_details=args.details)

