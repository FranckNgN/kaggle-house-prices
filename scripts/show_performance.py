import pandas as pd
import json
import os
from pathlib import Path

def show_performance(log_path="runs/model_performance.csv"):
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
    
    # Print the table
    print(display_df[['timestamp', 'model', 'rmse', 'notes']].to_string(index=False))
    
    print("\n" + "="*80)
    print("      BEST RUN FOR EACH MODEL")
    print("="*80)
    
    # Get best RMSE for each model
    best_df = df.loc[df.groupby('model')['rmse'].idxmin()].sort_values('rmse')
    best_df['rmse'] = best_df['rmse'].map('{:.6f}'.format)
    print(best_df[['model', 'rmse', 'timestamp']].to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    show_performance()

