#!/usr/bin/env python
"""
Run ModelComparison notebook cells to generate visualizations.
This script executes the notebook logic and generates all comparison plots.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_local import local_config as config
from scripts.compare_models import load_all_submissions

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("MODEL COMPARISON ANALYSIS")
print("="*80)

# Load data
print("\n1. Loading model predictions and performance metrics...")
df_predictions = load_all_submissions(config.SUBMISSIONS_DIR)
df_performance = pd.read_csv(config.MODEL_PERFORMANCE_CSV)
df_performance['timestamp'] = pd.to_datetime(df_performance['timestamp'])

# Get latest entry for each model
latest_performance = df_performance.sort_values('timestamp').groupby('model').last().reset_index()

# Create lookup dictionary
perf_lookup = {}
for _, row in latest_performance.iterrows():
    perf_lookup[row['model'].lower()] = {
        'rmse': row['rmse'],
        'kaggle_score': row['kaggle_score'] if pd.notna(row['kaggle_score']) else None,
        'timestamp': row['timestamp'],
        'notes': row['notes']
    }

# Normalize model names
def normalize_model_name(submission_name):
    name = submission_name.lower()
    if '_' in name:
        parts = name.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            name = '_'.join(parts[1:])
    
    name_mapping = {
        'xgboost': 'xgboost', 'xgb': 'xgboost',
        'lightgbm': 'lightgbm', 'lgb': 'lightgbm', 'lightgb': 'lightgbm',
        'catboost': 'catboost', 'cat': 'catboost',
        'ridge': 'ridge', 'lasso': 'lasso',
        'elasticnet': 'elastic_net', 'elastic_net': 'elastic_net',
        'randomforest': 'random_forest', 'random_forest': 'random_forest', 'rf': 'random_forest',
        'svr': 'svr', 'blending': 'blending', 'blend': 'blending',
        'stacking': 'STACKING_META', 'stack': 'STACKING_META',
        'linearregression': 'linear_regression', 'linear_regression': 'linear_regression',
    }
    
    if name in name_mapping:
        return name_mapping[name]
    for key, value in name_mapping.items():
        if key in name:
            return value
    return name

print(f"   Loaded {len(df_predictions.columns)} submission files")
print(f"   Loaded {len(latest_performance)} model performance entries")

# Filter valid models
print("\n2. Filtering valid models...")
valid_models = [col for col in df_predictions.columns if df_predictions[col].mean() < 1e7]
invalid_models = [col for col in df_predictions.columns if col not in valid_models]

print(f"   Valid models: {len(valid_models)}")
print(f"   Invalid models (exploded): {len(invalid_models)}")

if invalid_models:
    print(f"   [WARNING] Invalid models (excluded): {', '.join(invalid_models)}")

df_valid = df_predictions[valid_models]

# Load training target
print("\n3. Loading training target data...")
try:
    train_df = pd.read_csv(config.TRAIN_PROCESS8_CSV)
    y_train = np.expm1(train_df["logSP"])
    has_train = True
    print(f"   [OK] Loaded training data: {len(y_train)} samples")
    print(f"   Target range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    print(f"   Target mean: ${y_train.mean():,.0f}")
except Exception as e:
    print(f"   [ERROR] Could not load training data: {e}")
    y_train = None
    has_train = False

# Create output directory
output_dir = config.RUNS_DIR / "latest" / "comparison"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Distribution Comparison (KDE)
print("\n4. Generating distribution comparison plot...")
plt.figure(figsize=(14, 8))
if has_train:
    sns.kdeplot(y_train, label="TRAIN (Actual Target)", color="black", linewidth=3, linestyle="--", alpha=0.8)

for col in valid_models:
    sns.kdeplot(df_valid[col], label=col, alpha=0.6)

plt.title("SalePrice Distribution: Model Predictions vs. Actual Training Target", fontsize=16, pad=20, fontweight='bold')
plt.xlabel("SalePrice ($)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 800000)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "distribution_comparison.png", dpi=150, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'distribution_comparison.png'}")
plt.close()

# 2. Boxplot Comparison
print("\n5. Generating boxplot comparison...")
df_melted = df_valid.melt(var_name="Model", value_name="SalePrice")

if has_train:
    train_melted = pd.DataFrame({"Model": ["TRAIN (Actual)"] * len(y_train), "SalePrice": y_train})
    df_melted = pd.concat([train_melted, df_melted], ignore_index=True)

plt.figure(figsize=(14, 8))
sns.boxplot(x="SalePrice", y="Model", data=df_melted, orient="h")
plt.title("SalePrice Range Comparison: Model Predictions vs. Actual Training Target", fontsize=16, pad=20, fontweight='bold')
plt.xlabel("SalePrice ($)", fontsize=12)
plt.xlim(0, 800000)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(output_dir / "boxplot_comparison.png", dpi=150, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'boxplot_comparison.png'}")
plt.close()

# 3. Histogram comparison: Predictions vs Target
if has_train and len(valid_models) > 0:
    print("\n6. Generating histogram comparison (predictions vs target)...")
    n_models = min(4, len(valid_models))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    top_models = valid_models[:n_models]
    
    for idx, model_name in enumerate(top_models):
        ax = axes[idx]
        
        # Histogram comparison
        ax.hist(y_train, bins=50, alpha=0.5, label="TRAIN (Actual)", color="black", density=True)
        ax.hist(df_valid[model_name], bins=50, alpha=0.5, label=model_name, density=True)
        
        ax.set_xlabel("SalePrice ($)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{model_name} vs Target Distribution", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 800000)
    
    plt.suptitle("Model Predictions vs Training Target Distribution (Histograms)", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_vs_target.png", dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved: {output_dir / 'histogram_vs_target.png'}")
    plt.close()

# 4. Correlation Heatmap
print("\n7. Generating correlation heatmap...")
corr_matrix = df_valid.corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", center=1, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Between Model Predictions", fontsize=16, pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'correlation_heatmap.png'}")
plt.close()

# 5. Print statistics
print("\n" + "="*80)
print("DISTRIBUTION STATISTICS: Predictions vs Target")
print("="*80)
print(f"{'Model':<30s} {'Mean':<15s} {'Std':<15s} {'Min':<15s} {'Max':<15s}")
print("-"*80)
if has_train:
    print(f"{'TRAIN (Actual)':<30s} ${y_train.mean():>13,.0f} ${y_train.std():>13,.0f} ${y_train.min():>13,.0f} ${y_train.max():>13,.0f}")
for col in valid_models:
    preds = df_valid[col]
    print(f"{col:<30s} ${preds.mean():>13,.0f} ${preds.std():>13,.0f} ${preds.min():>13,.0f} ${preds.max():>13,.0f}")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print(f"All visualizations saved to: {output_dir}")
print("="*80)

