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

# 5. Pairwise Comparison (scatter plots for top models)
print("\n8. Generating pairwise comparison plot...")
if len(valid_models) >= 2:
    # Select top models (prioritize tree-based and ensemble models)
    top_models = []
    priority_keywords = ["catboost", "xgboost", "lightgbm", "blending", "stacking", "random_forest"]
    
    # First, add models with priority keywords
    for keyword in priority_keywords:
        for model in valid_models:
            if keyword.lower() in model.lower() and model not in top_models:
                top_models.append(model)
    
    # Add other models if we have space (limit to 6 models for readability)
    max_models = 6
    for model in valid_models:
        if len(top_models) >= max_models:
            break
        if model not in top_models:
            top_models.append(model)
    
    if len(top_models) >= 2:
        try:
            # Create pairwise scatter plot matrix
            fig, axes = plt.subplots(len(top_models), len(top_models), figsize=(20, 20))
            fig.suptitle("Pairwise Model Comparison Matrix", fontsize=16, fontweight='bold', y=0.995)
            
            for i, model1 in enumerate(top_models):
                for j, model2 in enumerate(top_models):
                    ax = axes[i, j]
                    
                    if i == j:
                        # Diagonal: show distribution
                        ax.hist(df_valid[model1], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                        ax.set_title(f"{model1}", fontsize=10, fontweight='bold')
                        ax.set_xlabel("SalePrice ($)", fontsize=8)
                        ax.set_ylabel("Frequency", fontsize=8)
                    else:
                        # Off-diagonal: scatter plot
                        ax.scatter(df_valid[model2], df_valid[model1], alpha=0.3, s=10, color='steelblue')
                        
                        # Add diagonal line (perfect correlation)
                        min_val = min(df_valid[model1].min(), df_valid[model2].min())
                        max_val = max(df_valid[model1].max(), df_valid[model2].max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
                        
                        # Calculate and display correlation
                        corr = df_valid[model1].corr(df_valid[model2])
                        ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                               fontsize=9, verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        if j == 0:
                            ax.set_ylabel(model1, fontsize=9, fontweight='bold')
                        if i == len(top_models) - 1:
                            ax.set_xlabel(model2, fontsize=9, fontweight='bold')
                    
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "pairwise_comparison.png", dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {output_dir / 'pairwise_comparison.png'}")
            plt.close()
        except Exception as e:
            print(f"   [WARNING] Error generating pairwise comparison: {e}")
            print(f"   [INFO] Falling back to seaborn pairplot...")
            # Fallback to seaborn pairplot
            try:
                plt.figure(figsize=(15, 15))
                sns.pairplot(df_valid[top_models], diag_kind="kde", plot_kws={'alpha': 0.4})
                plt.suptitle("Pairwise Comparison of Top Models", fontsize=16, fontweight='bold', y=1.02)
                plt.savefig(output_dir / "pairwise_comparison.png", dpi=150, bbox_inches='tight')
                print(f"   [OK] Saved: {output_dir / 'pairwise_comparison.png'} (using seaborn pairplot)")
                plt.close()
            except Exception as e2:
                print(f"   [ERROR] Could not generate pairwise comparison: {e2}")
    else:
        print(f"   [SKIP] Need at least 2 models for pairwise comparison (found {len(top_models)})")
else:
    print(f"   [SKIP] Need at least 2 models for pairwise comparison (found {len(valid_models)})")

# 6. Print statistics
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

