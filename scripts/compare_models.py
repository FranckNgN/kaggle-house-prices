import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config_local import local_config as config


def load_all_submissions(submissions_dir=None):
    """Load all model submission CSVs into a dictionary of DataFrames."""
    if submissions_dir is None:
        submissions_dir = config.SUBMISSIONS_DIR
    """Load all model submission CSVs into a dictionary of DataFrames."""
    submissions = {}
    
    # Recursively find all .csv files
    for csv_path in Path(submissions_dir).rglob("*.csv"):
        # Skip sample submission
        if csv_path.name.lower() == "sample_submission.csv":
            continue
            
        # Use folder name + filename for uniqueness
        model_name = f"{csv_path.parent.name}/{csv_path.name}"
        # Simpler name for display
        display_name = csv_path.parent.name if csv_path.parent.name != "submissions" else csv_path.stem
        
        try:
            df = pd.read_csv(csv_path)
            if "Id" in df.columns and "SalePrice" in df.columns:
                submissions[display_name] = df.set_index("Id")["SalePrice"]
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            
    return pd.DataFrame(submissions)

def plot_comparison(df, output_dir):
    """Generate various comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Training Distribution as Reference
    try:
        train_df = pd.read_csv(config.TRAIN_PROCESS8_CSV)
        y_train = np.expm1(train_df["logSP"])
    except Exception as e:
        print(f"Warning: Could not load training data for reference: {e}")
        y_train = None

    # 1. Correlation Heatmap
    # ... (skipping for brevity in search_replace, but I'll update the whole function)
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", center=1)
    plt.title("Correlation Between Model Predictions")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()
    
    # 2. Distribution Plot (KDE) - Now with Train Reference
    plt.figure(figsize=(12, 7))
    if y_train is not None:
        sns.kdeplot(y_train, label="TRAIN (Actual)", color="black", linewidth=3, linestyle="--")
    
    # Filter out models that exploded for clearer distribution plot
    valid_cols = [c for c in df.columns if df[c].mean() < 1e7]
    for col in valid_cols:
        sns.kdeplot(df[col], label=col, alpha=0.5)
        
    plt.title("SalePrice Distribution: Predictions vs. Train Reality")
    plt.xlabel("SalePrice")
    plt.xlim(0, 800000) # Limit x-axis for better visibility
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_comparison.png")
    plt.close()
    
    # 3. Boxplot Comparison
    plt.figure(figsize=(12, 8))
    # Melt the dataframe for seaborn boxplot
    df_melted = df.melt(var_name="Model", value_name="SalePrice")
    
    # Add Train data to the melted dataframe for comparison
    if y_train is not None:
        train_melted = pd.DataFrame({"Model": ["TRAIN (Actual)"] * len(y_train), "SalePrice": y_train})
        df_melted = pd.concat([train_melted, df_melted], ignore_index=True)

    # Filter out exploded models for boxplot
    valid_models = df_melted.groupby("Model")["SalePrice"].mean()
    valid_models = valid_models[valid_models < 1e7].index
    df_plot = df_melted[df_melted["Model"].isin(valid_models)]

    sns.boxplot(x="SalePrice", y="Model", data=df_plot)
    plt.title("SalePrice Range: Predictions vs. Train Reality")
    plt.xlim(0, 800000)
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_comparison.png")
    plt.close()

    # 4. Pairwise scatter plots for top models (most impactful)
    # Pick a few key models if too many
    top_models = [c for c in df.columns if any(x in c for x in ["xgboost", "lightgbm", "catboost", "blending", "stacking"])]
    if len(top_models) >= 2:
        plt.figure(figsize=(15, 15))
        sns.pairplot(df[top_models], diag_kind="kde", plot_kws={'alpha': 0.4})
        plt.suptitle("Pairwise Comparison of Top Models", y=1.02)
        plt.savefig(output_dir / "pairwise_comparison.png")
        plt.close()

    print(f"Comparison plots saved to: {output_dir}")

def main():
    submissions_dir = config.SUBMISSIONS_DIR
    output_dir = config.RUNS_DIR / "latest" / "comparison"
    
    print(f"Loading submissions from {submissions_dir}...")
    df_compare = load_all_submissions(submissions_dir)
    
    if df_compare.empty:
        print("No submission files found to compare.")
        return
        
    print(f"Found {len(df_compare.columns)} models to compare.")
    print(df_compare.describe())
    
    plot_comparison(df_compare, output_dir)

if __name__ == "__main__":
    main()

