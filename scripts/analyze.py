#!/usr/bin/env python
"""
Consolidated analysis script for model performance and comparisons.

Usage:
    python scripts/analyze.py performance          # Show model performance
    python scripts/analyze.py compare              # Compare model predictions
    python scripts/analyze.py errors <model_name>  # Analyze model errors
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing analysis functions using importlib to handle scripts directory
import importlib.util
SCRIPTS_DIR = Path(__file__).resolve().parent

# Import show_performance
show_perf_path = SCRIPTS_DIR / "show_performance.py"
spec_perf = importlib.util.spec_from_file_location("show_performance", show_perf_path)
show_perf_module = importlib.util.module_from_spec(spec_perf)
spec_perf.loader.exec_module(show_perf_module)
show_performance = show_perf_module.show_performance

# Import compare_models functions
compare_path = SCRIPTS_DIR / "compare_models.py"
spec_compare = importlib.util.spec_from_file_location("compare_models", compare_path)
compare_module = importlib.util.module_from_spec(spec_compare)
spec_compare.loader.exec_module(compare_module)
load_all_submissions = compare_module.load_all_submissions
plot_comparison = compare_module.plot_comparison

import config_local.local_config as config


def analyze_performance(show_details: bool = False):
    """Show model performance logs."""
    show_performance(show_details=show_details)


def analyze_compare():
    """Compare model predictions."""
    print("=" * 70)
    print("COMPARING MODEL PREDICTIONS")
    print("=" * 70)
    
    # Load all submissions
    print("\nLoading submission files...")
    df = load_all_submissions()
    
    if df.empty:
        print("\n[ERROR] No submission files found.")
        print(f"   Check: {config.SUBMISSIONS_DIR}")
        return
    
    print(f"\nFound {len(df.columns)} models:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Generate comparison plots
    output_dir = config.RUNS_DIR / "latest" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating comparison plots...")
    print(f"Output directory: {output_dir}")
    
    plot_comparison(df, output_dir)
    
    print("\n[SUCCESS] Comparison plots generated!")
    print(f"  - correlation_heatmap.png")
    print(f"  - distribution_comparison.png")
    print(f"  - boxplot_comparison.png")
    print(f"  - scatter_matrix.png")


def analyze_errors(model_name: str = "catboost"):
    """Analyze model prediction errors."""
    print("=" * 70)
    print(f"ANALYZING MODEL ERRORS: {model_name.upper()}")
    print("=" * 70)
    
    # Import error analysis function
    try:
        import importlib.util
        error_analysis_path = SCRIPTS_DIR / "analyze_model_errors.py"
        spec = importlib.util.spec_from_file_location("analyze_model_errors", error_analysis_path)
        error_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(error_module)
        error_module.main(model_name)
    except Exception as e:
        print(f"\n[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def analyze_best():
    """Analyze best models by CV RMSE and Kaggle score."""
    import pandas as pd
    
    log_file = config.MODEL_PERFORMANCE_CSV
    if not log_file.exists():
        print(f"[ERROR] Model performance log not found: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    df_valid = df[df['rmse'] < 1.0].copy()
    df_valid['kaggle_score'] = pd.to_numeric(df_valid['kaggle_score'], errors='coerce')
    
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal model runs: {len(df)}")
    print(f"Valid model runs (RMSE < 1.0): {len(df_valid)}")
    print(f"Models with Kaggle scores: {df_valid['kaggle_score'].notna().sum()}")
    
    # Best by CV RMSE
    print("\n" + "=" * 80)
    print("BEST MODELS BY CV RMSE (Cross-Validation)")
    print("=" * 80)
    best_cv = df_valid.nsmallest(10, 'rmse')[
        ['timestamp', 'model', 'rmse', 'kaggle_score', 'notes', 'runtime']
    ]
    print(best_cv.to_string(index=False))
    
    # Best by Kaggle Score
    print("\n" + "=" * 80)
    print("BEST MODELS BY KAGGLE SCORE (Leaderboard)")
    print("=" * 80)
    df_with_kaggle = df_valid[df_valid['kaggle_score'].notna()].copy()
    if len(df_with_kaggle) > 0:
        best_kaggle = df_with_kaggle.nsmallest(10, 'kaggle_score')[
            ['timestamp', 'model', 'rmse', 'kaggle_score', 'notes', 'runtime']
        ]
        print(best_kaggle.to_string(index=False))
        
        overall_best = df_with_kaggle.nsmallest(1, 'kaggle_score').iloc[0]
        print("\n" + "-" * 80)
        print("*** OVERALL BEST MODEL (Lowest Kaggle Score) ***")
        print("-" * 80)
        print(f"Model: {overall_best['model']}")
        print(f"Kaggle Score: {overall_best['kaggle_score']:.6f} (RMSLE)")
        print(f"CV RMSE: {overall_best['rmse']:.6f}")


def analyze_hyperparameters():
    """Analyze hyperparameters for models with Kaggle scores."""
    import pandas as pd
    import json
    
    log_file = config.MODEL_PERFORMANCE_CSV
    if not log_file.exists():
        print(f"[ERROR] Model performance log not found: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    df['kaggle_score'] = pd.to_numeric(df['kaggle_score'], errors='coerce')
    df_valid = df[(df['rmse'] < 1.0) & (df['kaggle_score'].notna())].copy()
    
    print("=" * 80)
    print("HYPERPARAMETER ANALYSIS FOR MODELS WITH KAGGLE SCORES")
    print("=" * 80)
    
    for model_name in df_valid['model'].unique():
        model_df = df_valid[df_valid['model'] == model_name].copy()
        if len(model_df) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        model_df = model_df.sort_values('kaggle_score')
        
        for idx, row in model_df.iterrows():
            print(f"\n--- Run (Kaggle: {row['kaggle_score']:.5f}, CV RMSE: {row['rmse']:.6f}) ---")
            print(f"Timestamp: {row['timestamp']}")
            try:
                hyperparams = json.loads(row['hyperparams'])
                print("Hyperparameters:")
                for key, value in sorted(hyperparams.items()):
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    elif isinstance(value, list):
                        print(f"  {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
                    else:
                        print(f"  {key}: {value}")
            except:
                print(f"  Could not parse hyperparameters")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze model performance and predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze.py performance         # Show performance summary
  python scripts/analyze.py compare             # Compare model predictions
  python scripts/analyze.py errors catboost     # Analyze CatBoost errors
  python scripts/analyze.py best                # Analyze best models
  python scripts/analyze.py hyperparameters     # Analyze hyperparameters
        """
    )
    
    parser.add_argument(
        'command',
        choices=['performance', 'compare', 'errors', 'best', 'hyperparameters'],
        help='Analysis command to execute'
    )
    parser.add_argument(
        'model_name',
        nargs='?',
        default='catboost',
        help='Model name for error analysis (default: catboost)'
    )
    parser.add_argument(
        '--details', '-d',
        action='store_true',
        help='Show detailed performance information'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'performance':
            analyze_performance(show_details=args.details)
        elif args.command == 'compare':
            analyze_compare()
        elif args.command == 'errors':
            analyze_errors(model_name=args.model_name)
        elif args.command == 'best':
            analyze_best()
        elif args.command == 'hyperparameters':
            analyze_hyperparameters()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

