#!/usr/bin/env python
"""
Consolidated Kaggle submission script.
Submit individual models or batch submit all models to Kaggle.

Usage:
    python -m kaggle.submit <model_name>          # Submit a specific model
    python -m kaggle.submit --interactive          # Interactive mode
    python -m kaggle.submit --all                  # Submit all available models
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle import (
    submit_and_check,
    load_submission_log,
    get_available_submissions,
    has_been_submitted,
    check_submission_limit
)
import config_local.local_config as config
from config_local import model_config


def get_model_config(model_name: str):
    """Get model configuration by name."""
    model_name_lower = model_name.lower().replace(" ", "_")
    
    # Map model names to config keys
    config_map = {
        'xgboost': ('XGBOOST', 'xgboost'),
        'xgb': ('XGBOOST', 'xgboost'),
        'lightgbm': ('LIGHTGBM', 'lightgbm'),
        'lgb': ('LIGHTGBM', 'lightgbm'),
        'catboost': ('CATBOOST', 'catboost'),
        'cat': ('CATBOOST', 'catboost'),
        'ridge': ('RIDGE', 'ridge'),
        'lasso': ('LASSO', 'lasso'),
        'elastic_net': ('ELASTIC_NET', 'elastic_net'),
        'elasticnet': ('ELASTIC_NET', 'elastic_net'),
        'random_forest': ('RANDOM_FOREST', 'random_forest'),
        'randomforest': ('RANDOM_FOREST', 'random_forest'),
        'rf': ('RANDOM_FOREST', 'random_forest'),
        'svr': ('SVR', 'svr'),
        'blending': ('BLENDING', 'blending'),
        'stacking': ('STACKING', 'stacking'),
    }
    
    if model_name_lower not in config_map:
        return None, None
    
    config_key, normalized_name = config_map[model_name_lower]
    cfg = getattr(model_config, config_key, None)
    
    return cfg, normalized_name


def display_submissions(submissions: List[Dict]):
    """Display available submission files."""
    print("\n" + "=" * 70)
    print("AVAILABLE SUBMISSION FILES")
    print("=" * 70)
    for i, sub in enumerate(submissions, 1):
        print(f"  {i:2d}. {sub['name']:30s} ({sub['model']})")
    print("=" * 70)


def submit_single_model_interactive(submissions: List[Dict]) -> Optional[Dict]:
    """Submit a single model selected by user interactively."""
    display_submissions(submissions)
    
    while True:
        try:
            choice = input(f"\nSelect a model to submit (1-{len(submissions)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(submissions):
                selected = submissions[idx]
                message = input(f"Enter submission message (or press Enter for default): ").strip()
                if not message:
                    message = f"{selected['model']} model - Optuna optimized"
                
                print(f"\nSubmitting: {selected['name']}")
                result = submit_and_check(selected['file'], message)
                return result
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(submissions)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None


def submit_model_by_name(model_name: str) -> Optional[Dict]:
    """Submit a specific model by name."""
    print("=" * 70)
    print(f"MODEL SUBMISSION TO KAGGLE: {model_name.upper()}")
    print("=" * 70)
    
    # Get model config
    cfg, normalized_name = get_model_config(model_name)
    
    if cfg is None:
        print(f"\n[ERROR] Unknown model: {model_name}")
        print("   Use one of: xgboost, lightgbm, catboost, ridge, lasso, etc.")
        return None
    
    submission_path = config.get_model_submission_path(
        cfg["submission_name"], 
        cfg["submission_filename"]
    )
    
    # Check if file exists
    if not submission_path.exists():
        print(f"\n[WARNING] Submission file not found: {submission_path}")
        print(f"\nTo generate the submission file, run the model script:")
        
        # Find model script
        model_scripts = {
            'xgboost': '7XGBoostModel.py',
            'lightgbm': '8lightGbmModel.py',
            'catboost': '9catBoostModel.py',
            'ridge': '2ridgeModel.py',
            'lasso': '3lassoModel.py',
            'elastic_net': '4elasticNetModel.py',
            'random_forest': '5randomForestModel.py',
            'svr': '6svrModel.py',
            'blending': '10blendingModel.py',
            'stacking': '11stackingModel.py',
        }
        
        script_name = model_scripts.get(normalized_name, 'model script')
        print(f"   python notebooks/Models/{script_name}")
        return None
    
    print(f"\nSubmission file found: {submission_path}")
    print(f"Model: {normalized_name}")
    print(f"Message: {normalized_name} model - Optuna optimized")
    print("-" * 70)
    
    result = submit_and_check(
        submission_file=str(submission_path),
        message=f"{normalized_name} model - Optuna optimized",
        wait_time=5,
        max_retries=3,
        skip_if_submitted=True
    )
    
    return result


def submit_all_models_auto(submissions: List[Dict]) -> List[Dict]:
    """Submit all available models automatically."""
    results = []
    
    print("\n" + "=" * 70)
    print("AUTOMATIC SUBMISSION OF ALL MODELS")
    print("=" * 70)
    print(f"Found {len(submissions)} submission files\n")
    
    for i, sub in enumerate(submissions, 1):
        print(f"\n[{i}/{len(submissions)}] Processing: {sub['name']}")
        print("-" * 70)
        
        # Check if already submitted
        existing = has_been_submitted(sub['file'], check_hash=True)
        if existing:
            print(f"[SKIP] Already submitted (score: {existing.get('score', 'N/A')})")
            results.append({
                'model': sub['model'],
                'file': sub['name'],
                'status': 'skipped',
                **existing
            })
            continue
        
        # Submit
        message = f"{sub['model']} model - Optuna optimized"
        result = submit_and_check(sub['file'], message, skip_if_submitted=True)
        
        if result:
            results.append({
                'model': sub['model'],
                'file': sub['name'],
                'status': 'success',
                **result
            })
        else:
            results.append({
                'model': sub['model'],
                'file': sub['name'],
                'status': 'failed'
            })
    
    return results


def display_summary(results: List[Dict]):
    """Display summary of all submissions."""
    if not results:
        return
    
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30s} {'Status':<12s} {'Rank':<10s} {'Score':<15s}")
    print("-" * 70)
    
    # Sort by score (lower is better, but handle N/A)
    def sort_key(x):
        score = x.get('score', 'N/A')
        if isinstance(score, (int, float)):
            return (0, score)
        return (1, 0)
    
    sorted_results = sorted(results, key=sort_key)
    
    for r in sorted_results:
        status = r.get('status', 'unknown')
        score_str = f"{r['score']:.6f}" if isinstance(r.get('score'), (int, float)) else str(r.get('score', 'N/A'))
        rank_str = str(r.get('rank', 'N/A'))
        print(f"{r['model']:<30s} {status:<12s} {rank_str:<10s} {score_str:<15s}")
    
    print("=" * 70)
    
    # Show best model if we have valid scores
    valid_results = [r for r in results if isinstance(r.get('score'), (int, float))]
    if valid_results:
        best = min(valid_results, key=lambda x: x['score'])
        print(f"\nBest Model: {best['model']}")
        print(f"   Score: {best['score']:.5f}")
        if isinstance(best.get('rank'), int):
            print(f"   Rank: {best['rank']}")


def interactive_mode():
    """Run interactive submission menu."""
    print("=" * 70)
    print("KAGGLE MODEL SUBMISSION MANAGER")
    print("=" * 70)
    
    # Get available submissions
    submissions = get_available_submissions(PROJECT_ROOT)
    
    if not submissions:
        print("No submission files found in data/submissions/")
        print("   Make sure you have generated submission CSV files.")
        sys.exit(1)
    
    # Main menu loop
    while True:
        print("\n" + "=" * 70)
        print("MAIN MENU")
        print("=" * 70)
        print("  1. Submit a single model")
        print("  2. Submit all models")
        print("  3. View available submissions")
        print("  4. View submission history")
        print("  5. Exit")
        print("=" * 70)
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            result = submit_single_model_interactive(submissions)
            if result:
                input("\nPress Enter to continue...")
        
        elif choice == '2':
            confirm = input(f"\nThis will submit {len(submissions)} models. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                results = submit_all_models_auto(submissions)
                display_summary(results)
                input("\nPress Enter to continue...")
            else:
                print("Cancelled.")
        
        elif choice == '3':
            display_submissions(submissions)
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            log = load_submission_log()
            if log:
                print("\n" + "=" * 70)
                print("SUBMISSION HISTORY")
                print("=" * 70)
                print(f"{'Timestamp':<20s} {'File':<30s} {'Rank':<10s} {'Score':<15s}")
                print("-" * 70)
                for entry in log[-10:]:  # Show last 10
                    rank = entry.get('rank', 'N/A')
                    score = entry.get('score', 'N/A')
                    if isinstance(score, (int, float)):
                        score_str = f"{score:.5f}"
                    else:
                        score_str = str(score)
                    print(f"{entry['timestamp']:<20s} {entry['file']:<30s} {rank:<10s} {score_str:<15s}")
                print("=" * 70)
            else:
                print("\nNo submission history found.")
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Submit models to Kaggle competition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kaggle.submit catboost          # Submit CatBoost model
  python -m kaggle.submit --interactive     # Interactive menu
  python -m kaggle.submit --all             # Submit all models
        """
    )
    
    parser.add_argument(
        'model_name',
        nargs='?',
        help='Model name to submit (e.g., catboost, xgboost, lightgbm)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Submit all available models'
    )
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_mode()
        elif args.all:
            submissions = get_available_submissions(PROJECT_ROOT)
            if not submissions:
                print("No submission files found in data/submissions/")
                sys.exit(1)
            results = submit_all_models_auto(submissions)
            display_summary(results)
        elif args.model_name:
            result = submit_model_by_name(args.model_name)
            if not result:
                sys.exit(1)
        else:
            # No arguments - show usage and available models
            print("=" * 70)
            print("KAGGLE MODEL SUBMISSION")
            print("=" * 70)
            print("\nUsage:")
            print("  python -m kaggle.submit <model_name>          # Submit a specific model")
            print("  python -m kaggle.submit --interactive          # Interactive mode")
            print("  python -m kaggle.submit --all                  # Submit all models")
            print("\nAvailable models:")
            print("  - catboost, cat")
            print("  - xgboost, xgb")
            print("  - lightgbm, lgb")
            print("  - ridge, lasso, elastic_net")
            print("  - random_forest, rf")
            print("  - svr")
            print("  - blending, stacking")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

