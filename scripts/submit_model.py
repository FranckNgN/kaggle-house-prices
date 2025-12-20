#!/usr/bin/env python
"""
Submit any model to Kaggle and automatically log the score.
Generalized version that works with any model name.

Usage:
    python scripts/submit_model.py <model_name>
    
Examples:
    python scripts/submit_model.py xgboost
    python scripts/submit_model.py catboost
    python scripts/submit_model.py lightgbm
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import submit_and_check
from config_local import local_config, model_config


def get_model_config(model_name):
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


def main():
    """Submit model and log score."""
    if len(sys.argv) < 2:
        print("=" * 70)
        print("MODEL SUBMISSION TO KAGGLE")
        print("=" * 70)
        print("\nUsage: python scripts/submit_model.py <model_name>")
        print("\nAvailable models:")
        print("  - xgboost, xgb")
        print("  - lightgbm, lgb")
        print("  - catboost, cat")
        print("  - ridge")
        print("  - lasso")
        print("  - elastic_net, elasticnet")
        print("  - random_forest, randomforest, rf")
        print("  - svr")
        print("  - blending")
        print("  - stacking")
        print("\nExample:")
        print("  python scripts/submit_model.py xgboost")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 70)
    print(f"MODEL SUBMISSION TO KAGGLE: {model_name.upper()}")
    print("=" * 70)
    
    # Get model config
    cfg, normalized_name = get_model_config(model_name)
    
    if cfg is None:
        print(f"\n[ERROR] Unknown model: {model_name}")
        print("   Use one of: xgboost, lightgbm, catboost, ridge, lasso, etc.")
        sys.exit(1)
    
    submission_path = local_config.get_model_submission_path(
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
        print(f"\nThis will:")
        print(f"   1. Run hyperparameter optimization")
        print(f"   2. Train final model with best parameters")
        print(f"   3. Generate predictions and save submission file")
        print(f"\nWould you like to:")
        print(f"  1. Run the model now (may take time)")
        print(f"  2. Exit and run manually")
        
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                print("\n" + "=" * 70)
                print(f"RUNNING {model_name.upper()} MODEL")
                print("=" * 70)
                import subprocess
                script_path = PROJECT_ROOT / "notebooks" / "Models" / script_name
                if not script_path.exists():
                    print(f"\n[ERROR] Model script not found: {script_path}")
                    sys.exit(1)
                
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(PROJECT_ROOT)
                )
                if result.returncode != 0:
                    print("\n[ERROR] Model training failed!")
                    sys.exit(1)
                
                # Check if file was created
                if not submission_path.exists():
                    print(f"\n[ERROR] Submission file still not found after training!")
                    sys.exit(1)
            else:
                print("\nExiting. Please run the model first, then run this script again.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            sys.exit(0)
    
    print(f"\nSubmission file found: {submission_path}")
    print(f"Model: {normalized_name}")
    print(f"Message: {normalized_name} model - Optuna optimized")
    print("-" * 70)
    
    # Submit to Kaggle
    result = submit_and_check(
        submission_file=str(submission_path),
        message=f"{normalized_name} model - Optuna optimized",
        wait_time=5,
        max_retries=3,
        skip_if_submitted=True
    )
    
    if result:
        if result.get('_skipped'):
            print("\n" + "=" * 70)
            print("[INFO] Submission was skipped (already submitted)")
            print("=" * 70)
            if result.get('score'):
                print(f"Previous score: {result['score']:.6f}")
                print(f"Previous rank: {result.get('rank', 'N/A')}")
        elif result.get('score'):
            print("\n" + "=" * 70)
            print("[SUCCESS] Submission complete and score logged!")
            print("=" * 70)
            print(f"Kaggle Score: {result['score']:.6f}")
            print(f"Rank: {result.get('rank', 'N/A')}")
            print(f"\nThe score has been automatically logged to:")
            print(f"  {local_config.MODEL_PERFORMANCE_CSV}")
        else:
            print("\n[WARNING] Submission completed but score not yet available.")
            print("   The score will be logged automatically once available.")
            print("   You can check later with:")
            print(f"   python scripts/get_kaggle_score.py {normalized_name}")
    else:
        print("\n[ERROR] Submission failed or was cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()

