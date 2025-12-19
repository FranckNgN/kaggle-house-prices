#!/usr/bin/env python
"""
Retrieve and log the latest Kaggle submission score for a model.
This can be run after a submission to automatically get and log the score.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import get_latest_submission_score
from utils.metrics import log_kaggle_score


def main():
    """Get latest Kaggle score and optionally log it to a model."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/get_kaggle_score.py <model_name>")
        print("\nExample:")
        print("  python scripts/get_kaggle_score.py xgboost")
        print("\nThis will:")
        print("  1. Fetch the latest Kaggle submission score")
        print("  2. Update the model performance log with the score")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    print("=" * 70)
    print("FETCHING KAGGLE SUBMISSION SCORE")
    print("=" * 70)
    
    # Get latest submission score
    result = get_latest_submission_score()
    
    if result and result.get("score"):
        score = result["score"]
        print(f"\n[SUCCESS] Latest submission score: {score:.6f}")
        
        # Log to model performance
        print(f"\nUpdating model performance log for '{model_name}'...")
        log_kaggle_score(model_name, score)
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Score logged successfully!")
        print("=" * 70)
    else:
        print("\n[WARNING] Could not retrieve score.")
        print("Possible reasons:")
        print("  - Submission is still processing")
        print("  - No submissions found")
        print("  - API authentication issue")
        sys.exit(1)


if __name__ == "__main__":
    main()

