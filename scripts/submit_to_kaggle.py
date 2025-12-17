#!/usr/bin/env python
"""Submit model predictions to Kaggle and check leaderboard ranking."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import submit_and_check


def main():
    """Main entry point for Kaggle submission."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/submit_to_kaggle.py <submission_file.csv> [message]")
        print("\nExample:")
        print("  python scripts/submit_to_kaggle.py data/submissions/elasticNetModel.csv \"ElasticNet model\"")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else "Model submission"
    
    result = submit_and_check(submission_file, message)
    
    if result:
        print("\n✅ Submission complete! Check the results above.")
    else:
        print("\n❌ Submission failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

