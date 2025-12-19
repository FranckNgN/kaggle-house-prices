#!/usr/bin/env python
"""Submit model predictions to Kaggle and check leaderboard ranking."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import submit_and_check, has_been_submitted, check_submission_limit


def main():
    """Main entry point for Kaggle submission."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/submit_to_kaggle.py <submission_file.csv> [message]")
        print("\nExample:")
        print("  python scripts/submit_to_kaggle.py data/submissions/elasticNetModel.csv \"ElasticNet model\"")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else "Model submission"
    
    # Check submission limit first
    limit_info = check_submission_limit()
    if limit_info.get('at_limit', False):
        print(f"\n[WARNING] Daily submission limit reached!")
        print(f"   Submitted today: {limit_info.get('submitted_today', 'unknown')}/{limit_info.get('limit', 10)}")
        print(f"   Please wait until tomorrow (UTC) to submit more.")
        sys.exit(1)
    
    # Check if already submitted
    existing = has_been_submitted(submission_file, check_hash=True)
    if existing:
        print(f"\n[INFO] This file has already been submitted!")
        print(f"   Previous submission: {existing.get('timestamp', 'unknown')}")
        print(f"   Previous score: {existing.get('score', 'N/A')}")
        print(f"\nTo force resubmission, you can modify the file or wait until tomorrow.")
        sys.exit(0)
    
    result = submit_and_check(submission_file, message, skip_if_submitted=True)
    
    if result:
        if result.get('_skipped', False):
            print("\n[INFO] Submission skipped (already submitted).")
        else:
            print("\n[SUCCESS] Submission complete! Check the results above.")
    else:
        print("\n[FAILED] Submission failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

