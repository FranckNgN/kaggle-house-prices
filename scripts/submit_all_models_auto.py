#!/usr/bin/env python
"""
Automatically submit all available models to Kaggle and retrieve their scores.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import submit_and_check, has_been_submitted, check_submission_limit
import config_local.local_config as config


def get_available_submissions():
    """Get list of available submission CSV files (searching recursively)."""
    submissions_dir = config.SUBMISSIONS_DIR
    if not submissions_dir.exists():
        return []
    
    # Exclude sample_submission.csv, search recursively
    csv_files = [
        f for f in submissions_dir.rglob("*.csv")
        if f.name != "sample_submission.csv"
    ]
    
    submissions = []
    for csv_file in sorted(csv_files):
        # Generate model name from filename or parent folder
        if csv_file.parent != submissions_dir:
            model_name = csv_file.parent.name.replace("_", " ").title()
        else:
            model_name = csv_file.stem.replace("_", " ").replace("Model", "").strip()
            
        if not model_name:
            model_name = csv_file.stem
        
        # Get relative path from project root
        try:
            rel_path = str(csv_file.relative_to(PROJECT_ROOT))
        except ValueError:
            # If relative_to fails, use the path as-is
            rel_path = str(csv_file)
        
        submissions.append({
            "file": str(csv_file.resolve()),
            "name": csv_file.name,
            "path": rel_path,
            "model": model_name
        })
    
    return submissions


def main():
    """Submit all models automatically."""
    print("=" * 70)
    print("AUTOMATIC SUBMISSION OF ALL MODELS TO KAGGLE")
    print("=" * 70)
    
    # Get available submissions
    submissions = get_available_submissions()
    
    if not submissions:
        print("No submission files found in data/submissions/")
        sys.exit(1)
    
    print(f"\nFound {len(submissions)} submission files:")
    for i, sub in enumerate(submissions, 1):
        print(f"  {i:2d}. {sub['name']:30s} ({sub['model']})")
    
    # Check submission limit first
    print("\n" + "=" * 70)
    print("CHECKING SUBMISSION STATUS")
    print("=" * 70)
    limit_info = check_submission_limit()
    print(f"Submissions today: {limit_info.get('submitted_today', 'unknown')}/{limit_info.get('limit', 10)}")
    print(f"Remaining: {limit_info.get('remaining', 'unknown')}")
    
    if limit_info.get('at_limit', False):
        print("\n[WARNING] Daily submission limit reached!")
        print("   Please wait until tomorrow (UTC) to submit more.")
        sys.exit(0)
    
    # Filter out already-submitted models
    print("\n" + "=" * 70)
    print("CHECKING FOR PREVIOUS SUBMISSIONS")
    print("=" * 70)
    pending_submissions = []
    skipped_submissions = []
    
    for sub in submissions:
        existing = has_been_submitted(sub['path'], check_hash=True)
        if existing:
            skipped_submissions.append({
                "model": sub['model'],
                "file": sub['name'],
                "score": existing.get('score', 'N/A'),
                "timestamp": existing.get('timestamp', 'unknown')
            })
        else:
            pending_submissions.append(sub)
    
    if skipped_submissions:
        print(f"\n[Skipping {len(skipped_submissions)} already-submitted model(s):]")
        for skip in skipped_submissions:
            score_str = f"{skip['score']:.6f}" if isinstance(skip['score'], (int, float)) else str(skip['score'])
            print(f"  - {skip['model']:30s} (Score: {score_str}, Submitted: {skip['timestamp']})")
    
    if not pending_submissions:
        print("\n[INFO] All models have already been submitted!")
        print("   No new submissions to make.")
        sys.exit(0)
    
    print(f"\n[Submitting {len(pending_submissions)} new model(s)...]")
    
    print("\n" + "=" * 70)
    print("STARTING SUBMISSIONS")
    print("=" * 70)
    
    results = []
    for i, sub in enumerate(pending_submissions, 1):
        print(f"\n[{i}/{len(pending_submissions)}] Submitting: {sub['name']}")
        print("-" * 70)
        
        message = f"{sub['model']} model - automatic submission"
        result = submit_and_check(sub['path'], message, wait_time=8, max_retries=5, skip_if_submitted=True)
        
        if result:
            # Check if it was skipped (already submitted) or newly submitted
            if result.get('_skipped', False):
                print(f"[SKIPPED] {sub['model']}: Already submitted (Score: {result.get('score', 'N/A'):.6f})")
                results.append({
                    "file": sub['name'],
                    "model": sub['model'],
                    "rank": result.get('rank', 'unknown'),
                    "score": result.get('score', 'N/A'),
                    "status": "skipped"
                })
            else:
                results.append({
                    "file": sub['name'],
                    "model": sub['model'],
                    "rank": result.get('rank', 'unknown'),
                    "score": result.get('score', 'N/A'),
                    "status": "new"
                })
                score_val = result.get('score', 'N/A')
                if isinstance(score_val, (int, float)):
                    print(f"[SUCCESS] {sub['model']}: Score = {score_val:.6f}")
                else:
                    print(f"[SUCCESS] {sub['model']}: Submitted (score pending)")
        else:
            results.append({
                "file": sub['name'],
                "model": sub['model'],
                "rank": "failed",
                "score": "N/A",
                "status": "failed"
            })
            print(f"[FAILED] {sub['model']}: Could not retrieve score")
        
        # Wait between submissions to avoid rate limiting
        if i < len(submissions):
            print(f"\nWaiting 10 seconds before next submission...")
            time.sleep(10)
    
    # Display summary
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35s} {'Score':<15s} {'Status':<10s}")
    print("-" * 70)
    
    # Sort by score (lower is better, but handle N/A)
    def sort_key(x):
        score = x['score']
        if isinstance(score, (int, float)):
            return (0, score)  # Valid score
        return (1, 0)  # N/A scores go to end
    
    sorted_results = sorted(results, key=sort_key)
    
    for r in sorted_results:
        score_str = f"{r['score']:.6f}" if isinstance(r['score'], (int, float)) else str(r['score'])
        status = r.get('status', 'unknown').upper()
        if status == 'SKIPPED':
            status = "SKIPPED (already submitted)"
        elif status == 'NEW':
            status = "SUCCESS" if isinstance(r['score'], (int, float)) else "PENDING"
        print(f"{r['model']:<35s} {score_str:<15s} {status:<20s}")
    
    print("=" * 70)
    
    # Show best model
    valid_results = [r for r in results if isinstance(r['score'], (int, float))]
    if valid_results:
        best = min(valid_results, key=lambda x: x['score'])
        print(f"\nBest Model: {best['model']}")
        print(f"   Score: {best['score']:.6f}")
        print("=" * 70)
    
    print("\n[SUCCESS] All submissions complete!")
    print(f"   Scores have been automatically logged to {config.MODEL_PERFORMANCE_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Submission cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

