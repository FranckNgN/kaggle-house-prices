#!/usr/bin/env python
"""
Interactive script to submit models to Kaggle.
Allows selecting individual models or submitting all at once.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import submit_and_check, load_submission_log
import config_local.local_config as config


def get_available_submissions() -> List[Dict]:
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
        # Resolve csv_file to absolute path to match PROJECT_ROOT
        csv_file_abs = csv_file.resolve()
        
        # Generate model name from filename or parent folder
        if csv_file.parent != submissions_dir:
            model_name = csv_file.parent.name.replace("_", " ").title()
        else:
            model_name = csv_file.stem.replace("_", " ").replace("Model", "").strip()
            
        if not model_name:
            model_name = csv_file.stem
        
        submissions.append({
            "file": str(csv_file_abs),
            "name": csv_file.name,
            "path": str(csv_file_abs.relative_to(PROJECT_ROOT)),
            "model": model_name
        })
    
    return submissions


def display_submissions(submissions: List[Dict]):
    """Display available submission files."""
    print("\n" + "=" * 70)
    print("AVAILABLE SUBMISSION FILES")
    print("=" * 70)
    for i, sub in enumerate(submissions, 1):
        print(f"  {i:2d}. {sub['name']:30s} ({sub['model']})")
    print("=" * 70)


def submit_single_model(submissions: List[Dict]) -> Dict:
    """Submit a single model selected by user."""
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
                    message = f"{selected['model']} model"
                
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


def submit_all_models(submissions: List[Dict]) -> List[Dict]:
    """Submit all models one by one."""
    print(f"\nSubmitting {len(submissions)} models to Kaggle...")
    print("=" * 70)
    
    results = []
    for i, sub in enumerate(submissions, 1):
        print(f"\n[{i}/{len(submissions)}] Processing: {sub['name']}")
        print("-" * 70)
        
        message = f"{sub['model']} model"
        result = submit_and_check(sub['file'], message, wait_time=8, max_retries=5)
        
        if result:
            results.append({
                "file": sub['name'],
                "model": sub['model'],
                "rank": result['rank'],
                "score": result['score']
            })
        
        # Wait between submissions to avoid rate limiting
        if i < len(submissions):
            print(f"\nWaiting 10 seconds before next submission...")
            import time
            time.sleep(10)
    
    return results


def display_summary(results: List[Dict]):
    """Display summary of all submissions."""
    if not results:
        return
    
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30s} {'Rank':<10s} {'Score':<15s}")
    print("-" * 70)
    
    # Sort by score (lower is better)
    sorted_results = sorted(results, key=lambda x: x['score'])
    
    for r in sorted_results:
        print(f"{r['model']:<30s} {r['rank']:<10d} {r['score']:<15.5f}")
    
    print("=" * 70)
    print(f"\nBest Model: {sorted_results[0]['model']}")
    print(f"   Score: {sorted_results[0]['score']:.5f}")
    print(f"   Rank: {sorted_results[0]['rank']}")


def main():
    """Main interactive menu."""
    print("=" * 70)
    print("KAGGLE MODEL SUBMISSION MANAGER")
    print("=" * 70)
    
    # Get available submissions
    submissions = get_available_submissions()
    
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
            result = submit_single_model(submissions)
            if result:
                input("\nPress Enter to continue...")
        
        elif choice == '2':
            confirm = input(f"\nThis will submit {len(submissions)} models. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                results = submit_all_models(submissions)
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
                    print(f"{entry['timestamp']:<20s} {entry['file']:<30s} {entry['rank']:<10d} {entry['score']:<15.5f}")
                print("=" * 70)
            else:
                print("\nNo submission history found.")
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

