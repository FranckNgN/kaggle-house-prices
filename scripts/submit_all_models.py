#!/usr/bin/env python
"""
Interactive script to submit models to Kaggle.
Allows selecting individual models or submitting all at once.

Usage:
    python scripts/submit_all_models.py              # Interactive mode
    python scripts/submit_all_models.py --auto        # Automatic mode (non-interactive)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.kaggle_helper import submit_and_check, load_submission_log, get_available_submissions, has_been_submitted, check_submission_limit
import config_local.local_config as config


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




def display_summary(results: List[Dict]):
    """Display summary of all submissions."""
    if not results:
        return
    
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30s} {'Rank':<10s} {'Score':<15s}")
    print("-" * 70)
    
    # Sort by score (lower is better, but handle N/A)
    def sort_key(x):
        score = x.get('score', 'N/A')
        if isinstance(score, (int, float)):
            return (0, score)
        return (1, 0)
    
    sorted_results = sorted(results, key=sort_key)
    
    for r in sorted_results:
        score_str = f"{r['score']:.6f}" if isinstance(r.get('score'), (int, float)) else str(r.get('score', 'N/A'))
        rank_str = str(r.get('rank', 'N/A'))
        print(f"{r['model']:<30s} {rank_str:<10s} {score_str:<15s}")
    
    print("=" * 70)
    
    # Show best model if we have valid scores
    valid_results = [r for r in results if isinstance(r.get('score'), (int, float))]
    if valid_results:
        best = min(valid_results, key=lambda x: x['score'])
        print(f"\nBest Model: {best['model']}")
        print(f"   Score: {best['score']:.5f}")
        if isinstance(best.get('rank'), int):
            print(f"   Rank: {best['rank']}")


def main():
    """Main interactive menu."""
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
            result = submit_single_model(submissions)
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

