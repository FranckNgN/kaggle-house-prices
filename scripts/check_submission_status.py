#!/usr/bin/env python
"""
Check which models have been submitted to Kaggle and which are pending.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config_local.local_config as config
from utils.kaggle_helper import load_submission_log, get_available_submissions


def get_available_submissions_dict():
    """Get list of available submission CSV files as a dict (for backward compatibility)."""
    submissions_list = get_available_submissions(PROJECT_ROOT)
    submissions = {}
    for sub in submissions_list:
        # Normalize model name for comparison
        key = sub['name'].lower()
        submissions[key] = {
            "file": sub['name'],
            "path": sub['path'],
            "model": sub['model']
        }
    return submissions


def get_submitted_today():
    """Get models submitted today."""
    log = load_submission_log()
    today = datetime.now().strftime("%Y-%m-%d")
    
    submitted = {}
    for entry in log:
        if entry['timestamp'].startswith(today):
            file_key = entry['file'].lower()
            submitted[file_key] = {
                "file": entry['file'],
                "score": entry.get('score', 'N/A'),
                "timestamp": entry['timestamp']
            }
    
    return submitted


def main():
    """Display submission status."""
    print("=" * 70)
    print("KAGGLE SUBMISSION STATUS")
    print("=" * 70)
    
    # Get available and submitted models
    available = get_available_submissions_dict()
    submitted_today = get_submitted_today()
    
    print(f"\nTotal available models: {len(available)}")
    print(f"Submitted today: {len(submitted_today)}")
    print(f"Remaining: {len(available) - len(submitted_today)}")
    
    # Categorize models
    submitted_list = []
    pending_list = []
    
    for file_key, model_info in available.items():
        if file_key in submitted_today:
            submitted_list.append({
                "model": model_info["model"],
                "file": model_info["file"],
                "score": submitted_today[file_key]["score"],
                "timestamp": submitted_today[file_key]["timestamp"]
            })
        else:
            pending_list.append(model_info)
    
    # Display submitted models
    if submitted_list:
        print("\n" + "=" * 70)
        print("SUBMITTED TODAY (with scores)")
        print("=" * 70)
        print(f"{'Model':<30s} {'File':<30s} {'Score':<15s} {'Time':<15s}")
        print("-" * 70)
        
        # Sort by score (lower is better)
        sorted_submitted = sorted(
            [s for s in submitted_list if isinstance(s['score'], (int, float))],
            key=lambda x: x['score']
        )
        # Add non-numeric scores at the end
        other_submitted = [s for s in submitted_list if not isinstance(s['score'], (int, float))]
        sorted_submitted.extend(other_submitted)
        
        for s in sorted_submitted:
            score_str = f"{s['score']:.6f}" if isinstance(s['score'], (int, float)) else str(s['score'])
            time_str = s['timestamp'].split()[1] if ' ' in s['timestamp'] else s['timestamp']
            print(f"{s['model']:<30s} {s['file']:<30s} {score_str:<15s} {time_str:<15s}")
        
        if sorted_submitted:
            best = sorted_submitted[0]
            if isinstance(best['score'], (int, float)):
                print("\n" + "-" * 70)
                print(f"Best Model Today: {best['model']}")
                print(f"   Score: {best['score']:.6f}")
                print(f"   File: {best['file']}")
    
    # Display pending models
    if pending_list:
        print("\n" + "=" * 70)
        print("PENDING SUBMISSION")
        print("=" * 70)
        print(f"{'Model':<30s} {'File':<30s}")
        print("-" * 70)
        for p in sorted(pending_list, key=lambda x: x['model']):
            print(f"{p['model']:<30s} {p['file']:<30s}")
    
    print("\n" + "=" * 70)
    print("NOTE: Kaggle allows 10 submissions per day (UTC)")
    print("=" * 70)
    
    if len(submitted_today) >= 10:
        print("\n[WARNING] Daily submission limit reached!")
        print("   Please wait until tomorrow (UTC) to submit remaining models.")
    elif pending_list:
        print(f"\n[INFO] {len(pending_list)} model(s) can still be submitted today.")


if __name__ == "__main__":
    main()

