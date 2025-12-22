#!/usr/bin/env python
"""
Consolidated Kaggle scores and leaderboard management script.

Usage:
    python -m kaggle.scores status        # Check submission status
    python -m kaggle.scores latest        # Get latest submission score
    python -m kaggle.scores sync          # Sync scores to performance CSV
    python -m kaggle.scores quantile      # Get leaderboard quantile
    python -m kaggle.scores leaderboard   # Get full leaderboard
"""

import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle import (
    COMPETITION_NAME,
    get_username,
    get_latest_submission_score,
    load_submission_log,
    get_available_submissions
)
from utils.metrics import log_kaggle_score
import config_local.local_config as config


def get_submitted_today() -> Dict:
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


def check_submission_status():
    """Display submission status - which models submitted and which are pending."""
    print("=" * 70)
    print("KAGGLE SUBMISSION STATUS")
    print("=" * 70)
    
    # Get available and submitted models
    submissions_list = get_available_submissions(PROJECT_ROOT)
    available = {}
    for sub in submissions_list:
        key = sub['name'].lower()
        available[key] = {
            "file": sub['name'],
            "path": sub['path'],
            "model": sub['model']
        }
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


def get_latest_score():
    """Get and log the latest Kaggle submission score."""
    print("=" * 70)
    print("FETCHING KAGGLE SUBMISSION SCORE")
    print("=" * 70)
    
    # Get latest submission score
    result = get_latest_submission_score()
    
    if result and result.get("score"):
        score = result["score"]
        print(f"\n[SUCCESS] Latest submission score: {score:.6f}")
        
        # Prompt for model name to log
        print("\nEnter model name to log this score (or press Enter to skip):")
        model_name = input("Model name: ").strip()
        
        if model_name:
            log_kaggle_score(model_name, score)
            print("\n" + "=" * 70)
            print("[SUCCESS] Score logged successfully!")
            print("=" * 70)
        else:
            print("\n[INFO] Score retrieved but not logged.")
    else:
        print("\n[WARNING] Could not retrieve score.")
        print("Possible reasons:")
        print("  - Submission is still processing")
        print("  - No submissions found")
        print("  - API authentication issue")
        sys.exit(1)


def normalize_model_name(file_name: str) -> str:
    """Normalize submission file name to model name in performance CSV."""
    name = file_name.replace(".csv", "").lower()
    
    # Map file names to model names
    name_mapping = {
        "catboost_model": "catboost",
        "xgboost_model": "xgboost",
        "lightgbm_model": "lightgbm",
        "ridge_model_kfold": "ridge",
        "lasso_model_kfold": "lasso",
        "elasticnet_model": "elastic_net",
        "randomforest_model": "random_forest",
        "svr_model": "svr",
        "blend_xgb_lgb_cat_model": "blending",
        "stacking_submission": "STACKING_META",
        "naive_lr": "linear_regression",
    }
    
    # Try exact match first
    if name in name_mapping:
        return name_mapping[name]
    
    # Try partial matches
    for key, value in name_mapping.items():
        if key in name:
            return value
    
    # Default: return as-is
    return name


def sync_scores():
    """Sync Kaggle scores from submission log to performance CSV."""
    print("=" * 70)
    print("SYNCING KAGGLE SCORES TO MODEL PERFORMANCE CSV")
    print("=" * 70)
    
    # Load submission log
    submission_log = load_submission_log()
    if not submission_log:
        print("[WARNING] No submission log found.")
        return
    
    print(f"\nLoaded {len(submission_log)} submission entries")
    
    # Load performance CSV
    perf_csv = config.MODEL_PERFORMANCE_CSV
    if not perf_csv.exists():
        print(f"[ERROR] Performance CSV not found: {perf_csv}")
        return
    
    df = pd.read_csv(perf_csv)
    
    # Ensure kaggle_score column exists
    if "kaggle_score" not in df.columns:
        df["kaggle_score"] = ""
    
    # Convert timestamp to datetime for comparison
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Track updates
    updates = []
    
    # Process each submission
    for entry in submission_log:
        file_name = entry.get("file", "")
        score = entry.get("score")
        timestamp_str = entry.get("timestamp", "")
        
        if not score or not timestamp_str:
            continue
        
        # Normalize model name
        model_name = normalize_model_name(file_name)
        
        # Parse timestamp
        try:
            sub_timestamp = pd.to_datetime(timestamp_str)
        except:
            continue
        
        # Find matching entries in performance CSV
        model_entries = df[df["model"] == model_name].copy()
        
        if len(model_entries) == 0:
            # Try alternative names
            alt_names = {
                "STACKING_META": "stacking",
                "blending": "blend",
            }
            if model_name in alt_names:
                model_entries = df[df["model"] == alt_names[model_name]].copy()
        
        if len(model_entries) == 0:
            print(f"[SKIP] No performance entry found for model: {model_name} (file: {file_name})")
            continue
        
        # Find the entry closest to submission timestamp
        model_entries["time_diff"] = (model_entries["timestamp"] - sub_timestamp).abs()
        closest_idx = model_entries["time_diff"].idxmin()
        
        # Check if score is already logged
        current_score = df.at[closest_idx, "kaggle_score"]
        if pd.notna(current_score) and float(current_score) == float(score):
            continue  # Already logged
        
        # Update the score
        df.at[closest_idx, "kaggle_score"] = round(float(score), 6)
        updates.append({
            "model": model_name,
            "file": file_name,
            "score": score,
            "timestamp": timestamp_str
        })
    
    # Save updated CSV
    if updates:
        df.to_csv(perf_csv, index=False)
        print(f"\n[SUCCESS] Updated {len(updates)} entries:")
        for update in updates:
            print(f"  - {update['model']}: {update['score']:.6f} (from {update['file']})")
    else:
        print("\n[INFO] No updates needed. All scores are already synced.")
    
    print("\n" + "=" * 70)


def get_leaderboard_quantile():
    """Get current leaderboard ranking and calculate quantile percent."""
    try:
        api = KaggleApi()
        api.authenticate()
        
        username = get_username()
        print("=" * 70)
        print("LEADERBOARD RANKING & QUANTILE")
        print("=" * 70)
        print(f"Competition: {COMPETITION_NAME}")
        print(f"Looking for: {username}")
        print("-" * 70)
        
        # Get leaderboard
        leaderboard = api.competition_leaderboard_view(COMPETITION_NAME)
        total_participants = len(leaderboard)
        
        if total_participants == 0:
            print("\n[ERROR] No participants found on leaderboard.")
            return None
        
        # Try to get the team name from submissions first
        team_names = {username.lower()}
        best_score = None
        try:
            subs = api.competition_submissions(COMPETITION_NAME)
            if subs:
                for sub in subs:
                    team_name = getattr(sub, 'teamName', None)
                    if team_name:
                        team_names.add(team_name.lower())
                        team_names.add(team_name.lower().replace(' ', ''))
                        team_names.add(team_name.lower().replace('_', ''))
                
                latest = subs[0]
                score = getattr(latest, 'publicScore', None) or getattr(latest, 'score', None)
                if score is not None:
                    best_score = float(score)
        except Exception as e:
            print(f"[INFO] Could not get team names from submissions: {e}")
        
        print(f"Trying to match team names: {team_names}")
        if best_score:
            print(f"Best submission score: {best_score:.6f}")
        
        # Find user's entry
        user_entry = None
        
        # Method 1: Match by team name
        for i, entry in enumerate(leaderboard):
            team_name = getattr(entry, 'teamName', '')
            entry_lower = team_name.lower()
            if (entry_lower in team_names or 
                entry_lower.replace(' ', '') in team_names or
                entry_lower.replace('_', '') in team_names):
                user_entry = {
                    "rank": i + 1,
                    "score": float(entry.score) if hasattr(entry, 'score') and entry.score else 0.0,
                    "username": team_name,
                    "submission_date": str(getattr(entry, 'submissionDate', 'unknown'))
                }
                break
        
        # Method 2: Match by score
        if not user_entry and best_score:
            print("\n[INFO] Not found by team name, trying to match by score...")
            for i, entry in enumerate(leaderboard):
                entry_score = float(entry.score) if hasattr(entry, 'score') and entry.score else None
                if entry_score and abs(entry_score - best_score) < 0.0001:
                    user_entry = {
                        "rank": i + 1,
                        "score": entry_score,
                        "username": getattr(entry, 'teamName', 'unknown'),
                        "submission_date": str(getattr(entry, 'submissionDate', 'unknown'))
                    }
                    break
        
        if not user_entry:
            print(f"\n[WARNING] No submission found for team names: {team_names}")
            print(f"Total participants on leaderboard: {total_participants}")
            return None
        
        # Calculate quantile percent
        rank = user_entry["rank"]
        quantile_percent = (1 - (rank - 1) / total_participants) * 100
        
        # Display results
        print(f"\n[SUCCESS] Found your submission!")
        print("-" * 70)
        print(f"Rank: {rank:,} out of {total_participants:,} participants")
        print(f"Score: {user_entry['score']:.6f}")
        print(f"Username: {user_entry['username']}")
        print(f"Submission Date: {user_entry['submission_date']}")
        print("-" * 70)
        print(f"\nQuantile Percent: {quantile_percent:.4f}%")
        print(f"\nInterpretation:")
        print(f"  You are better than {quantile_percent:.2f}% of participants")
        print(f"  You are in the top {100 - quantile_percent:.2f}%")
        print("=" * 70)
        
        return {
            "rank": rank,
            "total_participants": total_participants,
            "score": user_entry["score"],
            "quantile_percent": quantile_percent,
        }
        
    except Exception as e:
        print(f"\n[ERROR] Error fetching leaderboard: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_full_leaderboard():
    """Download and display full leaderboard."""
    try:
        api = KaggleApi()
        api.authenticate()
        
        print("=" * 70)
        print("FULL LEADERBOARD")
        print("=" * 70)
        print(f"Competition: {COMPETITION_NAME}")
        print("-" * 70)
        
        leaderboard = api.competition_leaderboard_view(COMPETITION_NAME)
        total_participants = len(leaderboard)
        
        if total_participants == 0:
            print("\n[ERROR] No participants found on leaderboard.")
            return
        
        print(f"\nTotal participants shown: {total_participants}")
        print(f"{'Rank':<6} {'Team Name':<30} {'Score':<15}")
        print("-" * 70)
        
        for i, entry in enumerate(leaderboard[:50]):  # Show top 50
            team_name = getattr(entry, 'teamName', 'unknown')
            score = getattr(entry, 'score', 'unknown')
            print(f"{i+1:<6} {team_name:<30} {score}")
        
        if total_participants > 50:
            print(f"\n... and {total_participants - 50} more participants")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Error fetching leaderboard: {e}")
        import traceback
        traceback.print_exc()


def log_score(model_name: str, score: float):
    """Manually log a Kaggle score to performance CSV."""
    from utils.metrics import log_kaggle_score
    
    print("=" * 70)
    print(f"LOGGING KAGGLE SCORE FOR {model_name.upper()}")
    print("=" * 70)
    
    # Map model name for logging
    model_name_for_log = "STACKING_META" if model_name.lower() == "stacking" else model_name
    
    print(f"\nModel: {model_name_for_log}")
    print(f"Score: {score:.6f}")
    
    log_kaggle_score(model_name_for_log, score)
    
    print("\n[SUCCESS] Score logged successfully!")
    print(f"Logged to: {config.MODEL_PERFORMANCE_CSV}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage Kaggle scores and leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kaggle.scores status        # Check submission status
  python -m kaggle.scores latest        # Get latest submission score
  python -m kaggle.scores sync          # Sync scores to performance CSV
  python -m kaggle.scores quantile      # Get leaderboard quantile
  python -m kaggle.scores leaderboard   # Get full leaderboard
  python -m kaggle.scores log catboost 0.123456  # Manually log a score
        """
    )
    
    parser.add_argument(
        'command',
        choices=['status', 'latest', 'sync', 'quantile', 'leaderboard', 'log'],
        help='Command to execute'
    )
    parser.add_argument(
        'model_name',
        nargs='?',
        help='Model name (for log command)'
    )
    parser.add_argument(
        'score',
        nargs='?',
        type=float,
        help='Score value (for log command)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'status':
            check_submission_status()
        elif args.command == 'latest':
            get_latest_score()
        elif args.command == 'sync':
            sync_scores()
        elif args.command == 'quantile':
            get_leaderboard_quantile()
        elif args.command == 'leaderboard':
            get_full_leaderboard()
        elif args.command == 'log':
            if not args.model_name or args.score is None:
                print("[ERROR] log command requires model_name and score")
                print("Usage: python -m kaggle.scores log <model_name> <score>")
                print("Example: python -m kaggle.scores log catboost 0.123456")
                sys.exit(1)
            log_score(args.model_name, args.score)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

