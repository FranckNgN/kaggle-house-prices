#!/usr/bin/env python
"""Kaggle API helper functions for submissions and leaderboard checking."""

import json
import time
from pathlib import Path
from typing import Optional, Dict, List
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION_NAME = "house-prices-advanced-regression-techniques"
SUBMISSION_LOG = Path("data/submissions/submission_log.json")


def get_username() -> str:
    """Get Kaggle username from kaggle.json."""
    kaggle_json_path = Path(".kaggle/kaggle.json")
    if kaggle_json_path.exists():
        with open(kaggle_json_path) as f:
            kaggle_config = json.load(f)
            return kaggle_config.get("username", "unknown")
    return "unknown"


def load_submission_log() -> List[Dict]:
    """Load submission history from log file."""
    if SUBMISSION_LOG.exists():
        with open(SUBMISSION_LOG) as f:
            return json.load(f)
    return []


def save_submission_log(log: List[Dict]):
    """Save submission history to log file."""
    SUBMISSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBMISSION_LOG, 'w') as f:
        json.dump(log, f, indent=2)


def check_leaderboard(api: Optional[KaggleApi] = None) -> Optional[Dict]:
    """
    Check current leaderboard ranking and score.
    
    Returns:
        Dict with rank, score, username, and submission_date, or None if error
    """
    try:
        if api is None:
            api = KaggleApi()
            api.authenticate()
        
        username = get_username()
        print(f"Fetching leaderboard for '{COMPETITION_NAME}'...")
        print(f"   Looking for: {username}")
        
        # Try to get the team name from submissions first, as it might differ from username
        team_names = {username.lower()}
        try:
            subs = api.competition_submissions(COMPETITION_NAME)
            if subs:
                latest_team = getattr(subs[0], 'teamName', None)
                if latest_team:
                    team_names.add(latest_team.lower())
                total_submissions = len(subs)
            else:
                total_submissions = 0
        except:
            total_submissions = "unknown"
        
        # Use competition_leaderboard_view which returns a list of LeaderboardEntry objects
        leaderboard = api.competition_leaderboard_view(COMPETITION_NAME)
        
        # Find user's entry
        user_entry = None
        for i, entry in enumerate(leaderboard):
            team_name = getattr(entry, 'teamName', '')
            if team_name.lower() in team_names:
                user_entry = {
                    "rank": i + 1,
                    "score": float(entry.score) if hasattr(entry, 'score') and entry.score else 0.0,
                    "username": team_name,
                    "submission_date": str(getattr(entry, 'submissionDate', 'unknown')),
                    "total_submissions": total_submissions
                }
                break
        
        if user_entry:
            print(f"\nFound your submission!")
            return user_entry
        else:
            print(f"\nNo submission found for team names: {team_names}")
            if leaderboard:
                print("\nLeaderboard Top 5:")
                print("-" * 60)
                for i, entry in enumerate(leaderboard[:5]):
                    print(f"  {i+1:4d}. {getattr(entry, 'teamName', 'unknown'):20s} - Score: {getattr(entry, 'score', 'unknown')}")
            return None
            
    except Exception as e:
        print(f"Error checking leaderboard: {e}")
        return None


def submit_and_check(
    submission_file: str,
    message: str = "Model submission",
    wait_time: int = 5,
    max_retries: int = 3
) -> Optional[Dict]:
    """
    Submit a file to Kaggle and check leaderboard ranking.
    
    Args:
        submission_file: Path to submission CSV file
        message: Submission message
        wait_time: Seconds to wait before checking leaderboard
        max_retries: Maximum number of retries to check leaderboard
        
    Returns:
        Dict with submission results, or None if error
    """
    submission_path = Path(submission_file)
    
    if not submission_path.exists():
        print(f"File not found: {submission_file}")
        return None
    
    print("=" * 70)
    print("KAGGLE SUBMISSION")
    print("=" * 70)
    print(f"File: {submission_path.name}")
    print(f"Message: {message}")
    print(f"Path: {submission_path}")
    print("-" * 70)
    
    try:
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        # Submit
        print("Submitting to Kaggle...")
        api.competition_submit(
            file_name=str(submission_path),
            message=message,
            competition=COMPETITION_NAME
        )
        print("Submission successful!")
        print(f"Waiting {wait_time} seconds for processing...")
        time.sleep(wait_time)
        
        # Check leaderboard with retries
        print("\n" + "=" * 70)
        print("CHECKING LEADERBOARD")
        print("=" * 70)
        
        result = None
        for attempt in range(1, max_retries + 1):
            print(f"\nAttempt {attempt}/{max_retries}...")
            result = check_leaderboard(api)
            
            if result:
                break
            elif attempt < max_retries:
                print(f"Waiting {wait_time} more seconds...")
                time.sleep(wait_time)
        
        if result:
            # Load and update submission log
            log = load_submission_log()
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": submission_path.name,
                "message": message,
                "rank": result["rank"],
                "score": result["score"],
                "total_submissions": result["total_submissions"]
            }
            log.append(log_entry)
            save_submission_log(log)
            
            # Get previous best
            previous_best = None
            if len(log) > 1:
                previous_best = min(log[:-1], key=lambda x: x["score"])
            
            # Display results
            print("\n" + "=" * 70)
            print("SUBMISSION RESULTS")
            print("=" * 70)
            print(f"Current Rank: {result['rank']}")
            print(f"Current Score: {result['score']:.5f} (RMSLE)")
            print(f"Total Submissions: {result['total_submissions']}")
            print(f"Submission Date: {result['submission_date']}")
            
            if previous_best:
                print("\n" + "-" * 70)
                print("COMPARISON WITH PREVIOUS BEST")
                print("-" * 70)
                print(f"Previous Best Score: {previous_best['score']:.5f}")
                print(f"Previous Best Rank: {previous_best['rank']}")
                
                if result['score'] < previous_best['score']:
                    improvement = previous_best['score'] - result['score']
                    print(f"\nIMPROVEMENT! Better by {improvement:.5f}")
                elif result['score'] > previous_best['score']:
                    worse = result['score'] - previous_best['score']
                    print(f"\nNot an improvement. Worse by {worse:.5f}")
                else:
                    print(f"\nSame score as previous best")
            
            print("=" * 70)
            
            return result
        else:
            print("\nCould not retrieve leaderboard results.")
            print("   Check manually on Kaggle website.")
            return None
            
    except Exception as e:
        print(f"\nError during submission: {e}")
        print("\nTroubleshooting:")
        print("1. Check your kaggle.json credentials")
        print("2. Verify internet connection")
        print("3. Ensure submission file format is correct")
        return None


if __name__ == "__main__":
    # Test leaderboard check
    check_leaderboard()

