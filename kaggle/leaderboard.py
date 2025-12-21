"""Leaderboard-related functions for Kaggle competitions."""
from datetime import datetime, timezone
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # For type hints only
    from kaggle.api.kaggle_api_extended import KaggleApi
else:
    # Lazy import - will be imported when actually needed
    KaggleApi = None  # type: ignore

from .constants import COMPETITION_NAME
from .api import get_username


def _get_kaggle_api():
    """Lazy import helper to get KaggleApi from installed package."""
    global KaggleApi
    if KaggleApi is None:
        try:
            # Import from installed kaggle-api package
            import importlib
            import sys
            
            # Save current kaggle module state (except 'kaggle' itself)
            saved_modules = {}
            for key in list(sys.modules.keys()):
                if key.startswith('kaggle') and key != 'kaggle':
                    saved_modules[key] = sys.modules.pop(key)
            
            # Import from installed package
            kaggle_api_module = importlib.import_module('kaggle.api.kaggle_api_extended')
            KaggleApi = kaggle_api_module.KaggleApi
            
            # Restore modules
            sys.modules.update(saved_modules)
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "kaggle-api package not installed. Install it with: pip install kaggle"
            )
    return KaggleApi


def check_submission_limit(api: Optional[KaggleApi] = None) -> Dict:
    """
    Check how many submissions have been made today and remaining limit.
    
    Returns:
        Dict with 'submitted_today', 'limit', 'remaining', 'reset_time'
    """
    try:
        if api is None:
            KaggleApiClass = _get_kaggle_api()
            api = KaggleApiClass()
            api.authenticate()
        
        # Get all submissions
        submissions = api.competition_submissions(COMPETITION_NAME)
        
        # Count submissions today (UTC)
        today = datetime.now(timezone.utc).date()
        submitted_today = sum(
            1 for sub in submissions
            if hasattr(sub, 'date') and sub.date.date() == today
        )
        
        limit = 10  # Kaggle's daily limit
        remaining = max(0, limit - submitted_today)
        
        return {
            'submitted_today': submitted_today,
            'limit': limit,
            'remaining': remaining,
            'at_limit': remaining == 0
        }
    except Exception as e:
        print(f"[WARNING] Could not check submission limit: {e}")
        return {
            'submitted_today': 'unknown',
            'limit': 10,
            'remaining': 'unknown',
            'at_limit': False
        }


def get_latest_submission_score(api: Optional[KaggleApi] = None) -> Optional[Dict]:
    """
    Get the score of the most recent submission directly from submissions list.
    This is more reliable than checking the leaderboard.
    
    Returns:
        Dict with score, submission_date, and status, or None if error
    """
    try:
        if api is None:
            KaggleApiClass = _get_kaggle_api()
            api = KaggleApiClass()
            api.authenticate()
        
        print(f"Fetching latest submission for '{COMPETITION_NAME}'...")
        
        # Get submissions list - this includes scores even if not on leaderboard
        subs = api.competition_submissions(COMPETITION_NAME)
        
        if not subs:
            print("No submissions found.")
            return None
        
        # Get the most recent submission (first in list)
        latest = subs[0]
        
        # Extract score and status
        score = getattr(latest, 'publicScore', None) or getattr(latest, 'score', None)
        status = getattr(latest, 'status', 'unknown')
        submission_date = getattr(latest, 'date', 'unknown')
        description = getattr(latest, 'description', '')
        
        if score is None:
            # Score might not be available yet
            print(f"Submission status: {status}")
            print("Score not yet available. Submission may still be processing.")
            return None
        
        result = {
            "score": float(score),
            "status": status,
            "submission_date": str(submission_date),
            "description": description
        }
        
        print(f"Latest submission score: {result['score']:.6f}")
        print(f"Status: {result['status']}")
        
        return result
        
    except Exception as e:
        print(f"Error fetching submission score: {e}")
        return None


def check_leaderboard(api: Optional[KaggleApi] = None) -> Optional[Dict]:
    """
    Check current leaderboard ranking and score.
    
    Returns:
        Dict with rank, score, username, and submission_date, or None if error
    """
    try:
        if api is None:
            KaggleApiClass = _get_kaggle_api()
            api = KaggleApiClass()
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

