"""Kaggle API package for competition submissions and leaderboard management."""
# Main exports for easy importing
from .constants import COMPETITION_NAME
from .api import get_username
from .submissions import (
    submit_and_check,
    has_been_submitted,
    load_submission_log,
    save_submission_log,
    get_available_submissions
)
from .leaderboard import (
    check_leaderboard,
    get_latest_submission_score,
    check_submission_limit
)

__all__ = [
    # Constants
    'COMPETITION_NAME',
    # API
    'get_username',
    # Submissions
    'submit_and_check',
    'has_been_submitted',
    'load_submission_log',
    'save_submission_log',
    'get_available_submissions',
    # Leaderboard
    'check_leaderboard',
    'get_latest_submission_score',
    'check_submission_limit',
]

