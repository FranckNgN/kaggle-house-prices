"""Submission-related functions for Kaggle competitions."""
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    # For type hints only
    from kaggle.api.kaggle_api_extended import KaggleApi
else:
    # Lazy import - will be imported when actually needed
    KaggleApi = None  # type: ignore

import config_local.local_config as config
from .constants import COMPETITION_NAME, SUBMISSION_LOG
from .api import get_username
from .leaderboard import check_submission_limit, check_leaderboard, get_latest_submission_score
from .utils import compute_file_hash


def get_available_submissions(project_root: Optional[Path] = None) -> List[Dict]:
    """
    Get list of available submission CSV files (searching recursively).
    Enhanced with better error handling and validation.
    
    Args:
        project_root: Optional project root path. If None, uses config.SUBMISSIONS_DIR parent.
        
    Returns:
        List of dicts with 'file', 'name', 'path', and 'model' keys
    """
    if project_root is None:
        # Try to infer from config path
        project_root = config.SUBMISSIONS_DIR.parent.parent
    
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
        try:
            # Resolve csv_file to absolute path
            csv_file_abs = csv_file.resolve()
            
            # Validate file is readable
            if not csv_file_abs.is_file():
                continue
            
            # Generate model name from filename or parent folder
            if csv_file.parent != submissions_dir:
                model_name = csv_file.parent.name.replace("_", " ").title()
            else:
                model_name = csv_file.stem.replace("_", " ").replace("Model", "").strip()
                
            if not model_name:
                model_name = csv_file.stem
            
            # Get relative path from project root
            try:
                rel_path = str(csv_file_abs.relative_to(project_root))
            except ValueError:
                # If relative_to fails, use the path as-is
                rel_path = str(csv_file_abs)
            
            submissions.append({
                "file": str(csv_file_abs),
                "name": csv_file.name,
                "path": rel_path,
                "model": model_name
            })
        except Exception as e:
            # Skip files that can't be processed
            print(f"[WARNING] Skipping file {csv_file}: {e}")
            continue
    
    return submissions


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


def has_been_submitted(submission_file: str, check_hash: bool = True) -> Optional[Dict]:
    """
    Check if a submission file has already been submitted.
    
    Args:
        submission_file: Path to submission CSV file
        check_hash: If True, also check file hash to detect if file changed
        
    Returns:
        Dict with submission info if already submitted, None otherwise
    """
    submission_path = Path(submission_file)
    if not submission_path.exists():
        return None
    
    log = load_submission_log()
    filename = submission_path.name
    
    # Check by filename first
    for entry in reversed(log):  # Check most recent first
        if entry.get('file') == filename:
            # If checking hash, verify file hasn't changed
            if check_hash and 'file_hash' in entry:
                current_hash = compute_file_hash(submission_path)
                if entry['file_hash'] == current_hash:
                    return entry
                else:
                    # File changed, not the same submission
                    return None
            else:
                # Found by filename, return it
                return entry
    
    return None


def _get_kaggle_api():
    """Lazy import helper to get KaggleApi from installed package."""
    global KaggleApi
    if KaggleApi is None:
        try:
            # Import from installed kaggle-api package
            import importlib
            import sys
            
            # Save current kaggle module state
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


def submit_and_check(
    submission_file: str,
    message: str = "Model submission",
    wait_time: int = 5,
    max_retries: int = 3,
    skip_if_submitted: bool = True
) -> Optional[Dict]:
    """
    Submit a file to Kaggle and check leaderboard ranking.
    
    Args:
        submission_file: Path to submission CSV file
        message: Submission message
        wait_time: Seconds to wait before checking leaderboard
        max_retries: Maximum number of retries to check leaderboard
        skip_if_submitted: If True, skip submission if file was already submitted
        
    Returns:
        Dict with submission results, or None if error or skipped
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
        KaggleApiClass = _get_kaggle_api()
        api = KaggleApiClass()
        api.authenticate()
        
        # Check if already submitted
        if skip_if_submitted:
            existing = has_been_submitted(submission_file, check_hash=True)
            if existing:
                print(f"\n[INFO] This file has already been submitted!")
                print(f"   Previous submission: {existing.get('timestamp', 'unknown')}")
                print(f"   Previous score: {existing.get('score', 'N/A')}")
                print(f"   Skipping to avoid wasting daily submission limit.")
                # Mark as skipped in the result
                existing['_skipped'] = True
                return existing
        
        # Check submission limit
        limit_info = check_submission_limit(api)
        if limit_info.get('at_limit', False):
            print(f"\n[WARNING] Daily submission limit reached!")
            print(f"   Submitted today: {limit_info.get('submitted_today', 'unknown')}/{limit_info.get('limit', 10)}")
            print(f"   Please wait until tomorrow (UTC) to submit more.")
            return None
        elif limit_info.get('remaining') != 'unknown':
            remaining = limit_info.get('remaining', 0)
            if remaining > 0:
                print(f"\n[INFO] Submissions remaining today: {remaining}/{limit_info.get('limit', 10)}")
        
        # Compute file hash for tracking
        file_hash = compute_file_hash(submission_path)
        
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
        
        # Try to get score from submissions list (more reliable) with retries
        if not result:
            print("\nTrying alternative method: checking submissions list...")
            submission_result = None
            # Retry getting score (submissions can take time to process)
            # Use longer wait times for score retrieval
            score_wait_time = max(wait_time * 2, 10)  # At least 10 seconds between retries
            for retry in range(1, max_retries + 1):
                submission_result = get_latest_submission_score(api)
                if submission_result and submission_result.get("score"):
                    print(f"[SUCCESS] Score retrieved: {submission_result['score']:.6f}")
                    break
                elif retry < max_retries:
                    print(f"Score not yet available. Waiting {score_wait_time} more seconds... (retry {retry}/{max_retries})")
                    time.sleep(score_wait_time)
            
            if submission_result and submission_result.get("score"):
                # Create result dict from submission data
                result = {
                    "rank": "unknown",  # Can't get rank from submissions list
                    "score": submission_result["score"],
                    "username": get_username(),
                    "submission_date": submission_result["submission_date"],
                    "total_submissions": len(api.competition_submissions(COMPETITION_NAME)) if api else "unknown"
                }
        
        if result:
            # Load and update submission log
            log = load_submission_log()
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": submission_path.name,
                "message": message,
                "rank": result.get("rank", "unknown"),
                "score": result["score"],
                "total_submissions": result.get("total_submissions", "unknown"),
                "file_hash": file_hash  # Store hash to detect file changes
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
            
            # Automatically log Kaggle score to model performance log if we can infer model name
            if result and result.get("score"):
                try:
                    # Try to infer model name from submission file path
                    # Submission files are typically in: data/submissions/<model_name>/<model_name>_Model.csv
                    # or: data/submissions/<model_name>_Model.csv
                    model_name = None
                    
                    # First, try to get model name from parent directory
                    # e.g., "data/submissions/7_xgboost/xgboost_Model.csv" -> "7_xgboost" -> "xgboost"
                    parent_dir = submission_path.parent.name.lower()
                    if parent_dir and parent_dir != "submissions":
                        # Extract model name from directory (handle patterns like "7_xgboost", "xgboost", etc.)
                        parts = parent_dir.split("_")
                        if len(parts) > 1 and parts[0].isdigit():
                            # Pattern: "7_xgboost" -> "xgboost"
                            model_name_candidate = "_".join(parts[1:])
                        else:
                            # Pattern: "xgboost" -> "xgboost"
                            model_name_candidate = parent_dir
                        
                        # Map to standard model names
                        if model_name_candidate in ["xgboost", "xgb"]:
                            model_name = "xgboost"
                        elif model_name_candidate in ["lightgbm", "lgb", "lightgb"]:
                            model_name = "lightgbm"
                        elif model_name_candidate in ["catboost", "cat"]:
                            model_name = "catboost"
                        elif model_name_candidate == "ridge":
                            model_name = "ridge"
                        elif model_name_candidate == "lasso":
                            model_name = "lasso"
                        elif "elastic" in model_name_candidate:
                            model_name = "elastic_net"
                        elif "random" in model_name_candidate or model_name_candidate == "rf":
                            model_name = "random_forest"
                        elif model_name_candidate == "svr":
                            model_name = "svr"
                        elif "stacking" in model_name_candidate:
                            model_name = "STACKING_META"
                        elif "blend" in model_name_candidate:
                            model_name = "blending"
                    
                    # Fallback: try to infer from filename if directory didn't work
                    if not model_name:
                        file_stem = submission_path.stem.lower()
                        
                        # Common model name patterns in filename
                        if "xgboost" in file_stem or "xgb" in file_stem:
                            model_name = "xgboost"
                        elif "lightgbm" in file_stem or "lgb" in file_stem or "lightgb" in file_stem:
                            model_name = "lightgbm"
                        elif "catboost" in file_stem or "cat" in file_stem:
                            model_name = "catboost"
                        elif "ridge" in file_stem:
                            model_name = "ridge"
                        elif "lasso" in file_stem:
                            model_name = "lasso"
                        elif "elastic" in file_stem:
                            model_name = "elastic_net"
                        elif "random" in file_stem or "rf" in file_stem:
                            model_name = "random_forest"
                        elif "svr" in file_stem:
                            model_name = "svr"
                        elif "stacking" in file_stem:
                            model_name = "STACKING_META"
                        elif "blend" in file_stem:
                            model_name = "blending"
                    
                    if model_name:
                        from utils.metrics import log_kaggle_score
                        log_kaggle_score(model_name, result["score"])
                        print(f"\n[SUCCESS] Automatically logged Kaggle score ({result['score']:.6f}) to model performance log for '{model_name}'")
                    else:
                        print(f"\n[INFO] Could not infer model name from path: {submission_path}")
                        print("   Score was retrieved but not logged. You can manually log it using:")
                        print(f"   python scripts/get_kaggle_score.py <model_name>")
                except Exception as e:
                    print(f"\n[WARNING] Could not auto-log score: {e}")
                    print("   You can manually log it using: python scripts/get_kaggle_score.py <model_name>")
            
            return result
        else:
            print("\nCould not retrieve leaderboard results.")
            print("   Check manually on Kaggle website.")
            print("   Or run: python scripts/get_kaggle_score.py <model_name>")
            return None
            
    except Exception as e:
        print(f"\nError during submission: {e}")
        print("\nTroubleshooting:")
        print("1. Check your kaggle.json credentials")
        print("2. Verify internet connection")
        print("3. Ensure submission file format is correct")
        return None

