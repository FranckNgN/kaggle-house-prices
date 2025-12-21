#!/usr/bin/env python
"""
Download full leaderboard and analyze score distribution by rank.
Also calculates exact quantile percent for the user.
"""

import sys
import subprocess
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle import COMPETITION_NAME, get_username


def download_leaderboard_csv(competition_name: str, output_dir: Path) -> Optional[Path]:
    """
    Download full leaderboard CSV using Kaggle CLI or Python API.
    
    Returns:
        Path to downloaded CSV file, or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Method 1: Try Python API download
    try:
        print("Attempting to download full leaderboard using Python API...")
        api = KaggleApi()
        api.authenticate()
        
        # Try to download leaderboard file
        # The API might have a method to download leaderboard
        # Check if competition_leaderboard_download exists
        if hasattr(api, 'competition_leaderboard_download'):
            api.competition_leaderboard_download(
                competition_name,
                path=str(output_dir),
                quiet=False
            )
        else:
            # Alternative: try using the CLI via subprocess
            raise AttributeError("API method not available")
        
        # Find the downloaded file
        csv_files = list(output_dir.glob("*leaderboard*.csv"))
        if csv_files:
            leaderboard_file = csv_files[0]
            print(f"[SUCCESS] Downloaded leaderboard to: {leaderboard_file}")
            return leaderboard_file
            
    except (AttributeError, Exception) as e:
        print(f"[INFO] Python API download not available: {e}")
        print("Trying Kaggle CLI...")
    
    # Method 2: Try Kaggle CLI command
    try:
        print("Attempting to download full leaderboard using Kaggle CLI...")
        # Change to output directory for download
        original_cwd = Path.cwd()
        
        try:
            import os
            os.chdir(str(output_dir))
            
            cmd = [
                "kaggle", "competitions", "leaderboard",
                "-c", competition_name,
                "-d"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Find the downloaded file in current directory
                csv_files = list(Path.cwd().glob("*leaderboard*.csv"))
                if csv_files:
                    leaderboard_file = csv_files[0]
                    print(f"[SUCCESS] Downloaded leaderboard to: {leaderboard_file}")
                    return leaderboard_file
                else:
                    print("[WARNING] Command succeeded but no CSV file found")
                    print(f"Output: {result.stdout}")
            else:
                print(f"[WARNING] Kaggle CLI command failed: {result.stderr}")
        finally:
            os.chdir(str(original_cwd))
            
    except FileNotFoundError:
        print("[INFO] Kaggle CLI not found.")
        return None
    except subprocess.TimeoutExpired:
        print("[WARNING] Kaggle CLI command timed out")
        return None
    except Exception as e:
        print(f"[WARNING] Error using Kaggle CLI: {e}")
        return None
    
    return None


def get_leaderboard_via_api(api: KaggleApi, competition_name: str) -> Optional[pd.DataFrame]:
    """
    Try to get leaderboard via API (may be limited to top entries).
    
    Returns:
        DataFrame with leaderboard data, or None if failed
    """
    try:
        print("Attempting to fetch leaderboard via API...")
        leaderboard = api.competition_leaderboard_view(competition_name)
        
        if not leaderboard:
            return None
        
        # Convert to DataFrame
        data = []
        for i, entry in enumerate(leaderboard):
            data.append({
                'Rank': i + 1,
                'TeamName': getattr(entry, 'teamName', ''),
                'Score': float(entry.score) if hasattr(entry, 'score') and entry.score else None,
                'SubmissionDate': str(getattr(entry, 'submissionDate', ''))
            })
        
        df = pd.DataFrame(data)
        print(f"[INFO] Fetched {len(df)} entries via API (may be limited)")
        return df
        
    except Exception as e:
        print(f"[WARNING] Error fetching via API: {e}")
        return None


def analyze_leaderboard(df: pd.DataFrame, username: str, best_score: Optional[float] = None) -> Dict:
    """
    Analyze leaderboard distribution and find user's rank.
    
    Returns:
        Dict with analysis results
    """
    total_participants = len(df)
    
    print("\n" + "=" * 70)
    print("LEADERBOARD ANALYSIS")
    print("=" * 70)
    print(f"Total participants: {total_participants:,}")
    
    # Find user's entry
    user_entry = None
    team_names = {username.lower(), username.lower().replace(' ', ''), username.lower().replace('_', '')}
    
    # Try to match by team name
    for idx, row in df.iterrows():
        team_name = str(row.get('TeamName', '')).lower()
        if (team_name in team_names or 
            team_name.replace(' ', '') in team_names or
            team_name.replace('_', '') in team_names):
            user_entry = {
                'rank': int(row['Rank']),
                'score': float(row['Score']) if pd.notna(row['Score']) else None,
                'team_name': row['TeamName']
            }
            break
    
    # If not found by name, try to match by score (if provided)
    if not user_entry and best_score:
        print(f"\n[INFO] Not found by team name, trying to match by score ({best_score:.6f})...")
        for idx, row in df.iterrows():
            score = row.get('Score')
            if pd.notna(score):
                score = float(score)
                if abs(score - best_score) < 0.0001:  # Very close match
                    user_entry = {
                        'rank': int(row['Rank']),
                        'score': score,
                        'team_name': row['TeamName']
                    }
                    print(f"[INFO] Found potential match by score: rank {user_entry['rank']}")
                    break
    
    # Score distribution analysis
    if 'Score' in df.columns:
        scores = df['Score'].dropna()
        if len(scores) > 0:
            print(f"\nScore Statistics:")
            print(f"  Min (best): {scores.min():.6f}")
            print(f"  Max (worst): {scores.max():.6f}")
            print(f"  Mean: {scores.mean():.6f}")
            print(f"  Median: {scores.median():.6f}")
            print(f"  Std Dev: {scores.std():.6f}")
            
            # Percentile distribution
            print(f"\nScore Distribution by Percentiles:")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                val = scores.quantile(p / 100)
                print(f"  {p:3d}th percentile: {val:.6f}")
    
    # Calculate quantile percent for user
    result = {
        'total_participants': total_participants,
        'user_entry': user_entry,
        'score_stats': {}
    }
    
    if user_entry:
        rank = user_entry['rank']
        quantile_percent = (1 - (rank - 1) / total_participants) * 100
        percentile = quantile_percent
        
        print("\n" + "=" * 70)
        print("YOUR LEADERBOARD POSITION")
        print("=" * 70)
        print(f"Rank: {rank:,} out of {total_participants:,} participants")
        print(f"Score: {user_entry['score']:.6f}")
        print(f"Team Name: {user_entry['team_name']}")
        print("-" * 70)
        print(f"Quantile Percent: {quantile_percent:.4f}%")
        print(f"Percentile: {percentile:.4f}%")
        print(f"\nInterpretation:")
        print(f"  You are better than {quantile_percent:.2f}% of participants")
        print(f"  You are in the top {100 - quantile_percent:.2f}%")
        print("=" * 70)
        
        result['quantile_percent'] = quantile_percent
        result['percentile'] = percentile
    else:
        print("\n" + "=" * 70)
        print("USER NOT FOUND ON LEADERBOARD")
        print("=" * 70)
        print(f"Could not find your submission in the leaderboard.")
        if best_score:
            print(f"Your best score: {best_score:.6f}")
            # Find where this score would rank
            if 'Score' in df.columns:
                scores = df['Score'].dropna()
                if len(scores) > 0:
                    worse_scores = (scores > best_score).sum()
                    estimated_rank = worse_scores + 1
                    if estimated_rank > total_participants:
                        estimated_rank = total_participants
                    
                    quantile_percent = (1 - (estimated_rank - 1) / total_participants) * 100
                    print(f"\nEstimated rank (based on score): {estimated_rank:,}")
                    print(f"Estimated quantile percent: {quantile_percent:.4f}%")
                    result['estimated_rank'] = estimated_rank
                    result['quantile_percent'] = quantile_percent
    
    return result


def main():
    """Main function to download and analyze leaderboard."""
    print("=" * 70)
    print("FULL LEADERBOARD DOWNLOAD & ANALYSIS")
    print("=" * 70)
    print(f"Competition: {COMPETITION_NAME}")
    
    username = get_username()
    print(f"Username: {username}")
    
    # Get best score from submission log
    from kaggle import load_submission_log
    log = load_submission_log()
    best_score = None
    if log:
        valid_scores = [e for e in log if isinstance(e.get('score'), (int, float))]
        if valid_scores:
            best_entry = min(valid_scores, key=lambda x: x['score'])
            best_score = best_entry['score']
            print(f"Best submission score: {best_score:.6f}")
    
    # Try to download leaderboard CSV
    output_dir = PROJECT_ROOT / "data" / "raw"
    
    # First, check if leaderboard file already exists
    existing_files = list(output_dir.glob("*leaderboard*.csv"))
    leaderboard_file = None
    
    if existing_files:
        leaderboard_file = existing_files[0]
        print(f"\nFound existing leaderboard file: {leaderboard_file}")
    else:
        leaderboard_file = download_leaderboard_csv(COMPETITION_NAME, output_dir)
    
    df = None
    
    if leaderboard_file and leaderboard_file.exists():
        try:
            print(f"\nLoading leaderboard from: {leaderboard_file}")
            df = pd.read_csv(leaderboard_file)
            print(f"[SUCCESS] Loaded {len(df)} entries from CSV")
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            df = None
    
    # If CSV download failed, try API (limited)
    if df is None:
        try:
            api = KaggleApi()
            api.authenticate()
            df = get_leaderboard_via_api(api, COMPETITION_NAME)
        except Exception as e:
            print(f"[ERROR] Failed to fetch via API: {e}")
            df = None
    
    if df is None or len(df) == 0:
        print("\n[ERROR] Could not fetch leaderboard data.")
        print("Please try:")
        print("  1. Install Kaggle CLI: pip install kaggle")
        print("  2. Authenticate: kaggle api configure")
        print("  3. Run: kaggle competitions leaderboard -c house-prices-advanced-regression-techniques -d")
        return None
    
    # Analyze the leaderboard
    result = analyze_leaderboard(df, username, best_score)
    
    # Save distribution to file
    if 'Score' in df.columns:
        output_file = PROJECT_ROOT / "runs" / "leaderboard_distribution.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create distribution summary
        scores = df['Score'].dropna()
        if len(scores) > 0:
            distribution = pd.DataFrame({
                'rank': range(1, len(scores) + 1),
                'score': scores.sort_values().values
            })
            distribution.to_csv(output_file, index=False)
            print(f"\n[INFO] Score distribution saved to: {output_file}")
    
    return result


if __name__ == "__main__":
    main()

