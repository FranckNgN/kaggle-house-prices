#!/usr/bin/env python
"""
Get current leaderboard ranking and calculate quantile percent.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle.api.kaggle_api_extended import KaggleApi
from utils.kaggle_helper import COMPETITION_NAME, get_username


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
                # Get all unique team names from submissions
                for sub in subs:
                    team_name = getattr(sub, 'teamName', None)
                    if team_name:
                        team_names.add(team_name.lower())
                        # Also try variations (remove spaces, etc.)
                        team_names.add(team_name.lower().replace(' ', ''))
                        team_names.add(team_name.lower().replace('_', ''))
                
                # Get best score from latest submission
                latest = subs[0]
                score = getattr(latest, 'publicScore', None) or getattr(latest, 'score', None)
                if score is not None:
                    best_score = float(score)
        except Exception as e:
            print(f"[INFO] Could not get team names from submissions: {e}")
        
        print(f"Trying to match team names: {team_names}")
        if best_score:
            print(f"Best submission score: {best_score:.6f}")
        
        # Find user's entry - try multiple methods
        user_entry = None
        
        # Method 1: Match by team name
        for i, entry in enumerate(leaderboard):
            team_name = getattr(entry, 'teamName', '')
            entry_lower = team_name.lower()
            # Try exact match and variations
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
        
        # Method 2: If not found by name, try to match by best score (if close)
        if not user_entry and best_score:
            print("\n[INFO] Not found by team name, trying to match by score...")
            for i, entry in enumerate(leaderboard):
                entry_score = float(entry.score) if hasattr(entry, 'score') and entry.score else None
                if entry_score and abs(entry_score - best_score) < 0.0001:  # Very close match
                    user_entry = {
                        "rank": i + 1,
                        "score": entry_score,
                        "username": getattr(entry, 'teamName', 'unknown'),
                        "submission_date": str(getattr(entry, 'submissionDate', 'unknown'))
                    }
                    print(f"[INFO] Found potential match by score: rank {i+1}, score {entry_score:.6f}")
                    break
        
        if not user_entry:
            print(f"\n[WARNING] No submission found for team names: {team_names}")
            print(f"\nTotal participants on leaderboard: {total_participants}")
            if leaderboard:
                print("\nFull Leaderboard:")
                print("-" * 70)
                print(f"{'Rank':<6} {'Team Name':<30} {'Score':<15}")
                print("-" * 70)
                for i, entry in enumerate(leaderboard):
                    team_name = getattr(entry, 'teamName', 'unknown')
                    score = getattr(entry, 'score', 'unknown')
                    print(f"{i+1:<6} {team_name:<30} {score}")
                print("-" * 70)
            print(f"\n[NOTE] Your submission is not in the top {total_participants} on the leaderboard.")
            if best_score:
                print(f"       Your best score: {best_score:.6f}")
                # Check if score would be in top participants
                if leaderboard:
                    last_score = float(leaderboard[-1].score) if hasattr(leaderboard[-1], 'score') and leaderboard[-1].score else float('inf')
                    if best_score > last_score:
                        print(f"       Last place on leaderboard: {last_score:.6f}")
                        print(f"       Your score is worse than the last place, so you're ranked below position {total_participants}.")
                        
                        # Provide estimate based on common competition sizes
                        print("\n" + "=" * 70)
                        print("QUANTILE ESTIMATE (Based on Common Competition Sizes)")
                        print("=" * 70)
                        print("\nNote: Kaggle API only shows top participants.")
                        print("      Without total participant count, we can only estimate.")
                        print("\nAssuming different total participant counts:")
                        print("-" * 70)
                        
                        # Common competition sizes to estimate
                        estimated_totals = [100, 500, 1000, 2000, 5000, 10000]
                        for total in estimated_totals:
                            # Estimate rank: if last place is 0.00044 and user is 0.136930
                            # Score ratio suggests user is much further down
                            # Rough estimate: rank might be around total * (score_ratio)
                            score_ratio = best_score / last_score if last_score > 0 else 1
                            # Very rough estimate - rank could be anywhere from 21 to total
                            # Let's assume rank is somewhere in the middle-lower range
                            estimated_rank_min = total_participants + 1
                            estimated_rank_max = min(int(total * 0.8), total)  # Assume not in bottom 20%
                            
                            # Calculate quantile for estimated range
                            quantile_min = (1 - (estimated_rank_min - 1) / total) * 100
                            quantile_max = (1 - (estimated_rank_max - 1) / total) * 100
                            
                            print(f"If total participants = {total:,}:")
                            print(f"  Estimated rank range: {estimated_rank_min:,} - {estimated_rank_max:,}")
                            print(f"  Estimated quantile: {quantile_max:.2f}% - {quantile_min:.2f}%")
                            print(f"  (You're better than approximately {quantile_max:.1f}% - {quantile_min:.1f}% of participants)")
                        
                        print("\n" + "-" * 70)
                        print("To get your exact quantile percent, you need to:")
                        print("1. Check the Kaggle competition page manually")
                        print("2. Find your rank and total participants")
                        print("3. Use formula: Quantile % = (1 - (rank - 1) / total) * 100")
                        print("=" * 70)
            return None
        
        # Calculate quantile percent
        # Quantile percent = (1 - (rank - 1) / total_participants) * 100
        # This gives the percentage of participants you're better than
        rank = user_entry["rank"]
        quantile_percent = (1 - (rank - 1) / total_participants) * 100
        
        # Also calculate percentile (same as quantile percent)
        percentile = quantile_percent
        
        # Display results
        print(f"\n[SUCCESS] Found your submission!")
        print("-" * 70)
        print(f"Rank: {rank:,} out of {total_participants:,} participants")
        print(f"Score: {user_entry['score']:.6f}")
        print(f"Username: {user_entry['username']}")
        print(f"Submission Date: {user_entry['submission_date']}")
        print("-" * 70)
        print(f"\nQuantile Percent: {quantile_percent:.4f}%")
        print(f"Percentile: {percentile:.4f}%")
        print(f"\nInterpretation:")
        print(f"  You are better than {quantile_percent:.2f}% of participants")
        print(f"  You are in the top {100 - quantile_percent:.2f}%")
        print("=" * 70)
        
        return {
            "rank": rank,
            "total_participants": total_participants,
            "score": user_entry["score"],
            "quantile_percent": quantile_percent,
            "percentile": percentile
        }
        
    except Exception as e:
        print(f"\n[ERROR] Error fetching leaderboard: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    get_leaderboard_quantile()

