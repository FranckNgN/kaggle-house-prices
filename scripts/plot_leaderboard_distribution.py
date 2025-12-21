#!/usr/bin/env python
"""
Plot the distribution of scores by rank from the leaderboard.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle import get_username, load_submission_log


def plot_score_distribution(leaderboard_file: Path, output_file: Path):
    """Plot score distribution by rank."""
    # Load data
    print(f"Loading leaderboard data from: {leaderboard_file}")
    df = pd.read_csv(leaderboard_file)
    
    print(f"Loaded {len(df)} entries")
    print(f"Score range: {df['score'].min():.6f} to {df['score'].max():.6f}")
    
    # Get user's best score and rank
    username = get_username()
    log = load_submission_log()
    best_score = None
    user_rank = None
    
    if log:
        valid_scores = [e for e in log if isinstance(e.get('score'), (int, float))]
        if valid_scores:
            best_entry = min(valid_scores, key=lambda x: x['score'])
            best_score = best_entry['score']
            
            # Find rank for this score
            user_entry = df[df['score'] == best_score]
            if len(user_entry) > 0:
                user_rank = int(user_entry.iloc[0]['rank'])
            else:
                # Find closest rank
                closest_idx = (df['score'] - best_score).abs().idxmin()
                user_rank = int(df.loc[closest_idx, 'rank'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Main plot: Score vs Rank (full range)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['rank'], df['score'], 'b-', linewidth=0.5, alpha=0.7, label='All participants')
    
    # Highlight user's position
    if user_rank and best_score:
        ax1.plot(user_rank, best_score, 'ro', markersize=12, label=f'Your position (Rank {user_rank:,})', zorder=5)
        ax1.axvline(user_rank, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(best_score, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add percentile lines
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        rank_p = int(len(df) * p / 100)
        score_p = df.iloc[rank_p - 1]['score']
        ax1.axvline(rank_p, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        if p in [25, 50, 75]:
            ax1.text(rank_p, df['score'].max() * 0.9, f'{p}%', 
                    rotation=90, fontsize=8, alpha=0.7, ha='right')
    
    ax1.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (RMSLE)', fontsize=12, fontweight='bold')
    ax1.set_title('Score Distribution by Rank (Full Range)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, len(df))
    
    # 2. Zoomed view: Top 1000 ranks
    ax2 = fig.add_subplot(gs[1, 0])
    top_n = min(1000, len(df))
    df_top = df.head(top_n)
    ax2.plot(df_top['rank'], df_top['score'], 'b-', linewidth=1, alpha=0.8)
    
    if user_rank and user_rank <= top_n:
        ax2.plot(user_rank, best_score, 'ro', markersize=10, label=f'You (Rank {user_rank:,})', zorder=5)
        ax2.axvline(user_rank, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Rank', fontsize=11)
    ax2.set_ylabel('Score (RMSLE)', fontsize=11)
    ax2.set_title(f'Top {top_n} Ranks (Zoomed)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if user_rank and user_rank <= top_n:
        ax2.legend(loc='upper left', fontsize=9)
    
    # 3. Score histogram
    ax3 = fig.add_subplot(gs[1, 1])
    # Focus on reasonable score range (exclude extreme outliers)
    score_max = df['score'].quantile(0.95)  # 95th percentile
    df_filtered = df[df['score'] <= score_max]
    
    ax3.hist(df_filtered['score'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    
    if best_score and best_score <= score_max:
        ax3.axvline(best_score, color='r', linestyle='--', linewidth=2, label=f'Your score: {best_score:.6f}')
        ax3.legend(loc='upper right', fontsize=9)
    
    ax3.set_xlabel('Score (RMSLE)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Score Distribution (up to {score_max:.2f})', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Log scale view (to see distribution better)
    ax4 = fig.add_subplot(gs[2, 0])
    # Use log scale for scores to better visualize the distribution
    scores_positive = df[df['score'] > 0]['score']
    if len(scores_positive) > 0:
        ax4.plot(df[df['score'] > 0]['rank'], np.log10(scores_positive + 1e-6), 
                'b-', linewidth=0.5, alpha=0.7)
        
        if best_score and best_score > 0:
            ax4.plot(user_rank, np.log10(best_score + 1e-6), 'ro', markersize=10, 
                    label=f'You (Rank {user_rank:,})', zorder=5)
            ax4.axvline(user_rank, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    ax4.set_xlabel('Rank', fontsize=11)
    ax4.set_ylabel('Log10(Score + 1e-6)', fontsize=11)
    ax4.set_title('Score Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    if best_score and best_score > 0:
        ax4.legend(loc='upper left', fontsize=9)
    
    # 5. Percentile breakdown
    ax5 = fig.add_subplot(gs[2, 1])
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_scores = [df['score'].quantile(p/100) for p in percentiles]
    percentile_ranks = [int(len(df) * p / 100) for p in percentiles]
    
    ax5.barh(range(len(percentiles)), percentile_scores, color='steelblue', alpha=0.7)
    ax5.set_yticks(range(len(percentiles)))
    ax5.set_yticklabels([f'{p}th' for p in percentiles])
    ax5.set_xlabel('Score (RMSLE)', fontsize=11)
    ax5.set_ylabel('Percentile', fontsize=11)
    ax5.set_title('Score by Percentile', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add score values on bars
    for i, (p, score) in enumerate(zip(percentiles, percentile_scores)):
        ax5.text(score, i, f' {score:.4f}', va='center', fontsize=8)
    
    # Add user's percentile if available
    if user_rank:
        user_percentile = (1 - (user_rank - 1) / len(df)) * 100
        if user_percentile >= 1:
            # Find which percentile bracket user is in
            for i, p in enumerate(percentiles):
                if user_percentile <= p:
                    ax5.barh(i, percentile_scores[i], color='red', alpha=0.5, 
                            label=f'You: {user_percentile:.1f}th percentile')
                    break
    
    # Add summary text
    summary_text = f"""
    Total Participants: {len(df):,}
    Score Range: {df['score'].min():.6f} - {df['score'].max():.6f}
    Median Score: {df['score'].median():.6f}
    Mean Score: {df['score'].mean():.6f}
    """
    
    if user_rank and best_score:
        user_percentile = (1 - (user_rank - 1) / len(df)) * 100
        summary_text += f"""
    
    Your Position:
    Rank: {user_rank:,} / {len(df):,}
    Score: {best_score:.6f}
    Percentile: {user_percentile:.2f}%
    (Better than {user_percentile:.2f}% of participants)
    """
    
    fig.suptitle('Kaggle House Prices Competition - Leaderboard Score Distribution', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Plot saved to: {output_file}")
    
    # Also save a summary
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("LEADERBOARD DISTRIBUTION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(summary_text)
        f.write("\n" + "=" * 70 + "\n")
    print(f"[SUCCESS] Summary saved to: {summary_file}")
    
    plt.show()


def main():
    """Main function."""
    print("=" * 70)
    print("LEADERBOARD SCORE DISTRIBUTION PLOTTER")
    print("=" * 70)
    
    # Find leaderboard file
    leaderboard_file = PROJECT_ROOT / "runs" / "leaderboard_distribution.csv"
    
    if not leaderboard_file.exists():
        # Try to find it in data/raw
        raw_files = list((PROJECT_ROOT / "data" / "raw").glob("*leaderboard*.csv"))
        if raw_files:
            leaderboard_file = raw_files[0]
            print(f"Found leaderboard file: {leaderboard_file}")
        else:
            print("[ERROR] Leaderboard file not found.")
            print("Please run: python scripts/get_full_leaderboard.py")
            return
    else:
        print(f"Using leaderboard file: {leaderboard_file}")
    
    # Output file
    output_file = PROJECT_ROOT / "runs" / "leaderboard_distribution_plot.png"
    
    # Create plot
    plot_score_distribution(leaderboard_file, output_file)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()


