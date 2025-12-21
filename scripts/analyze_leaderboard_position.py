import pandas as pd
import numpy as np

df = pd.read_csv('runs/leaderboard_distribution.csv')
current = 0.12609
total = len(df)

# Find our rank (lower score = better rank)
# Rank 1 = best (lowest score), so we find all scores <= current
our_rank_row = df[df['score'] <= current].iloc[-1]  # Last row with score <= current
our_rank = int(our_rank_row['rank'])
percentile = 100 * (1 - our_rank / total)

print(f"Current Score: {current}")
print(f"Total Participants: {total}")
print(f"Rank: {our_rank} / {total}")
print(f"Percentile: {percentile:.2f}% (Top {100-percentile:.2f}%)")
print(f"\nPercentile Thresholds:")
print(f"Top 1% (rank {int(total*0.01)}): {df.iloc[int(total*0.01)]['score']:.5f}")
print(f"Top 5% (rank {int(total*0.05)}): {df.iloc[int(total*0.05)]['score']:.5f}")
print(f"Top 10% (rank {int(total*0.10)}): {df.iloc[int(total*0.10)]['score']:.5f}")
print(f"Top 25% (rank {int(total*0.25)}): {df.iloc[int(total*0.25)]['score']:.5f}")
print(f"Median (50%, rank {int(total*0.50)}): {df.iloc[int(total*0.50)]['score']:.5f}")
print(f"\nGap Analysis:")
top5_score = df.iloc[int(total*0.05)]['score']
top10_score = df.iloc[int(total*0.10)]['score']
top25_score = df.iloc[int(total*0.25)]['score']
print(f"Gap to Top 5%: {current - top5_score:.5f} RMSLE")
print(f"Gap to Top 10%: {current - top10_score:.5f} RMSLE")
print(f"Gap to Top 25%: {current - top25_score:.5f} RMSLE")
print(f"\nScore Distribution:")
print(f"Best Score: {df.iloc[0]['score']:.5f} (Rank 1)")
print(f"Worst Score: {df.iloc[-1]['score']:.5f} (Rank {total})")
print(f"Mean Score: {df['score'].mean():.5f}")
print(f"Median Score: {df['score'].median():.5f}")

