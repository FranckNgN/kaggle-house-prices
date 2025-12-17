# Kaggle Submission Scripts

This directory contains scripts for submitting models to Kaggle and checking leaderboard rankings.

## Setup

1. **Install Kaggle API:**
   ```bash
   pip install kaggle
   ```

2. **Configure Kaggle credentials:**
   - Download your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/account)
   - Place it in `.kaggle/kaggle.json` in the project root

## Scripts

### 1. `submit_to_kaggle.py` - Single Model Submission

Submit a single model file to Kaggle.

**Usage:**
```bash
python scripts/submit_to_kaggle.py <submission_file.csv> [message]
```

**Examples:**
```bash
# Submit ElasticNet model
python scripts/submit_to_kaggle.py data/submissions/elasticNetModel.csv "ElasticNet - alpha=0.0007"

# Submit Ridge model
python scripts/submit_to_kaggle.py data/submissions/ridgeModel.csv "Ridge - alpha=8"
```

### 2. `submit_all_models.py` - Interactive Submission Manager

Interactive script to submit models one by one or all at once.

**Usage:**
```bash
python scripts/submit_all_models.py
```

**Features:**
- View all available submission files
- Submit a single model (with selection menu)
- Submit all models at once
- View submission history
- Automatic leaderboard checking after each submission
- Summary of all submissions with rankings

**Menu Options:**
1. **Submit a single model** - Choose from available files
2. **Submit all models** - Batch submit all CSV files
3. **View available submissions** - List all submission files
4. **View submission history** - See past submissions and results
5. **Exit** - Quit the script

## Features

- ✅ Automatic submission to Kaggle
- ✅ Leaderboard ranking retrieval
- ✅ Score comparison with previous best
- ✅ Submission history logging
- ✅ Batch submission support
- ✅ Interactive model selection

## Submission Log

All submissions are logged to `data/submissions/submission_log.json` with:
- Timestamp
- File name
- Submission message
- Rank
- Score (RMSLE)
- Total submissions count

## Notes

- The script waits 5-8 seconds after submission before checking leaderboard
- Retries up to 3-5 times if leaderboard check fails
- 10-second delay between batch submissions to avoid rate limiting
- Make sure your submission files are in `data/submissions/` directory

