# Kaggle Submission & GPU Workflow Guide

Complete guide for submitting models to Kaggle and using Kaggle's GPU infrastructure.

---

## Quick Start: Run CatBoost on Kaggle GPU

### Prerequisites

1. ✅ Your code is committed and pushed to GitHub
2. ✅ You have a Kaggle account with GPU quota available (30 hours/week free tier)
3. ✅ Competition dataset linked to your notebook

### Step-by-Step Instructions

#### Step 1: Create a New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Set the following settings:
   - **Accelerator**: Select **"GPU"** (P100 or T4)
   - **Internet**: Enable (to clone repository)

#### Step 2: Add Competition Dataset

1. Click **"Add Data"** button (in the right panel)
2. Search for: `house-prices-advanced-regression-techniques`
3. Click **"Add"** to link the dataset to your notebook

#### Step 3: Use the Pre-made Notebook

1. Upload the notebook file: `kaggle/notebooks/run_catboost_kaggle.ipynb`
   - In Kaggle, click "File" → "Upload Notebook"
   - Select `kaggle/notebooks/run_catboost_kaggle.ipynb`

OR copy cells manually from `kaggle/notebooks/run_catboost_kaggle.ipynb`

#### Step 4: Run All Cells

Click **"Run All"** or run each cell sequentially. The notebook will:
- Clone your repository
- Install dependencies
- Setup Kaggle environment
- Verify GPU availability
- Train CatBoost model with GPU acceleration
- Generate submission file

#### Step 5: Download Results

After training completes:
1. Go to the **"Data"** panel (right side of notebook)
2. Navigate to: `/kaggle/working/project/data/submissions/`
3. Download the latest submission CSV file
4. Submit it on the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submissions)

---

## Hybrid Local-Kaggle Workflow

The hybrid workflow allows you to:
- **Develop locally**: Edit code, run preprocessing, test models on CPU (fast iteration)
- **Train on Kaggle GPU**: Run GPU-intensive model training on Kaggle's servers
- **Sync via Git**: Use Git to synchronize code between local and Kaggle environments

### Architecture

```
Local Machine (Development)     Git Repository          Kaggle Notebook (GPU)
     |                                |                         |
     |-- Edit Code ------------------>|                         |
     |-- Commit & Push -------------->|                         |
     |                                |<-- Clone Repo ----------|
     |                                |                         |
     |                                |-- Run Model (GPU) ------>|
     |                                |                         |
     |<-- Download Results -----------|-- Save Outputs ---------|
```

### Local Development Workflow

1. **Edit Code Locally**
   - Edit preprocessing scripts
   - Modify model configurations
   - Test on CPU (fast iteration)

2. **Check Git Status**
   ```bash
   python kaggle/sync.py
   ```
   This checks for uncommitted changes and unpushed commits.

3. **Commit and Push**
   ```bash
   git add .
   git commit -m "Your commit message"
   git push
   ```

### Kaggle GPU Execution

1. Create/Open Kaggle Notebook
2. Enable GPU in settings
3. Clone repository from GitHub
4. Run model training notebook
5. Download results

---

## Submitting Models to Kaggle

### Using Python Script

```bash
# Submit a specific model
python kaggle/submit.py <model_name>

# Examples:
python kaggle/submit.py catboost
python kaggle/submit.py xgboost
python kaggle/submit.py lightgbm
```

Available models: `catboost`, `xgboost`, `lightgbm`, `ridge`, `lasso`, `elastic_net`, `random_forest`, `svr`, `blending`, `stacking`

### Checking Submission Status

```bash
# Check submission status and scores
python kaggle/scores.py status

# Get latest score
python kaggle/scores.py latest

# Sync all scores
python kaggle/scores.py sync
```

---

## What to Expect

### GPU Detection
- If GPU is enabled, you'll see: `[OK] GPU available: Tesla P100` (or T4)
- CatBoost will automatically use GPU acceleration

### Training Process
- Optuna will run hyperparameter optimization (configurable trials)
- Final model will be trained with best parameters
- Training time: ~5-15 minutes depending on trials

### Output Files
- Submission file: `data/submissions/catboost_YYYYMMDD_HHMMSS.csv`
- Performance log: `runs/model_performance.csv`
- Model artifacts in `runs/` directory

---

## Troubleshooting

### GPU Not Detected
- Make sure GPU accelerator is enabled in notebook settings
- Restart the notebook session after enabling GPU

### Import Errors
- Check that all dependencies are installed (Cell 1)
- Verify the repository was cloned correctly

### Data Not Found
- Ensure competition dataset is added
- Check that symlinks were created (Cell 2 output)

### Out of Memory
- Reduce `n_trials` in model config
- Use smaller batch sizes if applicable

### Submission Failed
- Check Kaggle API credentials are set up correctly
- Verify submission file format is correct
- Check daily submission limit (10 submissions/day)

---

## Daily Submission Limit

Kaggle allows **10 submissions per day** per competition. The submission scripts automatically:
- Check if a submission with the same content has already been submitted
- Skip duplicate submissions to save your daily quota
- Log all submissions for tracking

---

## Next Steps

After successful run:
1. Download the submission file
2. Submit to competition leaderboard
3. Compare results with local runs
4. Iterate on hyperparameters if needed

