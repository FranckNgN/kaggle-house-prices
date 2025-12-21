# Quick Start: Run CatBoost on Kaggle GPU

This guide will help you run the CatBoost model on Kaggle's GPU servers in just a few steps.

## Prerequisites

1. ✅ Your code is committed and pushed to GitHub (repository: `FranckNgN/kaggle-house-prices`)
2. ✅ You have a Kaggle account
3. ✅ You have GPU quota available (free tier: 30 hours/week)

## Step-by-Step Instructions

### Step 1: Create a New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Set the following settings:
   - **Accelerator**: Select **"GPU"** (P100 or T4)
   - **Internet**: Enable (to clone repository)

### Step 2: Add Competition Dataset

1. Click **"Add Data"** button (in the right panel)
2. Search for: `house-prices-advanced-regression-techniques`
3. Click **"Add"** to link the dataset to your notebook

### Step 3: Copy Notebook Cells

You have two options:

#### Option A: Use the Pre-made Notebook (Recommended)

1. Upload the notebook file: `kaggle/notebooks/run_catboost_kaggle.ipynb`
   - In Kaggle, click "File" → "Upload Notebook"
   - Select `kaggle/notebooks/run_catboost_kaggle.ipynb`

#### Option B: Copy Cells Manually

Copy each cell from `kaggle/notebooks/run_catboost_kaggle.ipynb` into your Kaggle notebook.

### Step 4: Run All Cells

1. Click **"Run All"** or run each cell sequentially
2. The notebook will:
   - Clone your repository
   - Install dependencies
   - Setup Kaggle environment
   - Verify GPU availability
   - Train CatBoost model with GPU acceleration
   - Generate submission file

### Step 5: Download Results

After training completes:

1. Go to the **"Data"** panel (right side of notebook)
2. Navigate to: `/kaggle/working/project/data/submissions/`
3. Download the latest submission CSV file
4. Submit it on the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submissions)

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

## Next Steps

After successful run:
1. Download the submission file
2. Submit to competition leaderboard
3. Compare results with local runs
4. Iterate on hyperparameters if needed

## Support

For detailed workflow documentation, see:
- `docs/HYBRID_WORKFLOW.md` - Complete workflow guide
- `docs/FLAGSHIP_LOG.md` - Architecture documentation


