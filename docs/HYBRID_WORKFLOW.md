# Hybrid Local-Kaggle GPU Workflow Guide

This guide explains how to use the hybrid workflow for developing locally while leveraging Kaggle's GPU infrastructure for intensive model training.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Workflow](#local-development-workflow)
4. [Kaggle GPU Execution](#kaggle-gpu-execution)
5. [Result Retrieval](#result-retrieval)
6. [Troubleshooting](#troubleshooting)

## Overview

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

## Prerequisites

1. **Git Repository**: Your project must be in a Git repository (GitHub, GitLab, etc.)
2. **Kaggle Account**: Free Kaggle account with GPU quota (30 hours/week)
3. **Kaggle API**: For downloading results (optional, but recommended)

### Setup Kaggle API (Optional)

```bash
# Install Kaggle API
pip install kaggle

# Configure credentials (download kaggle.json from Kaggle account settings)
mkdir -p ~/.kaggle
# Copy kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Local Development Workflow

### 1. Edit Code Locally

Work on your code as usual:
- Edit preprocessing scripts
- Modify model configurations
- Test on CPU (fast iteration)

### 2. Check Git Status

Before syncing to Kaggle, check your git status:

```bash
python scripts/sync_to_kaggle.py
```

This script will:
- Check for uncommitted changes
- Check for unpushed commits
- Provide guidance on next steps

### 3. Commit and Push

```bash
# Review changes
git status

# Add changes
git add .

# Commit
git commit -m "Your commit message"

# Push to remote
git push
```

## Kaggle GPU Execution

### Step 1: Create/Open Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook" or open existing one
3. **Enable GPU**:
   - Click "Settings" (right panel)
   - Under "Accelerator", select "GPU" (P100 or T4)
   - Click "Save"

### Step 2: Add Competition Dataset

1. Click "Add Data" button (right panel)
2. Search for "house-prices-advanced-regression-techniques"
3. Click "Add" to link the dataset

### Step 3: Use the Template Notebook

**Option A: Use Template from Repository**

1. In your Kaggle notebook, clone the repository:

```python
# Cell 1: Clone repository
!git clone https://github.com/yourusername/house-prices-starter.git /kaggle/working/project
%cd /kaggle/working/project
!pip install -r requirements.txt

# Add project to Python path
import sys
sys.path.insert(0, '/kaggle/working/project')
```

2. Copy cells from `kaggle/notebooks/kaggle_gpu_runner.ipynb` or run it directly

**Option B: Manual Setup**

Follow the cells in `kaggle/notebooks/kaggle_gpu_runner.ipynb`:

#### Cell 1: Clone Repository
```python
REPO_URL = "https://github.com/yourusername/house-prices-starter.git"
PROJECT_DIR = "/kaggle/working/project"

!git clone {REPO_URL} {PROJECT_DIR}
%cd {PROJECT_DIR}
!pip install -r requirements.txt

import sys
sys.path.insert(0, PROJECT_DIR)
```

#### Cell 2: Setup Environment
```python
from kaggle.remote.setup_kaggle import setup_kaggle_environment
setup_kaggle_environment()
```

#### Cell 3: Verify GPU
```python
!nvidia-smi

from kaggle.remote.gpu_runner import verify_gpu_setup
verify_gpu_setup()
```

#### Cell 4: Run Preprocessing (Optional)
```python
# Uncomment if you need to regenerate preprocessed data
# %run notebooks/preprocessing/run_preprocessing.py
```

#### Cell 5: Run Model Training
```python
# Choose your model (GPU will be used automatically if available)
%run notebooks/Models/9catBoostModel.py  # CatBoost with GPU
# %run notebooks/Models/7XGBoostModel.py  # XGBoost with GPU
# %run notebooks/Models/8lightGbmModel.py  # LightGBM with GPU
```

#### Cell 6: Verify Outputs
```python
# Check generated files
!ls -la /kaggle/working/project/data/submissions/
!ls -la /kaggle/working/project/runs/
```

### Step 4: Run the Notebook

1. Run cells sequentially (or "Run All")
2. Monitor GPU usage in notebook output
3. Wait for training to complete

## Result Retrieval

### Method 1: Download from Kaggle Notebook

1. In Kaggle notebook, go to "Data" tab (right panel)
2. Navigate to `/kaggle/working/project/data/submissions/`
3. Download submission CSV files
4. Save to local `data/submissions/` directory

### Method 2: Use Kaggle API

```bash
# Download outputs from Kaggle notebook
# (requires notebook output to be saved/published)

# Or use kaggle API to download competition submissions
kaggle competitions submissions -c house-prices-advanced-regression-techniques
```

### Method 3: Commit Results to Git (Advanced)

If you want to track results in Git:

```python
# In Kaggle notebook, after training:
!cd /kaggle/working/project
!git config user.email "your.email@example.com"
!git config user.name "Your Name"
!git add data/submissions/*.csv runs/*.csv
!git commit -m "Add GPU training results"
!git push
```

## Environment Detection

The system automatically detects whether it's running locally or on Kaggle:

- **Local**: Uses standard paths (`data/`, `runs/`, etc.)
- **Kaggle**: Automatically adjusts paths:
  - Data input: `/kaggle/input/house-prices-advanced-regression-techniques/`
  - Working directory: `/kaggle/working/project/`
  - Outputs: `/kaggle/working/project/data/submissions/`

### Path Mapping

| Local Path | Kaggle Path |
|------------|-------------|
| `data/raw/train.csv` | `/kaggle/input/house-prices-advanced-regression-techniques/train.csv` |
| `data/interim/train/train_process8.csv` | `/kaggle/working/project/data/interim/train/train_process8.csv` |
| `data/submissions/` | `/kaggle/working/project/data/submissions/` |
| `runs/model_performance.csv` | `/kaggle/working/project/runs/model_performance.csv` |

## GPU Detection and Usage

Models automatically detect and use GPU when available:

- **XGBoost**: Uses `tree_method='gpu_hist'` on GPU, `tree_method='hist'` on CPU
- **CatBoost**: Uses `task_type='GPU'` on GPU, `task_type='CPU'` on CPU
- **LightGBM**: Uses `device='gpu'` on GPU, `device='cpu'` on CPU

GPU detection methods (in order):
1. `nvidia-smi` command
2. `CUDA_VISIBLE_DEVICES` environment variable
3. PyTorch CUDA availability
4. TensorFlow GPU availability

## Troubleshooting

### GPU Not Detected

**Problem**: GPU not available in Kaggle notebook

**Solutions**:
1. Check that GPU accelerator is enabled in notebook settings
2. Verify GPU in notebook: `!nvidia-smi`
3. Models will automatically fall back to CPU if GPU unavailable

### Import Errors in Kaggle

**Problem**: Module not found errors

**Solutions**:
1. Ensure you've installed dependencies: `!pip install -r requirements.txt`
2. Check Python path: `import sys; print(sys.path)`
3. Verify project is cloned to correct location

### Path Not Found Errors

**Problem**: File not found errors on Kaggle

**Solutions**:
1. Run `setup_kaggle_environment()` to create symlinks
2. Verify competition dataset is added to notebook
3. Check that data files exist: `!ls /kaggle/input/house-prices-advanced-regression-techniques/`

### Git Clone Fails

**Problem**: Cannot clone repository in Kaggle notebook

**Solutions**:
1. Verify repository URL is correct
2. Ensure repository is public (or configure Git credentials for private repos)
3. Check internet connection in notebook

### Out of GPU Quota

**Problem**: Cannot enable GPU in Kaggle notebook

**Solutions**:
1. Check your GPU quota: Kaggle provides 30 hours/week
2. Wait for quota reset (Friday 8pm EDT / 7pm EST)
3. Use CPU training as fallback (models automatically detect and adjust)

## Best Practices

1. **Commit Frequently**: Commit and push changes regularly for easy sync
2. **Test Locally First**: Test code on CPU locally before running on Kaggle GPU
3. **Monitor GPU Usage**: Check GPU utilization during training
4. **Save Important Outputs**: Download or commit results from Kaggle
5. **Use Descriptive Commits**: Clear commit messages help track changes
6. **Version Control Results**: Consider committing model performance logs

## Quick Reference

### Local → Kaggle Workflow

```bash
# 1. Check status
python scripts/sync_to_kaggle.py

# 2. Commit and push
git add .
git commit -m "Your changes"
git push

# 3. Open Kaggle notebook and run cells
```

### Kaggle → Local Workflow

```bash
# 1. Download submission files from Kaggle notebook
# 2. Save to local data/submissions/
# 3. Submit using existing scripts
python scripts/submit_model.py catboost
```

## Additional Resources

- **Template Notebook**: `kaggle/notebooks/kaggle_gpu_runner.ipynb`
- **Setup Scripts**: `kaggle/remote/setup_kaggle.py`, `kaggle/remote/gpu_runner.py`
- **Environment Detection**: `config_local/environment.py`
- **Sync Helper**: `scripts/sync_to_kaggle.py`
- **Architecture Documentation**: See Section 5.5 in `docs/FLAGSHIP_LOG.md`

