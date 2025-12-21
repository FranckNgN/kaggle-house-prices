# Kaggle Pipeline Explanation

## How It Currently Works

The Kaggle pipeline **directly runs your model scripts** from the `notebooks/Models/` folder on Kaggle's servers.

### Flow Diagram

```
Kaggle Notebook (run_catboost_kaggle.ipynb)
    |
    |--- Cell 1: Clone GitHub repository
    |         └─> Downloads your entire project code
    |
    |--- Cell 2: Setup environment
    |         └─> Creates symlinks, sets paths
    |
    |--- Cell 3: Verify GPU
    |         └─> Confirms GPU is available
    |
    |--- Cell 4: Run Model Script
    |         └─> %run notebooks/Models/9catBoostModel.py
    |             |
    |             └─> Executes YOUR actual model code
    |                 - Loads data from local_config
    |                 - Runs Optuna optimization
    |                 - Trains CatBoost model
    |                 - Generates submission file
    |
    |--- Cell 5: Verify outputs
    |         └─> Shows results and submission files
```

## Key Point: `%run` Magic Command

When you use `%run notebooks/Models/9catBoostModel.py` in a Jupyter/Kaggle notebook:

1. **It executes the exact same Python script** that's in your repository
2. **All code runs in the notebook's namespace** - same as if you copy-pasted it
3. **All imports work** - because the project root is in `sys.path`
4. **All your functions, configs, and logic execute** exactly as they do locally

### Example from Notebook (Cell 4):

```python
# Cell 4: Run CatBoost model training (GPU-accelerated)
%run notebooks/Models/9catBoostModel.py
```

This executes:
```python
# From notebooks/Models/9catBoostModel.py
if __name__ == "__main__":
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)  # ← Your code
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)     # ← Your code
    
    cfg = model_config.CATBOOST                           # ← Your config
    gpu_params = get_gpu_params_for_model("catboost")     # ← Your function
    
    best_params, best_rmse = run_optuna_study(...)        # ← Your optimization
    # ... rest of your model code ...
```

## What Happens Step-by-Step

### 1. Repository Clone (Cell 1)
```python
!git clone https://github.com/FranckNgN/kaggle-house-prices.git /kaggle/working/project
```
- Downloads your entire project codebase
- Includes all `notebooks/Models/*.py` files
- Includes all `config_local/`, `utils/`, `kaggle/` modules

### 2. Environment Setup (Cell 2)
```python
from kaggle.remote.setup_kaggle import setup_kaggle_environment
setup_kaggle_environment()
```
- Detects Kaggle environment
- Creates symlinks from `/kaggle/input/` to `data/raw/`
- Sets up paths for Kaggle's directory structure

### 3. GPU Verification (Cell 3)
- Checks if GPU is available
- Shows GPU info (Tesla P100/T4)

### 4. Model Execution (Cell 4) ⭐ **THIS IS THE KEY CELL**
```python
%run notebooks/Models/9catBoostModel.py
```

**What `%run` does:**
- Reads `notebooks/Models/9catBoostModel.py` from the cloned repository
- Executes it **line by line** in the notebook's Python kernel
- All imports resolve correctly (because `PROJECT_DIR` is in `sys.path`)
- All your code runs **exactly as written**
- Uses GPU if available (via `get_gpu_params_for_model()`)
- Saves outputs to `/kaggle/working/project/data/submissions/`

### 5. Results (Cell 5)
- Shows submission files
- Displays model performance metrics
- Provides download instructions

## Why This Works

1. **Same Code, Different Environment**: 
   - Your model script (`9catBoostModel.py`) is **unchanged**
   - It runs the same on Kaggle as it does locally
   - Only the **environment** (paths, GPU) differs

2. **Environment Detection**:
   - `config_local.environment.is_kaggle_environment()` detects Kaggle
   - Paths automatically adjust (`/kaggle/input/` vs local `data/raw/`)
   - GPU detection automatically enables GPU parameters

3. **No Code Duplication**:
   - You maintain **one source of truth** in `notebooks/Models/`
   - Kaggle notebook is just a **wrapper** that sets up environment
   - Model logic stays in your Python scripts

## Current Implementation Status

✅ **Working Correctly**: The pipeline runs your actual model code from `notebooks/Models/`

The Kaggle notebook:
- ✅ Clones your repository
- ✅ Sets up environment  
- ✅ Executes `notebooks/Models/9catBoostModel.py` using `%run`
- ✅ Uses GPU if available
- ✅ Generates submission files using your code

## To Run Different Models

Simply change Cell 4 to run a different model:

```python
# CatBoost
%run notebooks/Models/9catBoostModel.py

# XGBoost
%run notebooks/Models/7XGBoostModel.py

# LightGBM
%run notebooks/Models/8lightGbmModel.py

# Any other model from notebooks/Models/
%run notebooks/Models/10blendingModel.py
```

## Summary

**Your model code in `notebooks/Models/` IS running on Kaggle server.**

The Kaggle notebook is just a setup wrapper that:
1. Gets your code onto Kaggle (git clone)
2. Sets up the environment (paths, GPU)
3. Executes your model script (`%run`)
4. Shows results

You don't need separate Kaggle-specific model code - the same scripts work everywhere!

