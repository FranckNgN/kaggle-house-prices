# Scripts Directory

Core scripts for training, analysis, and preprocessing.

## Main Scripts

### Training
- **`train.py`** - Train all or specific models
  ```bash
  python scripts/train.py                    # Train all models
  python scripts/train.py --models catboost  # Train specific models
  python scripts/train.py --preprocess       # Run preprocessing first
  ```

### Analysis
- **`analyze.py`** - All analysis functionality
  ```bash
  python scripts/analyze.py performance      # Show performance
  python scripts/analyze.py compare          # Compare predictions
  python scripts/analyze.py errors catboost  # Analyze errors
  python scripts/analyze.py best             # Best models analysis
  python scripts/analyze.py hyperparameters  # Hyperparameter analysis
  ```

### Preprocessing
- **`run_raw_preprocessing.py`** - Run preprocessing without feature engineering
  ```bash
  python scripts/run_raw_preprocessing.py
  ```

## Kaggle Operations

All Kaggle operations are in the `kaggle/` folder:

```bash
python -m kaggle.submit catboost    # Submit model
python -m kaggle.scores status      # Check scores
python -m kaggle.sync               # Git sync
```

See main README.md for complete command reference.

## Utility Scripts

- `show_performance.py` - Performance display utilities (used by analyze.py)
- `compare_models.py` - Model comparison utilities (used by analyze.py)
- `analyze_model_errors.py` - Error analysis utilities (used by analyze.py)
- `run_with_venv.py` - Virtual environment wrapper

## Virtual Environment

The project automatically uses `.venv` if available. Setup once:

```bash
# Windows
.\scripts\setup\setup_venv.ps1

# Linux/Mac
./scripts/setup/setup_venv.sh
```
