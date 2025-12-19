# Scripts Directory

This directory contains utility scripts for running models, submitting to Kaggle, and managing the project.

## Virtual Environment Usage

All scripts are designed to run in a virtual environment. The system will automatically use the venv Python interpreter when available.

### Setup (One-time)

**Windows (PowerShell):**
```powershell
.\scripts\setup\setup_venv.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/setup/setup_venv.sh
./scripts/setup/setup_venv.sh
```

### Automatic venv Detection

When running Python scripts, the system automatically:
1. Checks for `.venv` directory
2. Uses `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac)
3. Falls back to system Python if venv not found

### Manual Activation

If you want to manually activate the venv:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

## Available Scripts

### Model Execution

- **`run_all_models_parallel.py`**: Run multiple models in parallel
  ```bash
  python scripts/run_all_models_parallel.py
  ```

### Kaggle Submission

- **`submit_to_kaggle.py`**: Submit a single model to Kaggle
  ```bash
  python scripts/submit_to_kaggle.py data/submissions/model.csv "Message"
  ```

- **`submit_all_models_auto.py`**: Submit all available models (with duplicate detection)
  ```bash
  python scripts/submit_all_models_auto.py
  ```

- **`get_kaggle_score.py`**: Retrieve and log latest Kaggle score
  ```bash
  python scripts/get_kaggle_score.py xgboost
  ```

- **`check_submission_status.py`**: View submission history and status
  ```bash
  python scripts/check_submission_status.py
  ```

### Analysis & Comparison

- **`show_performance.py`**: Display model performance metrics
- **`compare_models.py`**: Compare different model predictions
- **`check_model_progress.py`**: Check which models have been run

## Notes

- All scripts automatically use the virtual environment when available
- PYTHONPATH is automatically set to include the project root
- Scripts check for duplicate submissions to optimize Kaggle daily limit (10/day)
