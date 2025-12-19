# Virtual Environment Usage

This project is configured to **automatically use the virtual environment** for all Python commands. You don't need to manually activate it.

## How It Works

When running Python scripts, the system automatically:
1. Checks for `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac)
2. Uses venv Python if found
3. Falls back to system Python if venv doesn't exist

## Setup (One-time)

Run the setup script to create and configure the virtual environment:

**Windows:**
```powershell
.\setup_venv.ps1
```

**Linux/Mac:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

This will:
- Create `.venv` directory if it doesn't exist
- Install all dependencies from `requirements.txt`
- Configure the environment

## Running Scripts

All Python scripts automatically use the venv. Just run them normally:

```bash
# These automatically use venv Python
python scripts/submit_to_kaggle.py data/submissions/model.csv
python scripts/run_all_models_parallel.py
python scripts/get_kaggle_score.py xgboost
```

## Manual Activation (Optional)

If you want to manually activate the venv for interactive use:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

## Verification

To verify you're using venv Python:
```bash
python -c "import sys; print(sys.executable)"
```

This should show a path like:
- Windows: `D:\Project\Kaggle\house-prices-starter\.venv\Scripts\python.exe`
- Linux/Mac: `/path/to/project/.venv/bin/python`

## Troubleshooting

**If scripts don't use venv:**
1. Ensure `.venv` directory exists
2. Run `.\setup_venv.ps1` (Windows) or `./setup_venv.sh` (Linux/Mac)
3. Verify `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac) exists

**If dependencies are missing:**
```bash
# Activate venv manually, then:
pip install -r requirements.txt
```

