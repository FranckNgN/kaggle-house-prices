# Setup script to create and configure virtual environment
# Run this once: .\setup_venv.ps1

$PROJECT_ROOT = $PSScriptRoot
$VENV_PATH = Join-Path $PROJECT_ROOT ".venv"

Write-Host "=" * 70
Write-Host "SETTING UP VIRTUAL ENVIRONMENT"
Write-Host "=" * 70

# Check if venv exists
if (Test-Path $VENV_PATH) {
    Write-Host "[INFO] Virtual environment already exists at $VENV_PATH" -ForegroundColor Green
} else {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VENV_PATH
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Virtual environment created!" -ForegroundColor Green
}

# Activate venv
$ACTIVATE_SCRIPT = Join-Path $VENV_PATH "Scripts\Activate.ps1"
if (Test-Path $ACTIVATE_SCRIPT) {
    & $ACTIVATE_SCRIPT
} else {
    Write-Host "[ERROR] Could not find activation script" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "[INFO] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To activate the virtual environment in the future, run:"
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

