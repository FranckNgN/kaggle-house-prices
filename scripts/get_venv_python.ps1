# PowerShell script to get venv Python path
# Returns the path to Python in .venv, or creates venv if it doesn't exist

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VENV_PATH = Join-Path $PROJECT_ROOT ".venv"

function Get-VenvPython {
    if (Test-Path $VENV_PATH) {
        $pythonExe = Join-Path $VENV_PATH "Scripts\python.exe"
        if (Test-Path $pythonExe) {
            return $pythonExe
        }
    }
    return $null
}

function New-Venv {
    Write-Host "[INFO] Creating virtual environment at $VENV_PATH..." -ForegroundColor Yellow
    python -m venv $VENV_PATH
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Virtual environment created!" -ForegroundColor Green
        return Get-VenvPython
    } else {
        Write-Host "[ERROR] Failed to create venv" -ForegroundColor Red
        return $null
    }
}

$venvPython = Get-VenvPython

if (-not $venvPython) {
    $venvPython = New-Venv
}

if ($venvPython) {
    Write-Output $venvPython
} else {
    # Fallback to system Python
    Write-Output (Get-Command python).Source
}

