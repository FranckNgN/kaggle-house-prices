@echo off
REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Go to project root (3 levels up from notebooks/preprocessing/)
cd /d "%SCRIPT_DIR%..\.."

REM Activate venv
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Run the preprocessing script
python notebooks\preprocessing\run_preprocessing.py

echo.
echo ==============================================
echo   Preprocessing finished.
echo ==============================================
pause >nul
