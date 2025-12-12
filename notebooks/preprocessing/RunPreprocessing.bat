@echo off
REM Go to project root
cd /d D:\Project\Kaggle\house-prices-starter

REM Activate venv
call .venv\Scripts\activate.bat

REM Run the preprocessing script
python notebooks\preprocessing\run_preprocessing.py

echo.
echo ==============================================
echo   Preprocessing finished.
echo ==============================================
pause >nul
