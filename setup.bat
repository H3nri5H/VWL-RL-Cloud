@echo off
REM Windows Setup Script für VWL-RL-Cloud

echo ========================================
echo VWL-RL-Cloud Setup (Windows)
echo ========================================
echo.

REM Check ob Python 3.11 installiert ist
echo Checking Python 3.11...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [FEHLER] Python 3.11 nicht gefunden!
    echo.
    echo Bitte installiere Python 3.11:
    echo 1. Öffne: https://www.python.org/downloads/release/python-3119/
    echo 2. Download: "Windows installer (64-bit)"
    echo 3. Installiere mit "Add python.exe to PATH" aktiviert
    echo.
    echo Alternativ über winget:
    echo    winget install -e --id Python.Python.3.11
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.11 gefunden
echo.

REM Create venv
echo Creating virtual environment...
py -3.11 -m venv .venv

REM Activate venv
echo Activating venv...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies (dauert ~5 Minuten)...
pip install -r requirements.txt

REM Set PYTHONPATH
set PYTHONPATH=%cd%

REM Test installation
echo.
echo Testing RLlib...
python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; print('✅ RLlib ready:', ray.__version__)"

if %errorlevel% neq 0 (
    echo [FEHLER] RLlib Test fehlgeschlagen
    pause
    exit /b 1
)

REM Run environment tests
echo.
echo Running tests...
python tests\test_env.py

echo.
echo ========================================
echo Setup erfolgreich!
echo ========================================
echo.
echo Nächste Schritte:
echo   1. Frontend: streamlit run frontend/app.py
echo   2. Training: python train/train_single.py
echo.
echo VS Code: Öffne Ordner und wähle .venv als Interpreter
echo ========================================
echo.
pause
