@echo off
REM Development Environment Setup Script for HMM Research Environment (Windows)
REM This script sets up the complete research environment with all dependencies

echo ==========================================
echo IMP Research Environment Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed or not in PATH
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo [X] Python version is not compatible. Required: >= 3.9
    exit /b 1
)
echo [+] Python version is compatible

REM Check if virtual environment exists
if not exist ".venv" (
    echo.
    echo Creating virtual environment...
    python -m venv .venv
    echo [+] Virtual environment created
) else (
    echo [!] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo [+] Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo [+] pip upgraded

REM Install package with all dependencies
echo.
echo Installing IMP package with research dependencies...
pip install -e ".[dev,optimization,research]"
echo [+] Package installed with all dependencies

REM Setup Jupyter kernel
echo.
echo Setting up Jupyter kernel...
python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"
echo [+] Jupyter kernel 'imp-research' installed

REM Install Jupyter extensions
echo.
echo Installing Jupyter extensions...
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build 2>nul
jupyter lab build --minimize=False 2>nul
echo [+] Jupyter extensions configured

REM Validate installation
echo.
echo Validating installation...
python -c "import sys; import importlib.util; required_packages = ['numpy', 'pandas', 'polars', 'sklearn', 'hmmlearn', 'pomegranate', 'jupyter', 'ipywidgets', 'matplotlib', 'seaborn', 'plotly', 'pydantic', 'pytest']; missing = []; [missing.append(pkg) for pkg in required_packages if importlib.util.find_spec(pkg) is None]; sys.exit(1) if missing else print('All required packages are installed')"
if errorlevel 1 (
    echo [X] Some dependencies are missing
    exit /b 1
)
echo [+] All dependencies validated successfully

REM Create necessary directories
echo.
echo Creating directory structure...
if not exist "notebooks\utils" mkdir notebooks\utils
if not exist "notebooks\processed_data" mkdir notebooks\processed_data
if not exist "notebooks\model_comparison_results" mkdir notebooks\model_comparison_results
if not exist "notebooks\regime_analysis_results" mkdir notebooks\regime_analysis_results
if not exist "processed_data" mkdir processed_data
if not exist "temp_configs" mkdir temp_configs
echo [+] Directory structure created

REM Run environment validation
echo.
echo Running comprehensive environment validation...
python -m imp.utils.env_validator
echo [+] Environment validation completed

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Activate the environment: .venv\Scripts\activate.bat
echo   2. Start Jupyter Lab: jupyter lab
echo   3. Select kernel: 'IMP Research Environment'
echo   4. Open notebooks in notebooks\ directory
echo.
echo For more information, see py\docs\DEVELOPMENT_SETUP.md
echo.

pause
