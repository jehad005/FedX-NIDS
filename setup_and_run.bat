@echo off
setlocal enabledelayedexpansion
title SecureFed-NIDS Windows Setup and Launcher
echo ========================================================
echo        SecureFed-NIDS Windows Setup and Launcher
echo ========================================================

cd /d "%~dp0"
set PROJECT_ROOT=%~dp0
set VENV_PATH=%PROJECT_ROOT%venv
set PYTHON_CMD=python

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.9+ from https://www.python.org/ and ensure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

set MODE=%1
if "%MODE%"=="" set MODE=all

if "%MODE%"=="setup" goto do_setup
if "%MODE%"=="pipeline" goto do_pipeline
if "%MODE%"=="hq" goto do_hq
if "%MODE%"=="client" goto do_client
if "%MODE%"=="all" goto do_all

echo [ERROR] Unknown mode: %MODE%
echo Usage: setup_and_run.bat [setup^|pipeline^|hq^|client^|all]
pause
exit /b 1

:do_setup
call :check_python
call :setup_venv
call :install_dependencies
echo [SUCCESS] Setup completed successfully!
if "%MODE%"=="setup" pause
goto :eof

:do_pipeline
call :check_python
if not exist "%VENV_PATH%\Scripts\python.exe" call :setup_venv
echo [INFO] Running 7-stage training pipeline...
"%VENV_PATH%\Scripts\python.exe" main.py --mode all --epochs 3
echo [SUCCESS] Pipeline completed!
if "%MODE%"=="pipeline" pause
goto :eof

:do_hq
echo [INFO] Starting SecureFed HQ API and Dashboard...
call run_hq.bat
goto :eof

:do_client
echo [INFO] Starting SecureFed Edge Client Node...
call run_client.bat 5000
goto :eof

:do_all
call :check_python
if not exist "%VENV_PATH%\Scripts\python.exe" (
    call :setup_venv
    call :install_dependencies
)
echo ========================================================
echo Starting SecureFed HQ Server and Dashboard...
echo ========================================================
call run_hq.bat
goto :eof

:check_python
echo [INFO] Verifying Python installation...
python --version
goto :eof

:setup_venv
echo [INFO] Setting up virtual environment in %VENV_PATH%...
if not exist "%VENV_PATH%" (
    python -m venv "%VENV_PATH%"
    echo [SUCCESS] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)
goto :eof

:install_dependencies
echo [INFO] Upgrading pip and installing required dependencies...
"%VENV_PATH%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_PATH%\Scripts\pip.exe" install -r requirements.txt
echo [SUCCESS] All dependencies installed successfully.
goto :eof
