@echo off
title SecureFed-NIDS HQ Server & Dashboard
echo ========================================================
echo    Starting SecureFed-NIDS HQ Server and Dashboard
echo ========================================================
cd /d "%~dp0"

set PYTHON_BIN=python
if exist venv\Scripts\python.exe (
    set PYTHON_BIN="%~dp0venv\Scripts\python.exe"
)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

%PYTHON_BIN% start_hq.py %*
pause
