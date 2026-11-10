@echo off
title SecureFed-NIDS HQ Server & Dashboard
echo ========================================================
echo    Starting SecureFed-NIDS HQ Server and Dashboard
echo ========================================================
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    echo Activating Virtual Environment...
    call venv\Scripts\activate.bat
)

python start_hq.py
pause
