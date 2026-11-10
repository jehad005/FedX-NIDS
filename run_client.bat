@echo off
title SecureFed-NIDS Edge Client (Port 5000)
echo ========================================================
echo         Starting SecureFed-NIDS Edge Client Node
echo ========================================================
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    echo Activating Virtual Environment...
    call venv\Scripts\activate.bat
)

set PYTHONPATH=%~dp0;%~dp0Client Server;%PYTHONPATH%
cd /d "%~dp0Client Server"
python client_app.py
pause
