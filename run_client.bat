@echo off
title SecureFed-NIDS Edge Client Node
echo ========================================================
echo         Starting SecureFed-NIDS Edge Client Node
echo ========================================================
cd /d "%~dp0"

set PYTHON_BIN=python
if exist venv\Scripts\python.exe (
    set PYTHON_BIN="%~dp0venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set PYTHON_BIN="%~dp0..\venv\Scripts\python.exe"
)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

set PYTHONPATH=%~dp0;%~dp0Client Server;%PYTHONPATH%
cd /d "%~dp0Client Server"
%PYTHON_BIN% client_app.py %*
pause
